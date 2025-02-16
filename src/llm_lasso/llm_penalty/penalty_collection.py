"""
Memory-retained hybrid RAG GPT model queried through OpenAI API
"""

import os
import warnings
from tqdm import tqdm
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import SystemMessage, HumanMessage
from Expert_RAG.utils import *
from Expert_RAG.helper import *
import logging
import time
import json
from pydantic import BaseModel
from openai import OpenAI
from Expert_RAG.omim_RAG_process import *
from Expert_RAG.pubMed_RAG_process import pubmed_retrieval
# from Expert_RAG.o1 import O1 # if one uses RAG via o1

from Expert_RAG.rag_context import get_rag_context


warnings.filterwarnings("ignore")  # Suppress warnings
os.environ["OPENAI_API_KEY"] = "YOUR KEY HERE"

# Define paths and settings
# persist_directory = "omim_scrape/omim_all/persist_omim_chunked"  # Vector store path
data_file = "DATABASE LOCATION"  # Path to JSON data file
PERSIST = True  # Enable persistence

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# create new directory using chunked_GPT!!!
persist_directory = "DATABASE_DIRECTORY" # "omim_scrape/omim_all/persist_omim_chunked" #  # Vector store path

# Initialize vector store and embeddings
if os.path.exists(persist_directory):
    print("Reusing existing combined database...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    raise FileNotFoundError(f"Vector store not found at {persist_directory}. Ensure data is preprocessed and saved.")

#retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) # Retrieve top 8 most relevant chunks
llm_model = ChatOpenAI(model="gpt-4o", temperature=0.5)

openai_client = OpenAI()


class Score(BaseModel):
    gene: str
    penalty_factor: float
    reasoning: str


class GeneScores(BaseModel):
    scores: list[Score]


def wipe_RAG(save_dir):
    files_to_remove = [
        "results_RAG.txt",
        "gene_scores_RAG.pkl",
        "gene_scores_RAG.txt",
        "trial_scores_RAG.json",
    ]
    for file_name in files_to_remove:
        file_path = os.path.join(save_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed file: {file_path}")


def hybrid_chain_GPT(
    category, genenames, prompt_dir="prompts/prompt_file.txt", save_dir="LLM_score/debug",
    vectorstore=vectorstore, chat=llm_model, batch_size=30, n_trials=1, save_retrieved=False, final_batch=False,
    wipe=False, summarized_gene_docs=False, filtered_cancer_docs=False, pubmed_docs=False, original_docs=False,
    retry_limit=10, temp=0.5, original_rag_k=3, memory_size=200, model_type="gpt4-o", reweight=False
):
    """
    Hybrid RAG chain combined with batch processing for large queries, integrating memory-enhanced context.
    This implementation calculates the final scores as an average across trials.

    Args:
        category (str): The category or context for the query.
        genenames (list[str]): List of gene names.
        prompt_dir (str): Path to the prompt file.
        vectorstore: VectorStore for retrieval.
        chat: LLM model for querying.
        batch_size (int): Number of genes to process per batch.
        n_trials (int): Number of trials to run.
        save_retrieved (bool): Whether to save retrieved documents.
        final_batch (bool): Indicator for saving the final batch of documents.
        save_dir (str): Directory to save the trial scores.
        wipe (bool): If True, wipe all files in save_dir before starting.

    Returns:
        list[dict]: Trial scores.
        list[float]: Final averaged scores.
    """
    if wipe:
        logging.info("Wiping save directory before starting.")
        print("Wiping save directory before starting.")
        wipe_RAG(save_dir)

    total_genes = len(genenames)
    print(f"Processing {total_genes} features...")
    trial_scores = []

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    trial_scores_file = os.path.join(save_dir, "trial_scores_RAG.json")

    # Load existing progress if the file already exists
    if os.path.exists(trial_scores_file):
        with open(trial_scores_file, "r") as json_file:
            trial_scores = json.load(json_file)

    # Determine which trial to start from
    start_trial = len(trial_scores)

    results = []

    trial = start_trial
    while trial < n_trials:
        idxs = np.arange(len(genenames))
        # np.random.shuffle(idxs)

        logging.info(f"Starting trial {trial + 1}/{n_trials}")
        batch_scores = []

        best = []

        for start_idx in tqdm(range(0, total_genes, batch_size), desc=f"Processing trial {trial + 1}..."):
            end_idx = min(start_idx + batch_size, total_genes)
            batch_genes = [genenames[i] for i in idxs[start_idx:end_idx]]

            # Construct query for the batch
            query = create_general_prompt(prompt_dir, category, batch_genes)

            # get documents for RAG
            context = get_rag_context(
                batch_genes, category, vectorstore, chat, model_type,
                pubmed_docs=pubmed_docs, filtered_cancer_docs=filtered_cancer_docs,
                summarized_gene_docs=summarized_gene_docs, original_docs=original_docs,
                orig_doc_k=original_rag_k
            )
            
            # if save_retrieved:
            #     output_dir = f'retrieval_docs/test/RAG/final_batch' if final_batch else f'retrieval_docs/test/RAG/batch{start_idx + 1}'
            #     os.makedirs(output_dir, exist_ok=True)
            #     for idx, doc in enumerate(unique_docs):
            #         file_name = f"{'weights_doc' if final_batch else 'doc'}_{idx + 1}.txt"
            #         file_path = os.path.join(output_dir, file_name)
            #         with open(file_path, "w", encoding="utf-8") as file:
            #             file.write(doc.page_content)  # Save document content
            #             file.write("\n\nMetadata:\n")
            #             file.write(str(doc.metadata))

            # Retrieve memory context
            if model_type != "o1":
                memory = ConversationSummaryBufferMemory(llm=chat, max_token_limit=memory_size)
                memory_context = memory.load_memory_variables({})
                full_context = memory_context.get("history", "")
            else:
                full_context=""

            if context.strip() != "":
                full_prompt = f"{full_context}\n\nUsing the following context, provide the most accurate and relevant answer to the question. " \
                              f"Prioritize the provided context, but if the context does not contain enough information to fully address the question, " \
                              f"use your best general knowledge to complete the answer:\n\n{context}\n\nQuestion: {query}"

            else:
                # Fallback to general knowledge
                full_prompt = f"{full_context}\n\nUsing your best general knowledge, provide the most accurate and relevant answer to the question:\n\nQuestion: {query}"

            messages = [
                {"role": "system", "content": "You are an expert assistant with access to gene and cancer knowledge."},
                {"role": "user", "content": full_prompt}
            ]

            time.sleep(1)
            print("Querying GPT")
            if model_type != "o1":
                completion = openai_client.beta.chat.completions.parse(
                    model=model_type,
                    messages=messages,
                    response_format=GeneScores,
                    temperature=temp
                )
            else:
                completion = openai_client.beta.chat.completions.parse(
                    model=model_type,
                    messages=messages,
                    response_format=GeneScores,
                )
            time.sleep(1)


            gene_scores = completion.choices[0].message.parsed
            results.append(gene_scores.model_dump_json())

            upper_batch_names = [n.upper() for n in batch_genes]
            scores_list = [score for score in gene_scores.scores if score.gene.upper() in upper_batch_names]

            genes_retrieved = set([score.gene for score in scores_list])
            missing = set(batch_genes).difference(genes_retrieved)
            n_retries = 0
            while len(missing) > 0:
                print(f"We are missing genes {missing}")
                assert n_retries < retry_limit
                n_retries += 1

                # Optional delay to handle rate limits or avoid spamming
                time.sleep(1)

                if model_type != "o1":
                    completion = openai_client.beta.chat.completions.parse(
                        model=model_type,
                        messages=messages,
                        response_format=GeneScores,
                        temperature=temp
                    )
                else:
                    completion = openai_client.beta.chat.completions.parse(
                        model=model_type,
                        messages=messages,
                        response_format=GeneScores,
                    )

                gene_scores = completion.choices[0].message.parsed
                upper_batch_names = [n.upper() for n in batch_genes]
                scores_list = [score for score in gene_scores.scores if score.gene.upper() in upper_batch_names]

                genes_retrieved = set([score.gene for score in scores_list])
                missing = set(batch_genes).difference(genes_retrieved)


            genes_to_scores = {
                score.gene: score.penalty_factor for score in gene_scores.scores
            }
            batch_scores_partial = [genes_to_scores[gene] for gene in batch_genes]
            logging.info(f"Successfully retrieved valid scores for batch: {batch_genes}")
            batch_scores.append(batch_scores_partial)
            print(batch_scores_partial)

            best_pair = min(genes_to_scores.items(), key=lambda x: x[1])
            if reweight:
                print(f"Best pair: {best_pair}")
            best.append(best_pair)

            # Save to memory
            if model_type != "o1":
                memory.save_context({"input": query}, {"output": str(gene_scores)})

        weights = np.ones(len(batch_scores))
        if reweight:
            _, weights = hybrid_chain_GPT(
                category, [x[0] for x in best], prompt_dir, save_dir + "/best",vectorstore, llm_model,
                batch_size=len(best), n_trials=1, save_retrieved=False, final_batch=False,
                wipe=wipe, summarized_gene_docs=summarized_gene_docs,
                filtered_cancer_docs=filtered_cancer_docs, pubmed_docs=pubmed_docs,
                original_docs=original_docs, retry_limit=retry_limit, temp=temp,
                original_rag_k=original_rag_k, memory_size=memory_size, model_type=model_type,
                reweight=False
            )
            weights = [weight / best_item[1] for (weight, best_item) in zip(weights, best)]

        final_batch_scores = []
        for (scores, weight) in zip(batch_scores, weights):
            final_batch_scores.extend(list(np.array(scores) * weight))
            
        # Check if the trial scores match the total genes
        if len(final_batch_scores) == total_genes:
            trial_scores.append({"iteration": trial + 1, "scores": final_batch_scores})

            # Incrementally save progress after each trial
            with open(trial_scores_file, "w") as json_file:
                json.dump(trial_scores, json_file, indent=4)

            logging.info(f"Trial {trial + 1} completed and saved.")
            trial += 1
        else:
            logging.warning(f"Trial {trial + 1} scores length mismatch.")
            sys.exit(1)
        

    # Calculate final scores averaged across trials
    if trial_scores:
        final_scores = [sum(scores) / len(scores) for scores in zip(*[trial["scores"] for trial in trial_scores])]
    else:
        final_scores = []

    logging.info(f"Final scores vector (averaged across trials) calculated with length: {len(final_scores)}")

    print(f"Trial scores saved to {trial_scores_file}")

    return results, final_scores
