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
import Expert_RAG.data_processing as dp
import logging
import time
import json
from pydantic import BaseModel
from openai import OpenAI
from Expert_RAG.omim_RAG_process import *
from Expert_RAG.pubMed_RAG_process import pubmed_retrieval
from adversarial.utils import replace_top, replace_random, insert_fake_names_into_context
from baselines.llm_select import llm_score
import argparse


warnings.filterwarnings("ignore")  # Suppress warnings
os.environ["OPENAI_API_KEY"] = "YOUR KEY HERE"

# Define paths and settings
persist_directory = "DATABASE DIRECTORY"  # Vector store path
data_file = "DATABASE DIRECTORY"  # Path to JSON data file
PERSIST = True  # Enable persistence

# Initialize embeddings
embeddings = OpenAIEmbeddings()

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


def adversarial_hybrid_chain_GPT(
    category, genenames, new_genenames, prompt_dir="prompts/prompt_file.txt", save_dir="LLM_score/debug",
    vectorstore=vectorstore, chat=llm_model, batch_size=30, n_trials=1,
    max_replace=200, replace_names_in_rag=True, replace_top_genes=True, wipe=False, 
    summarized_gene_docs=False, filtered_cancer_docs=False, pubmed_docs=False, original_docs=True,
    retry_limit=10, temp=0.5
):
    if wipe:
        logging.info("Wiping save directory before starting.")
        print("Wiping save directory before starting.")
        wipe_RAG(save_dir)
    
    if not replace_names_in_rag:
        genenames = new_genenames

    total_genes = len(genenames)
    print(f"Processing {total_genes} features...")
    results = []
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


    trial = start_trial
    while trial < n_trials:
        idxs = np.arange(len(genenames))
        # np.random.shuffle(idxs)

        logging.info(f"Starting trial {trial + 1}/{n_trials}")
        batch_scores = []

        for start_idx in tqdm(range(0, total_genes, batch_size), desc=f"Processing trial {trial + 1}..."):
            end_idx = min(start_idx + batch_size, total_genes)
            batch_genes = [genenames[i] for i in idxs[start_idx:end_idx]]
            batch_new_genes = [new_genenames[i] for i in idxs[start_idx:end_idx]]

            # Construct query for the batch
            query = create_general_prompt(prompt_dir, category, batch_new_genes)

            # get documents for RAG
            context = ""
            skip_genes = set()
            if pubmed_docs:
                context += pubmed_retrieval(batch_genes, category, "gpt-4o") + "\n"
                time.sleep(1)
            if filtered_cancer_docs:
                (add_ctx, skip_genes) = get_filtered_cancer_docs_and_genes_found(
                    batch_genes, vectorstore.as_retriever(search_kwargs={"k": 100}),
                    chat, category
                )
                context += add_ctx + "\n"
                time.sleep(1)
            if summarized_gene_docs:
                preamble = "\nAdditional gene information: \n" if context.strip() != "" else ""
                context += preamble + get_summarized_gene_docs(
                    [gene for gene in batch_genes if gene not in skip_genes],
                    chat
                ) + "\n"
                time.sleep(1)
            
            if original_docs:
                assert (not pubmed_docs) and (not filtered_cancer_docs) and (not summarized_gene_docs)
                docs = retrieval_docs(batch_genes, category, vectorstore.as_retriever(search_kwargs={"k": 3}), small=False)
                unique_docs = get_unique_docs(docs)
                context = "\n".join([doc.page_content for doc in unique_docs])

            if replace_names_in_rag:
                context = insert_fake_names_into_context(batch_genes, batch_new_genes, context)
            batch_genes = batch_new_genes

            # Retrieve memory context
            memory = ConversationSummaryBufferMemory(llm=chat, max_token_limit=500)
            memory_context = memory.load_memory_variables({})
            full_context = memory_context.get("history", "")

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
            completion = openai_client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=messages,
                response_format=GeneScores,
            )
            time.sleep(1)

            gene_scores = completion.choices[0].message.parsed
            upper_batch_names = [n.upper() for n in batch_genes]
            scores_list = [score for score in gene_scores.scores if score.gene.upper() in upper_batch_names]

            results.append(gene_scores.model_dump_json())

            genes_retrieved = set([score.gene for score in scores_list])
            missing = set(batch_genes).difference(genes_retrieved)
            n_retries = 0
            while len(missing) > 0:
                print(f"We are missing genes {missing}")
                assert n_retries < retry_limit
                n_retries += 1

                # Optional delay to handle rate limits or avoid spamming
                time.sleep(0.1)

                completion = openai_client.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=messages,
                    response_format=GeneScores,
                    temperature=temp
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
            batch_scores.extend(batch_scores_partial)
            print(batch_scores_partial)

            # Save to memory
            memory.save_context({"input": query}, {"output": str(completion.choices[0].message)})

        # Check if the trial scores match the total genes
        if len(batch_scores) == total_genes:
            trial_scores.append({"iteration": trial + 1, "scores": batch_scores})

            # Incrementally save progress after each trial
            with open(trial_scores_file, "w") as json_file:
                json.dump(trial_scores, json_file, indent=4)

            logging.info(f"Trial {trial + 1} completed and saved.")
            trial += 1
        else:
            logging.warning(f"Trial {trial + 1} scores length mismatch. Retrying...")

    # Calculate final scores averaged across trials
    if trial_scores:
        final_scores = [sum(scores) / len(scores) for scores in zip(*[trial["scores"] for trial in trial_scores])]
    else:
        final_scores = []

    logging.info(f"Final scores vector (averaged across trials) calculated with length: {len(final_scores)}")

    print(f"Trial scores saved to {trial_scores_file}")

    return results, final_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG LLM-Lasso and LLM-Select with some fake gene names")
    parser.add_argument("--prompt_dir", type=str, required=True, help="Path to the prompt file.")
    parser.add_argument("--experiment", type=str, choices=["llm-select", "llm-lasso"])
    parser.add_argument("--wipe", action="store_true", help="If set, wipe the save directory before starting.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Name of the GPT model to use.")
    parser.add_argument("--genenames_path", type=str, required=True, help="Path to the gene names file (.pkl or .txt).")
    parser.add_argument("--fake_genenames_path", type=str, required=True, help="Path to the gene names file from get_new_genenames.py (.pkl or .txt).")
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size for processing.")
    parser.add_argument("--category", type=str, required=True, help="Category for the query (e.g., cancer type).")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the results and scores.")
    parser.add_argument("--LLM_type", type=str, default="GPT", choices=["GPT", "General"], help="Specify LLM type: 'GPT' for OpenAI or 'General' for OpenRouter.")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature for randomness in responses.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter.")
    parser.add_argument("--repetition_penalty", type=float, default=0.9, help="Penalty for repeated tokens.")
    # LLM-SELECT arguments
    parser.add_argument("--k_min", type=int, default=0, help="Minimum number of top genes to select.")
    parser.add_argument("--k_max", type=int, default=50, help="Maximum number of top genes to select.")
    parser.add_argument("--step", type=int, default=5, help="Step size for the range of k values.")

    args = parser.parse_args()

    # Load gene names
    if args.genenames_path.endswith(".pkl"):
        with open(args.genenames_path, 'rb') as file:
            genenames = pkl.load(file)
    elif args.genenames_path.endswith(".txt"):
        with open(args.genenames_path, 'r') as file:
            genenames = file.read().splitlines()
    else:
        raise ValueError("Unsupported file format. Use .pkl or .txt.")
    
    if args.fake_genenames_path.endswith(".pkl"):
        with open(args.fake_genenames_path, 'rb') as file:
            new_genenames = pkl.load(file)
    elif args.fake_genenames_path.endswith(".txt"):
        with open(args.fake_genenames_path, 'r') as file:
            new_genenames = file.read().splitlines()
    else:
        raise ValueError("Unsupported file format. Use .pkl or .txt.")
    print(f'Total number of features in processing: {len(genenames)}.')

    if args.experiment == "llm-lasso":
        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings()

        if os.path.exists(persist_directory):
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        else:
            raise FileNotFoundError(f"Vector store not found at {persist_directory}. Ensure data is preprocessed and saved.")

        if args.LLM_type == "GPT":
            if args.model_name == 'o1':
                raise NotImplementedError
            else:
                chat = ChatOpenAI(model=args.model_name, temperature=args.temp)
                results, all_scores = adversarial_hybrid_chain_GPT(
                    args.category, genenames, new_genenames, args.prompt_dir, args.save_dir, vectorstore,
                    chat, args.batch_size, args.n_trials, wipe = args.wipe,
                )
        elif args.LLM_type == "General":
            raise NotImplementedError
        print(f'Total number of scores collected: {len(all_scores)}.')
        # Save results
        os.makedirs(args.save_dir, exist_ok=True)
        results_file = os.path.join(args.save_dir, "results_RAG.txt")
        save_responses_to_file(results, results_file)

        scores_pkl_file = os.path.join(args.save_dir, "gene_scores_RAG.pkl")
        scores_txt_file = os.path.join(args.save_dir, "gene_scores_RAG.txt")
        save_scores_to_pkl(all_scores, scores_pkl_file)
        dp.convert_pkl_to_txt(scores_pkl_file, scores_txt_file)

        print(f"Results saved to {results_file}")
        print(f"Scores saved to {scores_pkl_file} and {scores_txt_file}")

        if len(all_scores) != len(genenames):
            raise ValueError(
                f"Mismatch between number of scores ({len(all_scores)}) and number of gene names ({len(genenames)}).")

    else: # llm-select
        top_k_dict = llm_score(
            LLM_type=args.LLM_type,
            category=args.category,
            genenames=new_genenames,
            prompt_dir=args.prompt_dir,
            save_dir=args.save_dir,
            model_name=args.model_name,
            k_min= int(args.k_min),
            k_max= int(args.k_max),
            step= int(args.step),
            temp=args.temp,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            batch_size=args.batch_size,
            n_trials=args.n_trials,
            wipe=args.wipe,
        )

        print("Top-k gene selection complete. Results saved.")