"""
Memory-retained hybrid RAG GPT model queried through OpenAI API
"""

import os
import warnings
from tqdm import tqdm
from LlaMa.lama import OpenRouterLLM
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import SystemMessage, HumanMessage
from Expert_RAG.utils import *
from Expert_RAG.hybrid_memory_GPT import wipe_RAG
from Expert_RAG.rag_context import get_rag_context
import logging
import time
warnings.filterwarnings("ignore")  # Suppress warnings
os.environ["OPENAI_API_KEY"] = "YOUR KEY HERE"

# Define paths and settings
persist_directory = "DATABASE DIRECTORY" # Vector store path

# Initialize vector store and embeddings
embeddings = OpenAIEmbeddings()
if os.path.exists(persist_directory):
    print("Reusing existing combined database...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    raise FileNotFoundError(f"Vector store not found at {persist_directory}. Ensure data is preprocessed and saved.")


# Innitialize Llama
OPENROUTER_API_KEY = constants.OPEN_ROUTER

model_name = "meta-llama/llama-3-8b-instruct"
# Initialize the custom LLM
llm_model = OpenRouterLLM(
    api_key=OPENROUTER_API_KEY,
    model=model_name,
    top_p=0.9,
    temperature=0.9,
    repetition_penalty=1.0
)

# import custom prompt
prompt_dir = "prompts/prompt_file.txt"
# prompt_dir = "prompts/lung_prompt.txt"

# plain method
def hybrid_chain_open(
    category, genenames, prompt_dir="prompts/prompt_file.txt", save_dir="LLM_scores/debug", vectorstore=vectorstore,
    chat=llm_model, batch_size=30, n_trials=1, save_retrieved=False, final_batch=False, wipe = False,
    summarized_gene_docs=False, filtered_cancer_docs=False, pubmed_docs=False, original_docs=False,
    original_rag_k=3, memory_size=200, small=False
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
        save_dir (str): Directory to save the trial scores and results.
        wipe (bool): If True, wipe all files in save_dir before starting.

    Returns:
        tuple: Contains results and final scores.
    """
    if wipe:
        logging.info("Wiping save directory before starting.")
        print("Wiping save directory before starting.")
        wipe_RAG(save_dir)

    total_genes = len(genenames)
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
        logging.info(f"Starting trial {trial + 1}/{n_trials}")
        batch_scores = []

        for start_idx in tqdm(range(0, total_genes, batch_size), desc="Processing..."):
            end_idx = min(start_idx + batch_size, total_genes)
            batch_genes = genenames[start_idx:end_idx]

            # Construct query for the batch
            query = create_general_prompt(prompt_dir, category, batch_genes)

            context = context = get_rag_context(
                batch_genes, category, vectorstore, chat, "o1",
                pubmed_docs=pubmed_docs, filtered_cancer_docs=filtered_cancer_docs,
                summarized_gene_docs=summarized_gene_docs, original_docs=original_docs,
                orig_doc_k=original_rag_k, small=small
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
            memory = ConversationSummaryBufferMemory(llm=chat, max_token_limit=memory_size) # scale this accordingly for smaller mode, llama-3 only has input token limit of 1024
            memory_context = memory.load_memory_variables({})
            full_context = memory_context.get("history", "")

            if context != "":
                # Document-grounded response
                full_prompt = f"{full_context}\n\nUsing the following context, provide the most accurate and relevant answer to the question. " \
                              f"Prioritize the provided context, but if the context does not contain enough information to fully address the question, " \
                              f"use your best general knowledge to complete the answer:\n\n{context}\n\nQuestion: {query}"
            else:
                # Fallback to general knowledge
                full_prompt = f"{full_context}\n\nUsing your best general knowledge, provide the most accurate and relevant answer to the question:\n\nQuestion: {query}"

            messages = [
                SystemMessage(content="You are an expert assistant with access to gene and cancer knowledge."),
                HumanMessage(content=full_prompt)
            ]

            serialized_prompt = "\n".join([f"{msg.content}" for msg in messages])

            # Query OpenRouter
            response = chat(serialized_prompt)
            results.append(response)

            # Append batch scores
            batch_scores_partial = extract_scores_from_responses(
                response if isinstance(response, list) else [response],
                batch_genes
            )

            # Retry logic for score validation
            while len([score for score in batch_scores_partial if score is not None]) != len(batch_genes):
                print(response)
                try:
                    logging.warning(f"Batch scores count mismatch for genes {batch_genes}. Retrying...")

                    # Regenerate the query prompt
                    query = create_general_prompt(prompt_dir, category, batch_genes)
                    messages = [
                        SystemMessage(content="You are an expert assistant with access to gene and cancer knowledge."),
                        HumanMessage(content=full_prompt)
                    ]

                    # Retry querying the model
                    serialized_prompt = "\n".join([f"{msg.content}" for msg in messages])

                    # Query OpenRouter
                    response = chat(serialized_prompt)

                    # Extract scores again
                    batch_scores_partial = extract_scores_from_responses(
                        response if isinstance(response, list) else [response],
                        batch_genes
                    )

                    # Optional delay to handle rate limits or avoid spamming
                    time.sleep(0.1)

                except Exception as e:
                    logging.error(f"Error during retry: {str(e)}. Continuing retry...")

            logging.info(f"Successfully retrieved valid scores for batch: {batch_genes}")
            batch_scores.extend(batch_scores_partial)

            # Save to memory
            memory.save_context({"input": query}, {"output": response})

            time.sleep(0.1)  # Short delay to avoid rate limits

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
        final_scores = [
            sum(scores) / len(scores) for scores in zip(*[trial["scores"] for trial in trial_scores])
        ]
    else:
        final_scores = []

    logging.info(f"Final scores vector (averaged across trials) calculated with length: {len(final_scores)}")

    # Save results and trial scores
    os.makedirs(save_dir, exist_ok=True)
    trial_scores_file = os.path.join(save_dir, "trial_scores_RAG.json")

    with open(trial_scores_file, "w") as json_file:
        json.dump(trial_scores, json_file, indent=4)

    print(f"Trial scores saved to {trial_scores_file}")

    return results, final_scores



# Tests
if __name__ == "__main__":
    genenames_unique = ["AASS", "ABCA6", "ABCB1", "ABHD6", "ABHD8", "ABRACL", "ABTB2"]
    category = "Diffuse large B-cell lymphoma (DLBCL)"
    # print created prompt
    results, final_scores = hybrid_chain_open(category = category, genenames = genenames_unique, batch_size = 3)
    # Combine and display results
    for idx, batch_result in enumerate(results):
        print(f"Batch {idx + 1} Results:\n{batch_result}\n")

    print(len(final_scores))

    scores = extract_scores_from_responses(results,genenames_unique,True)
    print(scores)














