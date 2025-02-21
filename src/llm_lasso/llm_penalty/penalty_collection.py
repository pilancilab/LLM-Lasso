
from pydantic import BaseModel
from dataclasses import dataclass, field
from langchain_community.vectorstores import Chroma
from llm_lasso.llm_penalty.llm import LLMQueryWrapperWithMemory
from llm_lasso.utils.score_collection import create_general_prompt, \
    save_responses_to_file, save_scores_to_pkl
from llm_lasso.utils.data import convert_pkl_to_txt
from llm_lasso.llm_penalty.rag.rag_context import get_rag_context
from llm_lasso.llm_penalty.query_scores import query_scores_with_retries
import os
import logging
import json
import numpy as np
from tqdm import tqdm


@dataclass
class PenaltyCollectionParams:
    """
    Parameters for collecting penalty factors via LLM-Lasso
    """
    batch_size: int = field(default=30, metadata={
        "help": "Number of genes to pass into the LLM at once"})
    n_trials: int = field(default=1, metadata={
        "help": "Number of trials to average over"})
    wipe: bool = field(default=False, metadata={
        "help": "Wipe the save directory before starting"})
    retry_limit: int = field(default=10, metadata={
        "help": "Maximum number of times to retry querying the LLM if not all genes are found"
    })
    summarized_gene_doc_rag: bool = field(default=False, metadata={
        "help": "Whether to perform RAG with summarized OMIM docs for each gene"})
    filtered_cancer_doc_rag: bool = field(default=False, metadata={
        "help": "Whether to perform RAG with OMIM docs about the category, filtered for the relevant genes"
    })
    pubmed_rag: bool = field(default=False, metadata={
        "help": "Whether to perform RAG with pubmed docs"})
    omim_rag: bool = field(default=False, metadata={
        "help": "Whether to perform RAG with the default OMIM vector store"})
    omim_rag_num_docs: int = field(default=3, metadata={
        "help": "Number of documents to retrieve for `omim_rag`"})
    small: bool = field(default=False, metadata={
        "help": "For LLMs with small context sizes, reduce the amount of informatio retrieved for `omim_rag`"
    })
    enable_memory: bool = field(default=True, metadata={
        "help": "Whether to pass memory of past queries into the LLM"})
    memory_size: int = field(default=200, metadata={
        "help": "Number of tokens in the memory"})
    shuffle: bool = field(default=False, metadata={
        "help": "Whether to shuffle the feature names for each trial"})

    def has_rag(self):
        return self.summarized_gene_doc_rag or \
            self.filtered_cancer_doc_rag or \
            self.pubmed_rag or \
            self.omim_rag


def wipe_llm_penalties(save_dir, rag: bool):
    """
    Wipe save directory for LLM Lasso penalties
    """
    if rag:
        files_to_remove = [
            "results_RAG.txt",
            "fial_scores_RAG.pkl",
            "final_scores_RAG.txt",
            "trial_scores_RAG.json"
        ]
    else:
        files_to_remove = [
            "results_plain.txt",
            "fial_scores_plain.pkl",
            "final_scores_plain.txt",
            "trial_scores_plain.json",
        ]
    for file_name in files_to_remove:
        file_path = os.path.join(save_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed file: {file_path}")


def collect_penalties(
    category: str,
    feature_names: list[str],
    prompt_file: str,
    save_dir: str,
    vectorstore: Chroma,
    model: LLMQueryWrapperWithMemory,
    params: PenaltyCollectionParams,
    omim_api_key: str = "",
):
    """
    Query features in batches and extract LLM-Lasso penalties.

    Splits a list of features into batches, constructs prompts, and queries the selected LLM.
    It also maintains conversation memory.

    Parameters:
    - `category`: The category or context for the query (e.g., "cancer type").
    - `feature_names`: List of feature names,
    - `prompt_file`: Path to the prompt file used for constructing queries.
    - `save_dir`: Directory where results and scores will be saved.
    - `vectorstore`: Chroma vectorstore for OMIM RAG.
    - `model`: LLMQueryWrapperWithMemory object for performing queries
    - `params`: PenaltyCollectionParams object
    - `omim_api_key`: OMIM API key, only needed if
        `params.summarized_gene_doc_rag` is True
    """
    if params.wipe:
        logging.info("Wiping save directory before starting.")
        print("Wiping save directory before starting.")
        wipe_llm_penalties(save_dir, params.has_rag())
    
    total_features = len(feature_names)
    print(f"Processing {total_features} features...")

    rag_or_plain = "RAG" if params.has_rag() else "plain"

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    trial_scores_file = os.path.join(save_dir, f"trial_scores_{rag_or_plain}.json")

     # Load existing progress if the file already exists
    if os.path.exists(trial_scores_file):
        with open(trial_scores_file, "r") as json_file:
            trial_scores = json.load(json_file)
    else:
        trial_scores = []

    # Determine which trial to start from
    start_trial = len(trial_scores)

    results = []
    trial = start_trial

    while trial < params.n_trials:
        # maybe shuffle the feature names
        idxs = np.arange(len(feature_names))
        if params.shuffle:
            np.random.shuffle(idxs)

        logging.info(f"Starting trial {trial + 1} out of {params.n_trials}")
        batch_scores = []

        if params.enable_memory:
            model.start_memory(params.memory_size)

        # loop through batches of genes
        for start_idx in tqdm(range(0, total_features, params.batch_size), desc=f"Processing trial {trial + 1}..."):
            end_idx = min(start_idx + params.batch_size, total_features)
            batch_features = [feature_names[i] for i in idxs[start_idx:end_idx]]

            # Construct the query for this batch of features
            query = create_general_prompt(prompt_file, category, batch_features)

            # If we're performing RAG, get the RAG context
            context = get_rag_context(
                batch_features, category, vectorstore,
                model, omim_api_key,
                pubmed_docs=params.pubmed_rag,
                filtered_cancer_docs=params.filtered_cancer_doc_rag,
                summarized_gene_docs=params.summarized_gene_doc_rag,
                original_docs=params.omim_rag,
                default_num_docs=params.omim_rag_num_docs,
                small=params.small,
            )

            # Construct the prompt
            if context != "":
                full_prompt = f"Using the following context, provide the most accurate and relevant answer to the question. " \
                              f"Prioritize the provided context, but if the context does not contain enough information to fully address the question, " \
                              f"use your best general knowledge to complete the answer:\n\n{context}\n\nQuestion: {query}"
            else:
                # Fallback to general knowledge
                full_prompt = f"Using your best general knowledge, provide the most accurate and relevant answer to the question:\n\nQuestion: {query}"
            system_message = "You are an expert assistant with access to gene and cancer knowledge."

            # Query the LLM, with special handling if the LLM allows
            # structured queries
            batch_scores_partial, output = query_scores_with_retries(
                model, system_message, full_prompt,
                batch_features, params.retry_limit
            )

            logging.info(f"Successfully retrieved valid scores for batch: {batch_features}")
            batch_scores.extend(batch_scores_partial)
            logging.info(batch_scores_partial)
            model.maybe_add_to_memory(query, output)
            results.append(output)
        # end batches for loop

        if len(batch_scores) == total_features:
            trial_scores.append({"iteration": trial + 1, "scores": batch_scores})

            # Incrementally save progress after each trial
            with open(trial_scores_file, "w") as json_file:
                json.dump(trial_scores, json_file, indent=4)

            logging.info(f"Trial {trial + 1} completed and saved.")
            trial += 1
        else:
            logging.warning(f"Trial {trial + 1} scores length mismatch. Retrying...")
        # end trial success if/else
    # end trial for loop

    # Calculate final scores averaged across trials
    if trial_scores:
        final_scores = [sum(scores) / len(scores) for scores in zip(*[trial["scores"] for trial in trial_scores])]
    else:
        final_scores = []

    logging.info(f"Final scores vector (averaged across trials) calculated with length: {len(final_scores)}")

    model.disable_memory()
    print(f"Trial scores saved to {trial_scores_file}")

    # save penalties to file
    results_file = os.path.join(save_dir, f"results_{rag_or_plain}.txt")
    save_responses_to_file(results, results_file)

    scores_pkl_file = os.path.join(save_dir, f"final_scores_{rag_or_plain}.pkl")
    scores_txt_file = os.path.join(save_dir, f"final_scores_{rag_or_plain}.txt")
    save_scores_to_pkl(final_scores, scores_pkl_file)
    convert_pkl_to_txt(scores_pkl_file, scores_txt_file)

    print(f"Results saved to {results_file}")
    print(f"Scores saved to {scores_pkl_file} and {scores_txt_file}")

    return results, final_scores
