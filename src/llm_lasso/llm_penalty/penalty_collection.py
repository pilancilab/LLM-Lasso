
from pydantic import BaseModel
from dataclasses import dataclass, field
from langchain_community.vectorstores import Chroma
from llm_lasso.llm_penalty.llm import LLMQueryWrapperWithMemory
from llm_lasso.utils.score_collection import create_general_prompt, extract_scores_from_responses, \
    save_responses_to_file, save_scores_to_pkl, create_json_prompt
from llm_lasso.utils.data import convert_pkl_to_txt
from llm_lasso.llm_penalty.rag.rag_context import get_rag_context
import os
import logging
import json
import numpy as np
from tqdm import tqdm


@dataclass
class PenaltyCollectionParams:
    batch_size: int = field(default=30)
    n_trials: int = field(default=1)
    wipe: bool = field(default=False)
    summarized_gene_doc_rag: bool = field(default=False)
    filtered_cancer_doc_rag: bool = field(default=False)
    pubmed_rag: bool = field(default=False)
    default_rag: bool = field(default=True)
    retry_limit: int = field(default=10)
    default_num_docs: int = field(default=3)
    enable_memory: bool = field(default=True)
    small: bool = field(default=False)
    memory_size: int = field(default=200)
    shuffle: bool = field(default=False)

    def has_rag(self):
        return self.summarized_gene_doc_rag or \
            self.filtered_cancer_doc_rag or \
            self.pubmed_rag or \
            self.default_rag


class Score(BaseModel):
    gene: str
    penalty_factor: float
    reasoning: str


class GeneScores(BaseModel):
    scores: list[Score]


def wipe_llm_penalties(save_dir, rag: bool):
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
    json_data=None
):
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

    # Determine which trial to start from
    start_trial = len(trial_scores)

    results = []
    trial_scores = []
    trial = start_trial

    while trial < params.n_trials:
        # maybe shuffle the feature names
        idxs = np.arange(len(feature_names))
        if params.shuffle:
            np.random.shuffle(idxs)

        logging.info(f"Starting trial {trial + 1} out of {params.n_trials}")
        batch_scores = []

        if params.enable_memory:
            model.start_memory()

        # loop through batches of genes
        for start_idx in tqdm(range(0, total_features, params.batch_size), desc=f"Processing trial {trial + 1}..."):
            end_idx = min(start_idx + params.batch_size, total_features)
            batch_features = [feature_names[i] for i in idxs[start_idx:end_idx]]
            upper_batch_names = [n.upper() for n in batch_features]

            # Construct the query for this batch of features
            if json_data is None:
                query = create_general_prompt(prompt_file, category, batch_features)
            else:
                query = create_json_prompt(prompt_file, batch_features, json_data)

            # If we're performing RAG, get the RAG context
            context = get_rag_context(
                batch_features, category, vectorstore,
                model, omim_api_key,
                pubmed_docs=params.pubmed_rag,
                filtered_cancer_docs=params.filtered_cancer_doc_rag,
                summarized_gene_docs=params.summarized_gene_doc_rag,
                original_docs=params.default_rag,
                default_num_docs=params.default_num_docs,
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
            if model.has_structured_output():
                gene_scores: GeneScores = model.structured_query(
                    system_message=system_message,
                    full_prompt=full_prompt,
                    response_format_class=GeneScores,
                    sleep_time=1,
                )
                scores_list = [score for score in gene_scores.scores if score.gene.upper() in upper_batch_names]
                genes_retrieved = set([score.gene.upper() for score in scores_list])
                missing = set(upper_batch_names).difference(genes_retrieved)

                # Retry logic for score validation
                n_retries = 0
                while len(missing) > 0:
                    logging.warning(f"We are missing genes {missing}")
                    assert n_retries < params.retry_limit
                    n_retries += 1

                    gene_scores: GeneScores = model.maybe_retry_last(sleep_time=1)
                    scores_list = [score for score in gene_scores.scores if score.gene.upper() in upper_batch_names]
                    genes_retrieved = set([score.gene.upper() for score in scores_list])
                    missing = set(upper_batch_names).difference(genes_retrieved)
                
                genes_to_scores = {
                    score.gene: score.penalty_factor for score in gene_scores.scores
                }
                batch_scores_partial = [genes_to_scores[gene] for gene in batch_features]
                output = gene_scores.model_dump_json()
            else:
                output = model.query(
                    system_message=system_message,
                    full_prompt=full_prompt,
                    sleep_time=1,
                )

                batch_scores_partial = extract_scores_from_responses(
                    output if isinstance(output, list) else [output],
                    batch_features
                )

                # Retry logic for score validation
                while len([score for score in batch_scores_partial if score is not None]) != len(batch_features):
                    logging.info(output)
                    try:
                        logging.warning(f"Batch scores count mismatch for genes {batch_features}. Retrying...")
                        output = model.maybe_retry_last(sleep_time=1)
                        batch_scores_partial = extract_scores_from_responses(
                            output if isinstance(output, list) else [output],
                            batch_features
                        )
                    except Exception as e:
                        logging.error(f"Error during retry: {str(e)}. Continuing retry...")
                # end retry while loop
            # end structured output if/else

            logging.info(f"Successfully retrieved valid scores for batch: {batch_features}")
            batch_scores.append(batch_scores_partial)
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
