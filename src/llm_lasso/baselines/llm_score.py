"""
Implements LLM-Select feature selector from {jeong2024llm_scorefeatureselectionlarge}. Three approaches that the authors consider:
(i) selecting features based on LLM-generated feature importance scores;
(ii) selecting features based on an LLM-generated ranking;
(iii) sequentially selecting features in a dialogue with an LLM.

We implement LLM-Scores.
"""
import warnings
import os
import logging
from llm_lasso.llm_penalty.llm import LLMQueryWrapperWithMemory
from llm_lasso.utils.score_collection import create_general_prompt, \
    save_responses_to_file, save_scores_to_pkl
from llm_lasso.utils.data import convert_pkl_to_txt
from llm_lasso.llm_penalty.query_scores import query_scores_with_retries
import json
from tqdm import tqdm
warnings.filterwarnings("ignore")  # Suppress warnings


# Wipe all files in save_dir related to this method
def wipe_save_dir(save_dir):
    files_to_remove = [
        "results_llm_score.txt",
        "feature_scores_llm_score.pkl",
        "feature_scores_llm_score.txt",
        "trial_scores_llm_score.json",
        "llm_score_selected_features.json"
    ]
    for file_name in files_to_remove:
        file_path = os.path.join(save_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed file: {file_path}")

def query_scores(
    category: str,
    feature_names: list[str],
    prompt_filename: str,
    save_dir: str,
    model: LLMQueryWrapperWithMemory,
    batch_size = 30,
    n_trials = 1,
    wipe = False
):
    """
    Query features in batches and extract LLM-Score importances.

    Splits a list of features into batches, constructs prompts, and queries the selected LLM.

    Parameters:
    - `category`: The category or context for the query (e.g., "cancer type").
    - `feature_names`: List of feature names,
    - `prompt_filename`: Path to the prompt file used for constructing queries.
    - `save_dir`: Directory where results and scores will be saved.
    - `model`: LLMQueryWrapperWithMemory object for performing queries
    - `batch_size`: Number of genes to pass into the LLM at once.
    - `n_trials`: Bumber of trials to average over.
    - `wipe`: Whether to wipe the save directory before getting scores.
    """
    if wipe:
        logging.info("Wiping save directory before starting.")
        wipe_save_dir(save_dir)
    
    total_features = len(feature_names)
    results = []
    trial_scores = []
    print(f'Total number of features in processing: {len(feature_names)}.')
    os.makedirs(save_dir, exist_ok=True)
    trial_scores_file = os.path.join(save_dir, "trial_scores_llm_score.json")

    # Load existing progress if the file already exists
    if os.path.exists(trial_scores_file):
        with open(trial_scores_file, "r") as json_file:
            trial_scores = json.load(json_file)

    # Determine which trial to start from
    start_trial = len(trial_scores)
    for trial in range(start_trial, n_trials):
        logging.info(f"Starting trial {trial + 1} out of {n_trials}")
        batch_scores = []

        for start_idx in tqdm(range(0, total_features, batch_size), desc=f'Processing trial {trial + 1}...'):
            end_idx = min(start_idx + batch_size, total_features)
            batch_features = feature_names[start_idx:end_idx]
            # Construct the prompt
            prompt = create_general_prompt(prompt_filename, category, batch_features)
            system_message = "For each feature input by the user, your task is to provide a feature importance score (between 0 and 1; larger value indicates greater importance) for predicting whether an individual will subscribe to a term deposit and a reasoning behind how the importance score was assigned."

            # Query the LLM, with special handling if the LLM allows
            # structured queries
            batch_scores_partial, response = query_scores_with_retries(
                model, system_message, prompt,
                batch_features
            )

            logging.info(f"Successfully retrieved valid scores for batch: {batch_features}")
            batch_scores.extend(batch_scores_partial)
            results.append(response)
        # end batch for loop
        # Check if the trial scores match the total genes
        if len(batch_scores) == total_features:
            trial_scores.append({"iteration": trial + 1, "scores": batch_scores})
            # Incrementally save progress after each trial
            with open(trial_scores_file, "w") as json_file:
                json.dump(trial_scores, json_file, indent=4)
            logging.info(f"Trial {trial + 1} completed and saved.")
        else:
            logging.warning(f"Trial {trial + 1} scores length mismatch. Retrying...")
    # end trial for loop

    # Calculate final scores averaged across trials
    if trial_scores:
        final_scores = [
            sum(score for score in scores if score is not None) / len(scores)
            for scores in zip(*[trial["scores"] for trial in trial_scores])
        ]
    else:
        final_scores = []
    print(f'Total number of scores collected: {len(final_scores)}.')
    logging.info(f"Final scores vector (averaged across trials) calculated with length: {len(final_scores)}")
    results_file = os.path.join(save_dir, "results_llmselect.txt")
    save_responses_to_file(results, results_file)
    scores_pkl_file = os.path.join(save_dir, "feature_scores_llm_score.pkl")
    scores_txt_file = os.path.join(save_dir, "feature_scores_llm_score.txt")
    save_scores_to_pkl(final_scores, scores_pkl_file)
    convert_pkl_to_txt(scores_pkl_file, scores_txt_file)
    print(f"Trial scores saved to {trial_scores_file}")
    if len(final_scores) != len(feature_names):
        raise ValueError(
            f"Mismatch between number of scores ({len(final_scores)}) and number of gene names ({len(feature_names)})."
        )
    return results, final_scores

# Select top genes based on ranking of the importance scores
def llm_score(
    category: str,
    feature_names: list[str],
    prompt_filename: str,
    save_dir: str,
    model: LLMQueryWrapperWithMemory,
    batch_size = 30,
    n_trials = 1,
    wipe = False,
    k_min=0, k_max=50, step=5
):
    """
    Select top genes based on LLM-generated importance scores and save results for multiple k values.

    Args:
        `category` (str): Category or context for the query.
        `feature_names` (list[str]): List of feature names.
        `prompt_filename` (str): Path to the prompt file.
        `model`: LLMQueryWrapperWithMemory object
        `batch_size` (int): Number of genes to process per batch.
        `n_trials` (int): Number of trials to run.
        `wipe` (bool): If True, wipe all files in save_dir before starting.
        `k_min` (int): Minimum number of top genes to select.
        `k_max` (int): Maximum number of top genes to select.
        `step` (int): Step size for k.

    Returns:
        dict: A dictionary with k values as keys and corresponding top gene lists as values.
    """
    _, final_scores = query_scores(
        category, feature_names, prompt_filename,
        save_dir, model, batch_size, n_trials, wipe
    )

    # Get sorted indices of the scores in descending order
    sorted_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)

    # Generate top-k genes for each k in the specified range
    top_k_dict = {}
    for k in range(k_min, k_max + 1, step):
        k = min(k, len(final_scores))  # Ensure k does not exceed the number of scores
        top_features = [feature_names[i] for i in sorted_indices[:k]]
        top_k_dict[k] = top_features

    # Save the top-k results to a JSON file
    os.makedirs(save_dir, exist_ok=True)
    top_k_file = os.path.join(save_dir, "llmselect_selected_features.json")
    with open(top_k_file, "w") as f:
        json.dump(top_k_dict, f, indent=4)

    print(f"Top-k feature lists saved to {top_k_file}")
    return top_k_dict