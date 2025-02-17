"""
Implements LLM-Select feature selector from {jeong2024llmselectfeatureselectionlarge}. Three approaches that the authors consider:
(i) selecting features based on LLM-generated feature importance scores;
(ii) selecting features based on an LLM-generated ranking;
(iii) sequentially selecting features in a dialogue with an LLM.

We implement LLM-Scores.
"""

import argparse
import os
import time
import warnings
from pydantic import BaseModel
from tqdm import tqdm
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from utils.score_collection import *
from llm_penalty.openrouter import OpenRouterLLM
import utils.data as dp
warnings.filterwarnings("ignore")  # Suppress warnings

os.environ["OPENAI_API_KEY"] = "YOUR KEY HERE"
OPENROUTER_API_KEY = "YOUR KEY HERE"

openai_client = OpenAI()


class Score(BaseModel):
    gene: str
    penalty_factor: float
    reasoning: str


class GeneScores(BaseModel):
    scores: list[Score]

# Wipe all files in save_dir related to this method
def wipe_save_dir(save_dir):
    files_to_remove = [
        "results_llmselect.txt",
        "gene_scores_llmselect.pkl",
        "gene_scores_llmselect.txt",
        "trial_scores_llmselect.json",
        "llmselect_selected_genes.json"
    ]
    for file_name in files_to_remove:
        file_path = os.path.join(save_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed file: {file_path}")

# Query in batches with OpenRouter
def query_genes_openrouter(category, genenames_unique, prompt_dir, save_dir, model_name, temp=0.5, top_p=0.9, repetition_penalty=0.9, batch_size=30, n_trials=1, wipe=False):
    """
    Query genes in batches using OpenRouter LLM and compute final scores as averages across trials.
    Args:
        category (str): The category or context for the query.
        genenames_unique (list[str]): A list of unique gene names to query.
        prompt_dir (str): Path to the prompt file.
        model_name (str): Name of the OpenRouter model to use.
        temp (float): Temperature for randomness in responses.
        top_p (float): Top-p sampling parameter.
        repetition_penalty (float): Penalty for repeated tokens.
        batch_size (int): Number of genes to process per batch.
        n_trials (int): Number of trials to run.
        wipe (bool): If True, wipe all files in save_dir before starting.
    Returns:
        tuple: Contains results and final scores.
    """
    if wipe:
        logging.info("Wiping save directory before starting.")
        wipe_save_dir(save_dir)
    total_genes = len(genenames_unique)
    results = []
    trial_scores = []
    print(f'Total number of features in processing: {len(genenames_unique)}.')
    os.makedirs(save_dir, exist_ok=True)
    trial_scores_file = os.path.join(save_dir, "trial_scores_llmselect.json")
    # Load existing progress if the file already exists
    if os.path.exists(trial_scores_file):
        with open(trial_scores_file, "r") as json_file:
            trial_scores = json.load(json_file)
    # Determine which trial to start from
    start_trial = len(trial_scores)
    for trial in range(start_trial, n_trials):
        logging.info(f"Starting trial {trial + 1}/{n_trials}")
        batch_scores = []
        openrouter_chat = OpenRouterLLM(
            api_key=OPENROUTER_API_KEY,
            model=model_name,
            top_p=top_p,
            temperature=temp,
            repetition_penalty=repetition_penalty
        )
        for start_idx in tqdm(range(0, total_genes, batch_size), desc=f'Processing trial {trial + 1}...'):
            end_idx = min(start_idx + batch_size, total_genes)
            batch_genes = genenames_unique[start_idx:end_idx]
            # Construct the prompt
            prompt = create_general_prompt(prompt_dir, category, batch_genes)
            messages = [
                SystemMessage(content="For each feature input by the user, your task is to provide a feature importance score (between 0 and 1; larger value indicates greater importance) for predicting whether an individual will subscribe to a term deposit and a reasoning behind how the importance score was assigned."),
                HumanMessage(content=f"{prompt}")
            ]
            # Serialize messages into a single string
            serialized_prompt = "\n".join([f"{msg.content}" for msg in messages])
            # Query OpenRouter
            response = openrouter_chat(serialized_prompt)
            # Extract scores and validate batch
            batch_scores_partial = extract_scores_from_responses([response] if not isinstance(response, list) else response, batch_genes)
            # Retry logic for score validation
            while len([score for score in batch_scores_partial if score is not None]) != len(batch_genes):
                print(response)
                try:
                    logging.warning(f"Batch scores count mismatch for genes {batch_genes}. Retrying...")
                    # Regenerate the query prompt
                    query = create_general_prompt(prompt_dir, category, batch_genes)
                    messages = [
                        SystemMessage(content="For each feature input by the user, your task is to provide a feature importance score (between 0 and 1; larger value indicates greater importance) for predicting whether an individual will subscribe to a term deposit and a reasoning behind how the importance score was assigned."),
                        HumanMessage(content=query)
                    ]
                    # Retry querying the model
                    serialized_prompt = "\n".join([f"{msg.content}" for msg in messages])
                    # Query OpenRouter
                    response = openrouter_chat(serialized_prompt)
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
            results.append(response)
            time.sleep(0.1)
        # Check if the trial scores match the total genes
        if len(batch_scores) == total_genes:
            trial_scores.append({"iteration": trial + 1, "scores": batch_scores})
            # Incrementally save progress after each trial
            with open(trial_scores_file, "w") as json_file:
                json.dump(trial_scores, json_file, indent=4)
            logging.info(f"Trial {trial + 1} completed and saved.")
        else:
            logging.warning(f"Trial {trial + 1} scores length mismatch. Retrying...")
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
    scores_pkl_file = os.path.join(save_dir, "gene_scores_llmselect.pkl")
    scores_txt_file = os.path.join(save_dir, "gene_scores_llmselect.txt")
    save_scores_to_pkl(final_scores, scores_pkl_file)
    dp.convert_pkl_to_txt(scores_pkl_file, scores_txt_file)
    print(f"Trial scores saved to {trial_scores_file}")
    if len(final_scores) != len(genenames_unique):
        raise ValueError(
            f"Mismatch between number of scores ({len(final_scores)}) and number of gene names ({len(genenames_unique)})."
        )
    return results, final_scores

# Query in batches with OpenAI API
def query_genes_GPT(category, genenames_unique, prompt_dir, save_dir, model_name, temp=0.5, batch_size=30, n_trials=1, wipe=False):
    """
    Query genes in batches using OpenAI GPT and compute final scores as averages across trials.

    Args:
        category (str): The category or context for the query.
        genenames_unique (list[str]): A list of unique gene names to query.
        prompt_dir (str): Path to the prompt file.
        model_name (str): Name of the GPT model to use.
        temp (float): Temperature for randomness in responses.
        batch_size (int): Number of genes to process per batch.
        n_trials (int): Number of trials to run.
        wipe (bool): If True, wipe all files in save_dir before starting.

    Returns:
        tuple: Contains results and final scores.
    """
    if wipe:
        logging.info("Wiping save directory before starting.")
        wipe_save_dir(save_dir)

    total_genes = len(genenames_unique)
    results = []
    trial_scores = []
    openai_chat = ChatOpenAI(model=model_name, temperature=temp)
    print(f'Total number of features in processing: {len(genenames_unique)}.')

    os.makedirs(save_dir, exist_ok=True)
    trial_scores_file = os.path.join(save_dir, "trial_scores_llmselect.json")

    # Load existing progress if the file already exists
    if os.path.exists(trial_scores_file):
        with open(trial_scores_file, "r") as json_file:
            trial_scores = json.load(json_file)

    # Determine which trial to start from
    start_trial = len(trial_scores)

    for trial in range(start_trial, n_trials):
        logging.info(f"Starting trial {trial + 1}/{n_trials}")
        batch_scores = []

        for start_idx in tqdm(range(0, total_genes, batch_size), desc=f'Processing trial {trial + 1}...'):
            end_idx = min(start_idx + batch_size, total_genes)
            batch_genes = genenames_unique[start_idx:end_idx]

            # Construct the prompt
            prompt = create_general_prompt(prompt_dir, category, batch_genes)

            messages = [
                SystemMessage(content="For each feature input by the user, your task is to provide a feature importance score (between 0 and 1; larger value indicates greater importance) for predicting whether an individual will subscribe to a term deposit and a reasoning behind how the importance score was assigned."),
                HumanMessage(content=f"{prompt}")
            ]

            # Query ChatOpenAI
            response = openai_chat(messages)
            results.append(response.content)

            # Extract scores and validate batch
            batch_scores_partial = extract_scores_from_responses(
                [response.content] if not isinstance(response.content, list) else response.content, batch_genes)

            # Retry logic for score validation
            while len([score for score in batch_scores_partial if score is not None]) != len(batch_genes):
                print(response.content)
                try:
                    logging.warning(f"Batch scores count mismatch for genes {batch_genes}. Retrying...")

                    # Regenerate the query prompt
                    query = create_general_prompt(prompt_dir, category, batch_genes)
                    messages = [
                        SystemMessage(content="For each feature input by the user, your task is to provide a feature importance score (between 0 and 1; larger value indicates greater importance) for predicting whether an individual will subscribe to a term deposit and a reasoning behind how the importance score was assigned."),
                        HumanMessage(content=query)
                    ]

                    # Query OpenRouter
                    response = openai_chat(messages)
                    # Extract scores again
                    batch_scores_partial = extract_scores_from_responses(
                        response.content if isinstance(response.content, list) else [response.content],
                        batch_genes
                    )

                    # Optional delay to handle rate limits or avoid spamming
                    time.sleep(0.1)

                except Exception as e:
                    logging.error(f"Error during retry: {str(e)}. Continuing retry...")

            logging.info(f"Successfully retrieved valid scores for batch: {batch_genes}")
            batch_scores.extend(batch_scores_partial)
            results.append(response)

            time.sleep(0.1)

        # Check if the trial scores match the total genes
        if len(batch_scores) == total_genes:
            trial_scores.append({"iteration": trial + 1, "scores": batch_scores})

            # Incrementally save progress after each trial
            with open(trial_scores_file, "w") as json_file:
                json.dump(trial_scores, json_file, indent=4)

            logging.info(f"Trial {trial + 1} completed and saved.")
        else:
            logging.warning(f"Trial {trial + 1} scores length mismatch. Retrying...")

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

    scores_pkl_file = os.path.join(save_dir, "gene_scores_llmselect.pkl")
    scores_txt_file = os.path.join(save_dir, "gene_scores_llmselect.txt")
    save_scores_to_pkl(final_scores, scores_pkl_file)
    dp.convert_pkl_to_txt(scores_pkl_file, scores_txt_file)

    print(f"Trial scores saved to {trial_scores_file}")

    if len(final_scores) != len(genenames_unique):
        raise ValueError(
            f"Mismatch between number of scores ({len(final_scores)}) and number of gene names ({len(genenames_unique)})."
        )

    return results, final_scores



# Select top genes based on ranking of the importance scores
def llm_score(LLM_type, category, genenames, prompt_dir, save_dir, model_name, k_min=0, k_max=50, step=5, temp=0.5, top_p=0.9, repetition_penalty=0.9, batch_size=40, n_trials=1, wipe=False):
    """
    Select top genes based on LLM-generated importance scores and save results for multiple k values.

    Args:
        LLM_type (str): Type of LLM to use ("GPT" or "General").
        category (str): Category or context for the query.
        genenames (list[str]): List of gene names.
        prompt_dir (str): Path to the prompt file.
        model_name (str): Name of the model to use.
        k_min (int): Minimum number of top genes to select.
        k_max (int): Maximum number of top genes to select.
        step (int): Step size for k.
        temp (float): Temperature for randomness in responses.
        top_p (float): Top-p sampling parameter.
        repetition_penalty (float): Penalty for repeated tokens.
        batch_size (int): Number of genes to process per batch.
        n_trials (int): Number of trials to run.
        wipe (bool): If True, wipe all files in save_dir before starting.

    Returns:
        dict: A dictionary with k values as keys and corresponding top gene lists as values.
    """
    if LLM_type == 'GPT':
        results, final_scores = query_genes_GPT(category, genenames, prompt_dir, save_dir, model_name, temp, batch_size, n_trials, wipe=wipe)
    elif LLM_type == 'General':
        results, final_scores = query_genes_openrouter(category, genenames, prompt_dir, save_dir, model_name, temp, top_p, repetition_penalty, batch_size, n_trials, wipe=wipe)
    else:
        raise ValueError(
            "LLM type should be either 'GPT', accessed through OpenAI, or 'General', accessed through OpenRouter.")

    # Get sorted indices of the scores in descending order
    sorted_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)

    # Generate top-k genes for each k in the specified range
    top_k_dict = {}
    for k in range(k_min, k_max + 1, step):
        k = min(k, len(final_scores))  # Ensure k does not exceed the number of scores
        top_genes = [genenames[i] for i in sorted_indices[:k]]
        top_k_dict[k] = top_genes

    # Save the top-k results to a JSON file
    os.makedirs(save_dir, exist_ok=True)
    top_k_file = os.path.join(save_dir, "llmselect_selected_genes.json")
    with open(top_k_file, "w") as f:
        json.dump(top_k_dict, f, indent=4)

    print(f"Top-k gene lists saved to {top_k_file}")
    return top_k_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-based feature selection for genes.")
    parser.add_argument("--prompt_dir", type=str, required=True, help="Path to the prompt file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Name of the LLM model to use.")
    parser.add_argument("--genenames_path", type=str, required=True, help="Path to the gene names file (.pkl or .txt).")
    parser.add_argument("--k_min", type=int, default=0, help="Minimum number of top genes to select.")
    parser.add_argument("--k_max", type=int, default=50, help="Maximum number of top genes to select.")
    parser.add_argument("--step", type=int, default=5, help="Step size for the range of k values.")
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size for processing.")
    parser.add_argument("--category", type=str, required=True, help="Category for the query (e.g., cancer type).")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the results and scores.")
    parser.add_argument("--LLM_type", type=str, default="GPT", choices=["GPT", "General"],
                        help="Specify LLM type: 'GPT' for OpenAI or 'General' for OpenRouter.")
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature for randomness in responses.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter.")
    parser.add_argument("--repetition_penalty", type=float, default=0.9, help="Penalty for repeated tokens.")
    parser.add_argument("--wipe", action="store_true", help="If set, wipe the save directory before starting.")

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

    # Call llm_score
    top_k_dict = llm_score(
        LLM_type=args.LLM_type,
        category=args.category,
        genenames=genenames,
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
        wipe=args.wipe
    )

    print("Top-k gene selection complete. Results saved.")