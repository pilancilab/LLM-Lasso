"""
Accessing memory-enhanced Llama through OpenRouter API.
"""
# key modifications: (i) import OpenRouterLLM (ii) use serialized prompt for system and human message.

import os
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import SystemMessage, HumanMessage
import warnings
import time
from tqdm import tqdm
from Expert_RAG.utils import *
from memory_gpt import wipe_plain
from LlaMa.lama import OpenRouterLLM
warnings.filterwarnings("ignore")  # Suppress warnings

# Replace OpenAI with OpenRouter
OPENROUTER_API_KEY = "YOUR KEY HERE"

model_name = "meta-llama/llama-3-8b-instruct"

# Initialize the custom LLM
llm_model = OpenRouterLLM(
    api_key=OPENROUTER_API_KEY,
    model=model_name,
    top_p=0.9,
    temperature=0.9,
    repetition_penalty=1.0
)

# Setup LangChain Memory
prompt_dir = "prompts/prompt_file.txt"

# Function to query OpenRouter API by batch with no reweighting of batch scores
def plain_open(category, genenames, json_data = None, is_json = False, prompt_dir="prompts/prompt_file.txt", save_dir="output", chat=llm_model, batch_size=30, n_trials=1, wipe = False):
    """
    Query genes in batches and compute the final scores as averages across trials.

    Args:
        category (str): The category or context for the query.
        genenames (list[str]): List of gene names.
        prompt_dir (str): Path to the prompt file.
        chat (OpenRouterLLM): LLM model for querying.
        batch_size (int): Number of genes to process per batch.
        n_trials (int): Number of trials to run.
        save_dir (str): Directory to save the trial scores and results.

    Returns:
        tuple: Contains results and final scores.
    """
    if wipe:
        logging.info("Wiping save directory before starting.")
        print("Wiping save directory before starting.")
        wipe_plain(save_dir)

    total_genes = len(genenames)
    results = []
    trial_scores = []

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    trial_scores_file = os.path.join(save_dir, "trial_scores_plain.json")

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

        for start_idx in tqdm(range(0, total_genes, batch_size), desc='Processing...'):
            end_idx = min(start_idx + batch_size, total_genes)
            batch_genes = genenames[start_idx:end_idx]

            # Construct the prompt
            if is_json:
                prompt = create_json_prompt(prompt_dir, batch_genes, json_data)
            else:
                prompt = create_general_prompt(prompt_dir, category, batch_genes)

            memory = ConversationSummaryBufferMemory(llm=chat, max_token_limit=200)
            memory.save_context(
                {"input": prompt},
                {"output": "Processing genes in this batch. Continuing from previous batches if any."}
            )

            # Retrieve memory context
            memory_context = memory.load_memory_variables({})
            full_context = memory_context.get("history", "")

            # Combine full context with current prompt
            messages = [
                SystemMessage(content="You are an expert in cancer genomics and bioinformatics."),
                HumanMessage(content=f"{full_context}\n{prompt}")
            ]

            # Serialize messages into a single string
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
                    messages = [
                        SystemMessage(content="You are an expert assistant with access to gene and cancer knowledge."),
                        HumanMessage(content=f"{full_context}\n{prompt}")
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

            memory.save_context(
                {"input": prompt},
                {"output": response}
            )

            time.sleep(0.1)

        # Check if the trial scores match the total genes
        if len(batch_scores) == total_genes:
            trial_scores.append({"iteration": trial + 1, "scores": batch_scores})
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

    os.makedirs(save_dir, exist_ok=True)
    trial_scores_file = os.path.join(save_dir, f"trial_scores_plain.json")

    # Save trial scores to JSON
    with open(trial_scores_file, "w") as json_file:
        json.dump(trial_scores, json_file, indent=4)

    print(f"Trial scores saved to {trial_scores_file}")

    return results, final_scores


def save_memory_to_file(memory, file_path): # summary created by langchain's memory module
    """
    Saves the conversation memory to a text file.

    Args:
        memory: The LangChain memory object containing conversation history.
        file_path: Path to the file where memory should be saved.
    """
    # Retrieve the stored memory context
    memory_context = memory.load_memory_variables({})
    history = memory_context.get("history", [])

    # Write the memory history to a file
    with open(file_path, "w") as f:
        if isinstance(history, list):  # If memory is structured as a list of dicts
            for i, entry in enumerate(history):
                input_text = entry.get("input", "No input found")
                output_text = entry.get("output", "No output found")
                f.write(f"Batch {i + 1}:\n")
                f.write("Input:\n")
                f.write(input_text + "\n\n")
                f.write("Output:\n")
                f.write(output_text + "\n\n")
                f.write("-" * 40 + "\n")  # Separator for readability
        else:  # If history is a single string or raw text
            f.write(history)


# Example
if __name__ == "__main__":
    genenames_unique = ["AASS", "ABCA6", "ABCB1", "ABHD6", "ABHD8", "ABRACL", "ABTB2"]
    category = "tFL (Transformed Follicular Lymphoma)"
    # print created prompt
    results, final_scores = plain_open(category, genenames_unique, batch_size=3)
    # Combine and display results
    for idx, batch_result in enumerate(results):
        print(f"Batch {idx + 1} Results:\n{batch_result}\n")

    print(final_scores) # test passed