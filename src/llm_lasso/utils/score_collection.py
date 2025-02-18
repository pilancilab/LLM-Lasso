"""
Utils for LLM
"""

from langchain.prompts import PromptTemplate
# from string import Template
import pickle as pkl
import re
import sys
import numpy as np

def create_general_prompt(prompt_dir,category,genes, singular = False, display = False):
    """
    Generates a dynamic prompt by replacing placeholders in a pre-defined template with provided arguments for the dataset.

    :param prompt_dir: (str)
        Path to the file containing the pre-defined prompt template. This file should have placeholders `{category}` and `{genes}` to be replaced dynamically.

    :param category: (str)
        The cancer category or subtype for which the penalty factors are being requested (e.g., "tFL (Transformed Follicular Lymphoma)").

    :param genes: (list[str])
        A list of gene names for which penalty factors need to be calculated (e.g., `["AASS", "ABCA6", "ABCB1"]`). These will be formatted as a comma-separated string in the prompt.

    :param display: (bool, optional)
        If set to `True`, the function prints the dynamically generated prompt to the console for review. Default is `False`.

    :return: (str)
        The final generated prompt with the placeholders `{category}` and `{genes}` replaced by the provided arguments.
    """
    with open(prompt_dir, "r", encoding="utf-8") as file:
        penalty_factors_prompt_template = file.read()

    # Create the PromptTemplate object
    penalty_factors_prompt = PromptTemplate(
        input_variables=["category", "genes"],  # Define variables to replace
        template=penalty_factors_prompt_template  # Pass the loaded template string
    )

    # Fill in the prompt dynamically
    if not singular:
        filled_prompt = penalty_factors_prompt.format(
            category=category,
            genes=", ".join(genes)  # Format genes as a comma-separated list
        )
    else:
        filled_prompt = penalty_factors_prompt.format(
            category=category,
            genes=genes
        )

    # Print the filled prompt
    if display:
        print(filled_prompt)

    return filled_prompt


def create_json_prompt(prompt_dir, genes, genes_dict, display=False):
    """
    Generates a dynamic prompt by replacing placeholders in a pre-defined template with provided arguments for the dataset.

    :param prompt_dir: (str)
        Path to the file containing the pre-defined prompt template. This file should have a placeholder `{genes}` to be replaced dynamically.

    :param genes_dict: (dict)
        A dictionary where each gene name is a key, and its corresponding explanation is the value.

    :param singular: (bool, optional)
        If set to `True`, the function formats the prompt differently for a single gene. Default is `False`.

    :param display: (bool, optional)
        If set to `True`, the function prints the dynamically generated prompt to the console for review. Default is `False`.

    :return: (str)
        The final generated prompt with the placeholder `{genes}` replaced by the provided arguments.
    """
    with open(prompt_dir, "r", encoding="utf-8") as file:
        new_prompt = file.read()

    # Create the PromptTemplate object
    new_prompt = PromptTemplate(
        input_variables=["genes", "exp"],  # Define variables to replace
        template=new_prompt  # Pass the loaded template string
    )


    # Format genes dictionary into a string
    genes_formatted = "; ".join([f"{gene}: {description}" for gene, description in genes_dict.items() if description])

    # Fill in the prompt dynamically
    filled_prompt = new_prompt.format(
        genes=genes_formatted,
        exp = genes
    )

    # Print the filled prompt
    if display:
        print(filled_prompt)

    return filled_prompt


# Helper Functions to extract scores from response
def save_responses_to_file(responses, file_path):
    """
    Saves only the actual responses from GPT to a text file.

    Args:
        results: List of responses (batch outputs) from GPT.
        file_path: Path to the file where the results should be saved.
    """
    with open(file_path, "w") as f:
        for idx, response in enumerate(responses):
            f.write(f"Batch {idx + 1} Results:\n{response}\n{'-' * 40}\n")

def normalize_genenames(genenames):
    """
    Normalizes gene names by removing special characters such as '|', '/', and '-',
    while preserving periods for valid gene names like 'KRT13.5'.

    Args:
        genenames (list[str]): List of original gene names.

    Returns:
        list[str]: List of normalized gene names.
    """
    normalized = []
    for gene in genenames:
        # Remove invalid characters, but preserve periods in valid gene names
        if re.match(r"[A-Za-z]+[0-9]+(?:\.[0-9]+)?", gene):  # Check valid format like 'KRT13.5'
            normalized.append(gene)
        else:
            normalized.append(gene.replace('|', '').replace('/', '').replace('-', '').replace('.', ''))
    return normalized

def extract_scores_from_responses(responses, dummy):
    """
    Extracts all numbers between 0 and 1 from GPT responses.

    Args:
        responses: List of GPT-generated responses as strings.

    Returns:
        A list of lists, where each inner list contains all scores from a single batch.
    """
    all_scores = []

    for response in responses:
        # Use a regex to find all numbers strictly in the range 0 <= num < 1
        matches = re.findall(r"\b0(?:\.\d+)?\b", response)

        # Convert matches to floats and store them
        scores = [float(match) for match in matches]
        all_scores.extend(scores)

    return all_scores

def save_scores_to_pkl(scores, file_path):
    """
    Saves extracted scores to a .pkl file.

    Args:
        scores: List of extracted scores (list of lists).
        file_path: Path to the .pkl file to save the data.
    """
    with open(file_path, "wb") as f:
        pkl.dump(scores, f)


def find_max_gene(batch_genes, results):
    """
        Find the gene with the max score in each batch; return [max_gene, max_score] for each batch.

        :param batch_genes: (list[str])
             A list of gene names for which penalty factors need to be calculated in a batch (e.g., `["AASS", "ABCA6", "ABCB1"]`).

        :param results: (str)
            The response of the generation GPT.

        :return: ([str,float])
            Pair of max gene and the corresponding score.
        """
    # extract scores from results
    scores = extract_scores_from_responses([results])
    # debugging
    try:
        assert len(scores) == len(batch_genes), "Length mismatch between scores and batch_genes."
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        print(f"{len(scores)} Scores: {scores}")
        print(f"genes: {batch_genes}")
        print(f"results: {results}")
        # return scores, results  # Return the scores for further debugging or handling
        sys.exit(1)
    scores = np.array(scores)
    max_scores = np.max(scores)
    max_index = np.argmax(scores)
    max_gene = batch_genes[max_index]
    return [max_gene, max_scores]


def retrieval_docs(batch_genes, category, retriever, small = False): # small parameters adjust for input token limit of llama-3b-instruct.
    # retrieval retrieval prompt
    docs = []
    prompt_dir = 'prompts/retrieval_prompt.txt'
    if not small:
        for gene in batch_genes:
            retrieval_query = create_general_prompt(prompt_dir, category, [gene], True)
            retrieved_docs = retriever.get_relevant_documents(retrieval_query)
            docs.extend(retrieved_docs)
    else:
        retrieval_query = create_general_prompt(prompt_dir, category, batch_genes)
        retrieved_docs = retriever.get_relevant_documents(retrieval_query)
        docs.extend(retrieved_docs)
    return docs


def get_unique_docs(docs):
    """
    Filters unique documents from a list of Document objects.

    Args:
        docs (list): List of Document objects.

    Returns:
        list: A list of unique Document objects.
    """
    # Use a set to track unique documents based on content and metadata
    seen = set()
    unique_docs = []

    for doc in docs:
        # Create a unique identifier for the document (content + metadata)
        doc_id = (doc.page_content, tuple(sorted(doc.metadata.items())))
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)

    return unique_docs
