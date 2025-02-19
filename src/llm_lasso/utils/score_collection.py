"""
This script implments the utils for score collection process from LLMs.
"""

from langchain.prompts import PromptTemplate
import pickle as pkl
import re
import sys
import numpy as np

def create_general_prompt(prompt_dir, category, genes, singular=False, display=False):
    """
    Generates a dynamic prompt by replacing placeholders in a pre-defined template 
    with provided arguments. The placeholders `{category}` and `{genes}` are expected 
    in the template file.

    Args:
        prompt_dir : str
            Path to the file containing the pre-defined prompt template. This file should 
            have placeholders `{category}` and `{genes}` to be replaced dynamically.

        category : str
            The cancer category or subtype for which the penalty factors are being requested 
            (e.g., "tFL (Transformed Follicular Lymphoma)").

        genes : list[str] or str
            A list of gene names (e.g., `["AASS", "ABCA6", "ABCB1"]`) or a single gene name 
            (if `singular=True`). If `singular=False`, this should be a list of strings, 
            which will be formatted as a comma-separated string in the prompt.

        singular : bool, optional
            Indicates whether `genes` is a single string instead of a list. If `True`, 
            the function will use the string as is. If `False`, it treats `genes` as 
            a list and joins the items with commas. Defaults to `False`.

        display : bool, optional
            If set to `True`, the function prints the dynamically generated prompt to the 
            console for review. Default is `False`.

    Returns
        str
            The final generated prompt with the placeholders `{category}` and `{genes}` 
            replaced by the provided arguments.
    """
    with open(prompt_dir, "r", encoding="utf-8") as file:
        penalty_factors_prompt_template = file.read()

    # Create the PromptTemplate object
    penalty_factors_prompt = PromptTemplate(
        input_variables=["category", "genes"],
        template=penalty_factors_prompt_template
    )

    # Fill in the prompt dynamically
    if not singular:
        # Expecting a list of genes
        filled_prompt = penalty_factors_prompt.format(
            category=category,
            genes=", ".join(genes)
        )
    else:
        # Expecting a single string for genes
        filled_prompt = penalty_factors_prompt.format(
            category=category,
            genes=genes
        )

    # Print the filled prompt if requested
    if display:
        print(filled_prompt)

    return filled_prompt


def save_responses_to_file(responses, file_path):
    """
    Saves only the responses from LLMs to a text file.

    Args:
        results: List of responses (batch outputs) from LLM.
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


def extract_scores_from_responses(responses, genenames, dict=False):
    """
    Extracts scores from LLM responses corresponding to the given gene names.

    Args:
        responses (list[str]): List of LLM responses as strings.
        genenames (list[str]): List of gene names to filter scores.
        dict (bool, optional): If True, returns a dictionary of gene names and their scores. Default is False.

    Returns:
        list[float or None] or dict[str, float or None]: A list of scores corresponding to the genenames in order, or a
        dictionary mapping gene names to their scores if dict=True.
    """
    # Normalize gene names
    normalized_genenames = normalize_genenames(genenames)
    scores = {gene: None for gene in normalized_genenames}  # Initialize with None for all genes
    LARGE_NUMBER = 1e10  # Define a large number to replace infinity values

    for response in responses:
        # Updated regex pattern to allow spaces within gene names
        matches = re.findall(r"\*\*(.*?)\*\*:\s*(-?\d+(?:\.\d+)?|inf(?:inity)?|∞)", response, re.IGNORECASE)

        for gene, score in matches:
            # Normalize gene name from response
            normalized_gene = gene.strip().replace('|', '').replace('/', '').replace('-', '').replace('.', '')

            if normalized_gene in normalized_genenames:
                if score.lower() in {"inf", "infinity", "∞"}:
                    scores[normalized_gene] = LARGE_NUMBER
                else:
                    scores[normalized_gene] = float(score)

    if dict:
        # Return as a dictionary
        return scores
    else:
        # Return as a list in the order of normalized genenames
        return [scores[gene] for gene in normalized_genenames]


# A naiver version of score extraction function, best used in cases where the scores are confined to between 0,1.
def extract_scores_from_responses_01(responses, dummy):
    """
    Extracts all numbers between 0 and 1 from LLM responses.

    Args:
        responses: List of LLM-generated responses as strings.

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



def retrieval_docs(batch_genes, category, retriever, small=False):
    """
    Retrieve relevant documents for a batch of genes under a specified category.

    When `small` is set to `False`, this function individually retrieves documents 
    for each gene to avoid token limit issues for larger models like llama-3b-instruct. 
    Otherwise, it aggregates all genes into a single query to reduce the total number 
    of calls to the retrieval model.

    :param batch_genes: (list[str])
        A list of gene names for which relevant documents need to be retrieved 
        (e.g., `["AASS", "ABCA6", "ABCB1"]`).

    :param category: (str)
        The category or domain context (e.g., a type of cancer or condition) used 
        to refine the document retrieval query.

    :param retriever: 
        An object or instance providing the `get_relevant_documents` method. 
        This is typically a component of a retrieval-augmented system or 
        language model pipeline.

    :param small: (bool, optional)
        If `False`, each gene is queried separately to keep the input prompt size smaller. 
        If `True`, all genes are combined into a single query. Defaults to `False`.

    :return: (list)
        A list of retrieved documents relevant to the specified genes and category. 
        Each item in this list is typically a document object or dictionary-like structure 
        containing text and metadata.
    """
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
