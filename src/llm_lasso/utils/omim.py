"""
This script implements functions to fetch and process OMIM data for specified genes or phenotypes.
"""
import os
import requests
import xml.etree.ElementTree as Tree
import pickle as pkl
from tqdm import tqdm
import time

def get_mim_number(gene, api_key, quiet=False): 
    """
    Query OMIM API to fetch the mimNumber for a given gene or phenotype.
    Args:
        gene (str): hgnc gene name.
    """
    base_url = "https://api.omim.org/api/entry/search"
    # Query string for 'search' parameter manually constructed
    search_query = f"{gene}"  # Use the exact query format required by OMIM; consider enhancement with f"+{gene}+gene"

    # Remaining parameters
    params = {
        "start": 0,
        "sort": "score desc",
        "limit": 1,
        "apiKey": api_key,
        "format": "xml",  # Ensure the response is in XML format
    }

    # Manually append the 'search' query to the URL
    full_url = f"{base_url}?search={search_query}"

    try:
        # Send the HTTP GET request
        response = requests.get(full_url, params=params)
        response.raise_for_status()

        # Parse the XML response
        root = Tree.fromstring(response.text)

        # Locate the mimNumber in the response
        mim_number_element = root.find(".//mimNumber")
        if mim_number_element is not None:
            return mim_number_element.text
        else:
            if not quiet:
                print(f"No mimNumber found for gene/phenotype: {gene}")
            return None
    except requests.exceptions.RequestException as e:
        if not quiet:
            print(f"Error fetching mimNumber for gene/phenotype {gene}: {e}")
        return None


def get_specified_mim(file_path, api_key, save_path='specific_mim.pkl', description=False):
    """
    Fetch mimNumbers for a specified list of genes/phenotypes (from file_path)
    and return as a dictionary.

    Args:
        file_path (str): Path to either a .pkl or .txt file containing gene/phenotype names.
        save_path (str): Path to save the list of MIM numbers as a .pkl file. Defaults to 'specific_mim.pkl'.
        description (bool): Whether to print progress messages.

    Returns:
        dict: A dictionary where the keys are gene/phenotype names and the values are mimNumbers.
    """
    
    # Determine the file extension
    _, file_ext = os.path.splitext(file_path)
    file_ext = file_ext.lower()
    
    # Load data based on extension
    if file_ext == '.pkl':
        # Load via pickle
        with open(file_path, 'rb') as f:
            gene_list = pkl.load(f)
    elif file_ext == '.txt':
        # Load from text file (one gene/phenotype per line)
        with open(file_path, 'r') as f:
            gene_list = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")

    mim_numbers = {}
    ls = []

    # Iterate over each gene/phenotype
    for gene in tqdm(gene_list, desc="Processing"):
        if description:
            print(f"Fetching mimNumber for gene/phenotype: {gene}...")
        mim_number = get_mim_number(gene, api_key)  # <--- make sure this function is defined elsewhere
        if mim_number:
            mim_numbers[gene] = mim_number
            ls.append(mim_number)
        time.sleep(0.1)

    # Save the list of MIM numbers as a pickle
    with open(save_path, 'wb') as f:
        pkl.dump(ls, f)

    return mim_numbers


def fetch_omim_data(mim_number, api_key):
    """
    Fetch OMIM data using the current API key, switching keys when the limit is reached.
    
    Args:
        mim_number (str): The MIM number to fetch data for.
    Returns:
        str: The response text from the API, or None if an error occurs.
    """
    url = f"https://api.omim.org/api/entry"
    params = {
        "mimNumber": mim_number,
        "include": "text,geneMap,clinicalSynopsis",
        "format": "xml",
        "apiKey": api_key,
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for unsuccessful status codes
        return response.text
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for MIM number {mim_number}: {e}")
        print("Terminating program due to API failure.")
        exit(1)


def parse_omim_response(response_text):
    """
    Parse the OMIM XML response to extract text description, gene map, titles, and clinical synopsis.
    Returns available data even if some fields are missing.
    """
    try:
        root = Tree.fromstring(response_text)
        entry = root.find(".//entry")

        # Extract titles
        preferred_title = entry.findtext(".//preferredTitle", default="Unknown Gene")
        gene_symbols = entry.findtext(".//geneSymbols", default="Unknown Symbol").split(",")[0].strip()

        # Extract text description
        text_sections = entry.findall(".//textSection")
        text_description = "\n\n".join(
            f"{section.findtext('textSectionTitle', 'No Title')}:\n{section.findtext('textSectionContent', 'No Content')}"
            for section in text_sections
        ) if text_sections else "No Text Description Available"

        # Extract gene map
        gene_map = entry.find(".//geneMap")
        gene_map_data = "\n".join(
            f"{child.tag}: {child.text.strip()}" for child in gene_map if child.tag and child.text
        ) if gene_map is not None else "No Gene Map Data Available"

        # Extract clinical synopsis
        clinical_synopsis = entry.find(".//clinicalSynopsis")
        clinical_synopsis_data = "\n".join(
            f"{child.tag}: {child.text.strip()}" for child in clinical_synopsis if child.tag and child.text
        ) if clinical_synopsis is not None else "No Clinical Synopsis Available"

        return gene_symbols, preferred_title, text_description, gene_map_data, clinical_synopsis_data

    except Exception as e:
        print(f"Error parsing response: {e}")
        return None, None, None, None, None