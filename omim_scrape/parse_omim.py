"""
Pulls down OMIM data via HTTP request.
"""

__author__ = 'Erica Zhang'

import requests
import xml.etree.ElementTree as Tree
from Expert_RAG import constants
import pickle as pkl
from tqdm import tqdm
import time

KEY = constants.OMIM_API

# 1. Search for mim-number given a hgnc gene name
def test_omim_api_access():
    """
    Test access to the OMIM API with a sample query.
    Returns:
        bool: True if the test succeeds, False otherwise.
    """
    url = "https://api.omim.org/api/entry/search"
    # Query string for 'search' parameter manually constructed
    search_query = "+ABHD6+gene"  # Example: use the exact query format required by OMIM

    # Remaining parameters
    params = {
        "start": 0,
        "sort": "score desc",
        "limit": 1,
        "apiKey": KEY,
        "format": "xml",  # Ensure the response is in XML format
    }

    # Manually append the 'search' query to the URL
    full_url = f"{url}?search={search_query}"

    try:
        # Send the GET request
        response = requests.get(full_url, params=params)

        # Check for HTTP success status
        if response.status_code == 200:
            print("Success! API is accessible.")
            print("Response:\n", response.text[:50])  # Print a snippet of the response
            return True
        else:
            print(f"Failed response URL:{response.url}")
            print(f"Failed! Status code: {response.status_code}")
            print("Response:\n", response.text[:50])  # Print a snippet of the response
            return False
    except requests.exceptions.RequestException as e:
        print(f"Error accessing the API: {e}")
        return False


def get_mim_number(gene, quiet=False): # works with cancer too
    """
    Query OMIM API to fetch the mimNumber for a given gene.
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
        "apiKey": KEY,
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
                print(f"No mimNumber found for gene: {gene}")
            return None
    except requests.exceptions.RequestException as e:
        if not quiet:
            print(f"Error fetching mimNumber for gene {gene}: {e}")
        return None


# 2. parse omim text
def get_all_mim(file_path, save_path = 'allMim_gene.pkl', description = False): # works with cancer list too
    """
    Fetch mimNumbers for a list of genes and return as a dictionary.

    Args:
        file_path (path to the pkl file)
        description (bool)
    Return:
        dictionary
    """
    mim_numbers = {}
    ls = []
    with open(file_path, "rb") as file:
        gene_list = pkl.load(file)
    for gene in tqdm(gene_list, desc="Processing"):
        if description:
            print(f"Fetching mimNumber for gene: {gene}...")
        mim_number = get_mim_number(gene)
        if mim_number:
            mim_numbers[gene] = mim_number
            ls.append(mim_number)
        time.sleep(0.1)
    with open(save_path, "wb") as file:
        pkl.dump(ls, file)
    return mim_numbers


# fetch data using mimNumber in OMIM.org

def fetch_omim_data(mim_number):
    """
    Fetch OMIM data for a given mimNumber from the API.
    """
    url = f"https://api.omim.org/api/entry"
    params = {
        "mimNumber": mim_number,
        "include": "text,geneMap,clinicalSynopsis",
        "format": "xml",
        "apiKey": KEY
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # raise an exception if the HTTP request returned an unsuccessful status code.
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for mimNumber {mim_number}: {e}")
        return None


# def parse_omim_response(response_text):
#     """
#     Parse the OMIM XML response to extract text description, gene map, and titles.
#     """
#     try:
#         root = Tree.fromstring(response_text)
#         entry = root.find(".//entry")
#
#         # Extract titles
#         preferred_title = entry.findtext(".//preferredTitle", default="Unknown Gene")
#         gene_symbols = entry.findtext(".//geneSymbols", default="Unknown Symbol").split(",")[0].strip()
#
#         # Extract text description
#         text_sections = entry.findall(".//textSection")
#         text_description = []
#         for section in text_sections:
#             section_title = section.findtext("textSectionTitle", default="No Title")
#             section_content = section.findtext("textSectionContent", default="No Content")
#             text_description.append(f"{section_title}:\n{section_content}")
#         text_description = "\n\n".join(text_description)
#
#         # Extract gene map
#         gene_map = entry.find(".//geneMap")
#         gene_map_data = []
#         if gene_map is not None:
#             for child in gene_map:
#                 gene_map_data.append(f"{child.tag}: {child.text}")
#         gene_map_data = "\n".join(gene_map_data)
#
#         return gene_symbols, preferred_title, text_description, gene_map_data # preferred_title is the full name of the gene
#     except Exception as e:
#         print(f"Error parsing response: {e}")
#         return None, None, None, None


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
        text_description = []
        if text_sections:
            for section in text_sections:
                section_title = section.findtext("textSectionTitle", default="No Title")
                section_content = section.findtext("textSectionContent", default="No Content")
                text_description.append(f"{section_title}:\n{section_content}")
        text_description = "\n\n".join(text_description) if text_description else "No Text Description Available"

        # Extract gene map
        gene_map = entry.find(".//geneMap")
        gene_map_data = []
        if gene_map is not None:
            for child in gene_map:
                if child.tag and child.text:  # Ensure valid content
                    gene_map_data.append(f"{child.tag}: {child.text.strip()}")
        gene_map_data = "\n".join(gene_map_data) if gene_map_data else "No Gene Map Data Available"

        # Extract clinical synopsis
        clinical_synopsis = entry.find(".//clinicalSynopsis")
        clinical_synopsis_data = []
        if clinical_synopsis is not None:
            for child in clinical_synopsis:
                if child.tag and child.text:  # Ensure valid content
                    clinical_synopsis_data.append(f"{child.tag}: {child.text.strip()}")
        clinical_synopsis_data = "\n".join(clinical_synopsis_data) if clinical_synopsis_data else "No Clinical Synopsis Available"

        # Return parsed data
        return gene_symbols, preferred_title, text_description, gene_map_data, clinical_synopsis_data

    except Exception as e:
        print(f"Error parsing response: {e}")
        return None, None, None, None, None


def save_to_txt(gene_symbol, preferred_title, mim_number, text_description, gene_map_data, output_file):
    """
    Save the extracted data to a text file.
    """
    with open(output_file, "a") as f:
        f.write(f"Cancer Data for {gene_symbol} ({preferred_title}) (MIM Number: {mim_number})\n")
        f.write("=" * 50 + "\n")
        f.write("TEXT DESCRIPTION:\n")
        f.write("=" * 50 + "\n")
        f.write(f"{text_description}\n\n")
        f.write("GENE MAP:\n")
        f.write("=" * 50 + "\n")
        f.write(f"{gene_map_data}\n\n")
        f.write("=" * 50 + "\n\n")


def process_mim_numbers(input_file, output_file, description = True):
    """
    Process a list of mimNumbers and save the results to a text file.
    Input_file is assumed to be in pkl format.
    """
    # Clear the output file
    open(output_file, "w").close()

    with open(input_file, "rb") as file:
        mim_numbers = pkl.load(file)
    for mim_number in tqdm(mim_numbers,desc="Processing"):
        if description:
            print(f"Processing MIM Number: {mim_number}")
        response_text = fetch_omim_data(mim_number)
        if response_text:
            gene_symbol, preferred_title, text_description, gene_map_data = parse_omim_response(response_text)
            if gene_symbol and preferred_title and text_description and gene_map_data:
                save_to_txt(gene_symbol, preferred_title, mim_number, text_description, gene_map_data, output_file)
        else:
            print(f"Skipping MIM Number: {mim_number}")
        time.sleep(0.1)

    print(f"Data saved to {output_file}")


# tests
if __name__ == "__main__":
    # 1. ensure API access is functional
    # test_omim_api_access()
    # 2. test fetching mimNumber for a single gene
    num = get_mim_number('ABHD6')
    print(num)
    # 3. Try fetching genes for the first 5 entries of Ash
    # mim_dict = get_all_mim_gene("genes1592.pkl")
    # print(mim_dict)
    # 4. fetching text data and gene map from mim number
    # process_mim_numbers("allMim_gene.pkl", "all_gene.txt")
    # 5. fetch cancer data
    cancer_list = [151400, 113970, 605207, 613024, 254500, 236000, 186960, 153600] # incomplete list
    with open("allMim_cancer.pkl", "wb") as file:
        pkl.dump(cancer_list, file)
    process_mim_numbers("allMim_cancer.pkl", "omim_scrape/all_cancer.txt")