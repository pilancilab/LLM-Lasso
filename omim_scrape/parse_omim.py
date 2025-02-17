"""
Pulls down entire OMIM data in JSON format according to a complete list of valid MIM numbers via HTTP request with switching APIs and checkpoints for incremental saving.
"""

import os
import json
import time
import requests
import xml.etree.ElementTree as Tree
import pickle as pkl
from tqdm import tqdm

# List of OMIM API keys
API_KEYS = []
REQUEST_LIMIT = 4000  # Maximum requests per API key
key_index = 0  # Index of the current API key
request_count = 0  # Track the number of requests for the current key


def get_api_key():
    """
    Rotate to the next API key if the current key reaches its request limit.
    Returns:
        str: The current API key.
    """
    global key_index, request_count
    if request_count >= REQUEST_LIMIT:
        key_index = (key_index + 1) % len(API_KEYS)
        request_count = 0
        print(f"Switching to API key {key_index + 1}")
    return API_KEYS[key_index]


def fetch_omim_data_with_key(mim_number):
    """
    Fetch OMIM data using the current API key, switching keys when the limit is reached.
    
    Args:
        mim_number (str): The MIM number to fetch data for.
    Returns:
        str: The response text from the API, or None if an error occurs.
    """
    global request_count
    api_key = get_api_key()
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
        request_count += 1
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


def load_checkpoint(output_file):
    """
    Load processed data from an existing JSON file to resume processing.
    
    Args:
        output_file (str): Path to the output JSON file.
    Returns:
        set: Set of already processed MIM numbers.
    """
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            processed_data = {json.loads(line)["mim_number"] for line in f}
        print(f"Resuming from checkpoint. {len(processed_data)} entries loaded.")
        return processed_data
    return set()


def process_mim_numbers_to_json(input_file, output_file, description=True):
    """
    Process a list of MIM numbers, fetch their data, and save the results to a JSON file.

    Args:
        input_file (str): Path to the file containing MIM numbers (in pickle format).
        output_file (str): Path to save the JSON output.
        description (bool): Whether to print progress messages.
    """
    processed_mim_numbers = load_checkpoint(output_file)

    with open(input_file, "rb") as file:
        mim_numbers = pkl.load(file)

    with open(output_file, "a", encoding="utf-8") as f:
        for mim_number in tqdm(mim_numbers, desc="Processing MIM Numbers"):
            if mim_number in processed_mim_numbers:
                continue  # Skip already processed MIM numbers

            if description:
                print(f"Processing MIM Number: {mim_number}")

            response_text = fetch_omim_data_with_key(mim_number)
            if response_text:
                gene_symbol, preferred_title, text_description, gene_map_data, clinical_synopsis = parse_omim_response(response_text)
                if any([gene_symbol, preferred_title, text_description, gene_map_data, clinical_synopsis]):
                    gene_data = {
                        "gene_name": gene_symbol,
                        "preferred_title": preferred_title,
                        "mim_number": mim_number,
                        "text_description": text_description,
                        "gene_map_data": gene_map_data,
                        "clinical_synopsis": clinical_synopsis,
                    }
                    f.write(json.dumps(gene_data, ensure_ascii=False) + "\n")
                else:
                    print(f"Incomplete data for MIM Number: {mim_number}, skipping.")
            else:
                print(f"Failed to fetch data for MIM Number: {mim_number}, skipping.")

            # Add a short delay to avoid overwhelming the API
            time.sleep(0.1)

    print(f"Data successfully saved to {output_file}")


if __name__ == "__main__":
    # Example
    output_json = "omim_scrape/valid_mim_numbers.json" # Example path to save the JSON output
    process_mim_numbers_to_json("omim_scrape/valid_mim_numbers.pkl", output_json, description=True)
