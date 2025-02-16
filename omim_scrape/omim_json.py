import json
import time
from tqdm import tqdm
from omim_scrape.parse_omim import fetch_omim_data, parse_omim_response
import pickle as pkl
import os


def save_to_json(gene_symbol, preferred_title, mim_number, text_description, gene_map_data, clinical_synopsis, output_file):
    """
    Save the extracted gene data to a JSON file, including clinical synopsis.
    """
    # Create the gene data structure
    gene_data = {
        "gene_name": gene_symbol,
        "preferred_title": preferred_title,
        "mim_number": mim_number,
        "text_description": text_description,
        "gene_map_data": gene_map_data,
        "clinical_synopsis": clinical_synopsis,  # Add clinical synopsis
    }

    # Append the data to the JSON file
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(gene_data, ensure_ascii=False) + "\n")  # Write each gene as a JSON object




def process_mim_numbers_to_json(input_file, output_file, description=True):
    """
    Process a list of MIM numbers, fetch their data, and save the results to a JSON file.

    Args:
        input_file (str): Path to the file containing MIM numbers (in pickle format).
        output_file (str): Path to save the JSON output.
        description (bool): Whether to print progress messages.
    """
    # Ensure output file is clean before starting
    open(output_file, "w").close()

    # Load the MIM numbers from the input pickle file
    with open(input_file, "rb") as file:
        mim_numbers = pkl.load(file)

    for mim_number in tqdm(mim_numbers, desc="Processing MIM Numbers"):
        if description:
            print(f"Processing MIM Number: {mim_number}")

        # Fetch the OMIM data for the current MIM number
        response_text = fetch_omim_data(mim_number)
        if response_text:
            # Parse the response to extract relevant information
            gene_symbol, preferred_title, text_description, gene_map_data, clinical_synopsis = parse_omim_response(response_text)
            if any([gene_symbol, preferred_title, text_description, gene_map_data, clinical_synopsis]):
                # Save the parsed data to the JSON file
                save_to_json(gene_symbol, preferred_title, mim_number, text_description, gene_map_data, clinical_synopsis, output_file)
            else:
                print(f"Incomplete data for MIM Number: {mim_number}, skipping.")
        else:
            print(f"Failed to fetch data for MIM Number: {mim_number}, skipping.")

        # Add a short delay to avoid overwhelming the API
        time.sleep(0.1)

    print(f"Data successfully saved to {output_file}")



if __name__ == "__main__":

    # # Define the JSON output file
    # output_json = "omim_scrape/all_cancer.json"
    #
    # # Process the MIM numbers and save to JSON
    # process_mim_numbers_to_json('omim_scrape/allMim_cancer.pkl', output_json, description=True)
    #
    # # Check if the JSON file was created
    # if os.path.exists(output_json):
    #     print(f"JSON output saved to: {output_json}")
    #     # Display the first few lines for verification
    #     # with open(output_json, "r", encoding="utf-8") as file:
    #     #     print("\nSample JSON output:")
    #     #     for i, line in enumerate(file):
    #     #         print(line.strip())
    #     #         if i >= 2:  # Display only the first 3 entries
    #     #             break
    # else:
    #     print("Failed to create JSON output file.")

    # Define the JSON output file
    output_json = "omim_scrape/all_gene.json"

    # Process the MIM numbers and save to JSON
    process_mim_numbers_to_json('omim_scrape/allMim_gene.pkl', output_json, description=True)

    # Check if the JSON file was created
    if os.path.exists(output_json):
        print(f"JSON output saved to: {output_json}")
        # Display the first few lines for verification
        # with open(output_json, "r", encoding="utf-8") as file:
        #     print("\nSample JSON output:")
        #     for i, line in enumerate(file):
        #         print(line.strip())
        #         if i >= 2:  # Display only the first 3 entries
        #             break
    else:
        print("Failed to create JSON output file.")

    # # debug all_cancer
    # with open('omim_scrape/allMim_cancer.pkl', "rb") as file:
    #     cancer_mim_list = pkl.load(file)
    #     print("\nContent of 'allMim_cancer.pkl':")
    #     print(cancer_mim_list) # passed: [151400, 113970, 605207, 613024, 254500, 236000, 186960, 153600]

