import pickle
from omim_scrape.omim_json import *

# 1. Collect all valid omim numbers from mim2gene.txt
def collect_valid_mim_numbers(file_path, output_txt="omim_scrape/omim_all/valid_mim_numbers.txt", output_pkl="omim_scrape/omim_all/valid_mim_numbers.pkl"):
    """
    Collects a list of Mim numbers from a text file, excluding entries with "moved/removed",
    and saves the results to a text file and a pickle file.

    Parameters:
        file_path (str): Path to the input text file.
        output_txt (str): Path to save the valid Mim numbers as a text file. Default is 'valid_mim_numbers.txt'.
        output_pkl (str): Path to save the valid Mim numbers as a pickle file. Default is 'valid_mim_numbers.pkl'.

    Returns:
        list: A list of valid Mim numbers.
    """
    # Initialize an empty list to store valid Mim numbers
    valid_mim_numbers = []

    # Read the file and process each line
    with open(file_path, "r") as file:
        for line in file:
            # Split the line into columns based on tab delimiter
            columns = line.strip().split("\t")

            # Extract the Mim number and the corresponding entry
            mim_number = columns[0]  # First column
            entry = columns[1]       # Second column

            # Check if the entry is not "moved/removed"
            if "moved/removed" not in entry:
                valid_mim_numbers.append(mim_number)

    # Save the valid Mim numbers to a text file
    with open(output_txt, "w") as txt_file:
        txt_file.write("\n".join(valid_mim_numbers))

    # Save the valid Mim numbers to a pickle file
    with open(output_pkl, "wb") as pkl_file:
        pickle.dump(valid_mim_numbers, pkl_file)

    return valid_mim_numbers

# 2. Collect only the mimNumbers which correspond to gene instead of phenotypes.
def collect_gene_mim_numbers(file_path, output_txt="omim_scrape/omim_all/gene_mim_numbers.txt", output_pkl="omim_scrape/omim_all/gene_mim_numbers.pkl"):
    """
    Collects a list of Mim numbers corresponding to entries labeled as "gene",
    and saves the results to a text file and a pickle file.

    Parameters:
        file_path (str): Path to the input text file.
        output_txt (str): Path to save the gene Mim numbers as a text file. Default is 'gene_mim_numbers.txt'.
        output_pkl (str): Path to save the gene Mim numbers as a pickle file. Default is 'gene_mim_numbers.pkl'.

    Returns:
        list: A list of Mim numbers corresponding to entries labeled as "gene".
    """
    # Initialize an empty list to store gene Mim numbers
    gene_mim_numbers = []

    # Read the file and process each line
    with open(file_path, "r") as file:
        for line in file:
            # Split the line into columns based on tab delimiter
            columns = line.strip().split("\t")

            # Extract the Mim number and the corresponding entry
            mim_number = columns[0]  # First column
            entry = columns[1]       # Second column

            # Check if the entry is "gene"
            if entry == "gene":
                gene_mim_numbers.append(mim_number)

    # Save the gene Mim numbers to a text file
    with open(output_txt, "w") as txt_file:
        txt_file.write("\n".join(gene_mim_numbers))

    # Save the gene Mim numbers to a pickle file
    with open(output_pkl, "wb") as pkl_file:
        pickle.dump(gene_mim_numbers, pkl_file)

    return gene_mim_numbers



if __name__ == "__main__":
    # # 1. Scrape mimNumbers.
    # file_path = 'omim_scrape/omim_all/mim2gene.txt'
    # valid_mim_numbers = collect_valid_mim_numbers(file_path)
    # print(len(valid_mim_numbers))
    # gene_numbers = collect_gene_mim_numbers(file_path)
    # print(len(gene_numbers))

    # 2. parse json file for collected mimNumbers.
    output_json = "omim_scrape/omim_all/omim_all.json"
    valid_mim_numbers = "omim_scrape/omim_all/valid_mim_numbers.pkl"
    process_mim_numbers_to_json(valid_mim_numbers, output_json, description=True)

