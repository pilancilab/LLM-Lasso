"""
This script processes the mim2gene.txt file (provided from Omim API) to collect all valid OMIM numbers, with an option to collect the subset of gene-specific mim numbers in the OMIM database. 
It also offers additional MimNumber processing functions for a partial specified list of genes if user wishes to focus on a specific subset of the OMIM database.
Additionally, it provides a function to test the accessibility of the OMIM API.
"""

import pickle


KEY = "" # API key for OMIM API

# Utility function
def test_omim_api_access():
    """
    Test access to the OMIM API with a sample query.
    Returns:
        bool: True if the test succeeds, False otherwise.
    """
    url = "https://api.omim.org/api/entry/search"
    # Query string for 'search' parameter manually constructed
    search_query = "+ABHD6+gene"  # Example: use the exact query format required by OMIM (see formatting on OMIM API documentation)

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
    

######################################### Mim Number Processing Functions ######################################################

# 1. Collect all valid omim numbers from mim2gene.txt
def collect_all_valid_mim_numbers(file_path, output_txt="omim_scrape/omim_all/valid_mim_numbers.txt", output_pkl="omim_scrape/omim_all/valid_mim_numbers.pkl"):
    """
    Collects a list of Mim numbers from the mim2gene.txt file, excluding entries with "moved/removed",
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
def collect_all_gene_mim_numbers(file_path, output_txt="omim_scrape/omim_all/gene_mim_numbers.txt", output_pkl="omim_scrape/omim_all/gene_mim_numbers.pkl"):
    """
    Collects a list of Mim numbers from mim2gene.txt file, corresponding to entries labeled as "gene",
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

# 3. Collect subset of mimNumbers for a list of specific gene or phenotype.
def get_mim_number(gene, quiet=False): 
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
                print(f"No mimNumber found for gene/phenotype: {gene}")
            return None
    except requests.exceptions.RequestException as e:
        if not quiet:
            print(f"Error fetching mimNumber for gene/phenotype {gene}: {e}")
        return None


def get_specified_mim(file_path, save_path = 'Mim_specific.pkl', description = False):
    """
    Fetch mimNumbers for a specified list of genes/phenotypes and return as a dictionary.

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
            print(f"Fetching mimNumber for gene/phenotype: {gene}...")
        mim_number = get_mim_number(gene)
        if mim_number:
            mim_numbers[gene] = mim_number
            ls.append(mim_number)
        time.sleep(0.1)
    with open(save_path, "wb") as file:
        pkl.dump(ls, file)
    return mim_numbers




# if __name__ == "__main__":
    # # 1. ensure API access is functional
    # test_omim_api_access()

    # # 2. test fetching mimNumber for a single gene
    # num = get_mim_number('ABHD6')
    # print(num)

    # # 3. test fetching mimNumber for a list of genes
    # file_path = 'omim_scrape/genes1592.pkl' # path to the pkl file containing gene names
    # mim_dict = get_specified_mim(file_path)
    # print(mim_dict)

    # # 4. Scrape mimNumbers.
    # file_path = 'omim_scrape/mim2gene.txt'
    # valid_mim_numbers = collect_all_valid_mim_numbers(file_path)
    # print(len(valid_mim_numbers))
    # gene_numbers = collect_all_gene_mim_numbers(file_path)
    # print(len(gene_numbers))

    # # 5. scrape and parse contents to json file for collected mimNumbers.
    # # see omim_all.py

