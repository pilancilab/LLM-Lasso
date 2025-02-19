import numpy as np
import pickle
import json
import pandas as pd
import os

# helper function for cleaning up genes1882.pkl
def remove_suffix(s):
    return s.split('_')[0]

# process raw LLM txt outputs to dictionary
def load_gene_importance(file_path):
    gene_importance_dict = {}
    
    with open(file_path, "r") as file:
        for line in file:
            # Strip whitespace and split by ':' to get key and value
            key_value = line.strip().split(":")
            if len(key_value) == 2:  # Ensure there are exactly 2 elements
                key = key_value[0].strip()  # The gene name
                value = float(key_value[1].strip())  # The score, converted to a float
                gene_importance_dict[key] = value    
    return gene_importance_dict

# save dictionary
def save_dictionary_as_json(gene_importance_dict, file_path):
    with open(file_path, 'w') as file:
        pickle.dump(gene_importance_dict, file)

# save list/array
def save_list_as_pkl(gene_importance_ls, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(gene_importance_ls, json_file, indent=4)

# load json
def load_dictionary_from_json(file_path):
    with open(file_path, 'r') as json_file:
        dictionary = json.load(json_file)
    return dictionary

# load list/array
def load_dictionary_from_pkl(file_path):
    with open(file_path, 'r') as file:
        ls = pickle.load(file)
    return ls


# load values of dictionary as list (for loading importance score from dictionary)
def importance_score(importance_dict):
    return list(importance_dict.values())


# load importance matrix for all cancer types (excluding heealthy control)
def load_importance_matrix(directory):
    # Define the order of the stacking based on the specified list
    order = ['BL', 'CLL', 'DLBCL', 'FL', 'MCL', 'MM', 'PMBCL', 'TCL', 'cHL', 'tFL']
    
    # Initialize an empty list to store the importance data
    importance_data = []
    
    # Iterate over the specified order
    for label in order:
        # Construct the file name based on the current label
        filename = f"{label}_4omini_importance.json"
        file_path = os.path.join(directory, filename)
        
        # Check if the file exists
        if os.path.isfile(file_path):
            # Load the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)  # Load data from JSON file
                # Append the values from the dictionary to the list
                if isinstance(data, dict):
                    importance_data.append(list(data.values()))  # Convert dict values to list and append
        else:
            print(f"File not found: {file_path}")

    # Create a DataFrame from the collected data
    importance_matrix = pd.DataFrame(importance_data)
    
    # Return the DataFrame
    return importance_matrix



# the complete importance matrix with healthy (where it is calculated as the weighted average (could be contentious since we are using posteriori knowledge here) of the cancer importance scores.
def calculate_weighted_importance(importance_matrix, yclass):
    """
    Calculate the weighted average importance scores for the 'Healthy cfDNA' category
    based on occurrences of the cancer types.

    Parameters:
    - importance_matrix: DataFrame containing importance scores stacked vertically according to category_order.
    - yclass: ndarray of labels corresponding to the subjects.

    Returns:
    - Updated importance matrix with the 'Healthy cfDNA' weighted importance row appended.
    """
    unique_categories, class_counts = np.unique(yclass, return_counts=True)

    # masking out healthy category in class_counts too
    health_index = np.nonzero(unique_categories == 'Healthy cfDNA')[0][0]
    cancer_counts = np.delete(class_counts, health_index)
    # get the resulting weight
    cancer_weight = cancer_counts/(np.sum(cancer_counts))

    # health importance score
    health_row = np.sum(importance_matrix * cancer_weight[:, np.newaxis], axis=0)
    augmented_matrix = np.insert(importance_matrix, health_index, health_row, axis=0)

    # create pd dataframe
    augmented_matrix = pd.DataFrame(augmented_matrix)

    return augmented_matrix



# weighted average array of cancer type sum
def cancer_weighted(importance_matrix, yclass):
    unique_categories, class_counts = np.unique(yclass, return_counts=True)

    # masking out healthy category in class_counts too
    health_index = np.nonzero(unique_categories == 'Healthy cfDNA')[0][0]
    cancer_counts = np.delete(class_counts, health_index)
    # get the resulting weight
    cancer_weight = cancer_counts/(np.sum(cancer_counts))

    # weighted importance score
    weighted_row = np.sum(importance_matrix * cancer_weight[:, np.newaxis], axis=0)

    return weighted_row



# process npz file to eliminate duplicate measurement by taking average
def process_npz_file(input_file_path, output_file_path):
    """
    Processes an NPZ file by updating the genenames and xall arrays.
    
    Args:
    - input_file_path: Path to the input NPZ file.
    - output_file_path: Path to save the modified NPZ file.
    """
    # Load the NPZ file
    data = np.load(input_file_path, allow_pickle=True)

    # Extract xall, yall, yclass, and genenames
    xall = data['xall']
    genenames = data['genenames']

    # Step (i): Alter genenames to eliminate subscripts
    updated_genenames = [name.rsplit('_', 1)[0] for name in genenames]

    # Step (ii): Create a DataFrame for easier manipulation
    xall_df = pd.DataFrame(xall, columns=updated_genenames)

    # Group by the new gene names and calculate the mean
    combined_data = xall_df.groupby(xall_df.columns, axis=1).mean()

    # Step (iii): Create the new data structure
    new_xall = combined_data.values  # Updated xall
    new_genenames = combined_data.columns.values  # Updated genenames

    # Preparing the new data to save
    new_data = {
        'xall': new_xall,
        'yall': data['yall'],  # Keep yall unchanged
        'yclass': data['yclass'],  # Keep yclass unchanged
        'genenames': new_genenames
    }

    # Save the new data as an NPZ file
    np.savez(output_file_path, **new_data)

    print(f"Processed data saved to: {output_file_path}")

    return new_data


def load_scores_from_txt(file_path):
    """
    Load scores from a .txt file into a list.

    Parameters:
        file_path (str): Path to the .txt file containing scores (one per line).

    Returns:
        List[float]: A list of scores.
    """
    scores = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip whitespace and convert the line to a float
                scores.append(float(line.strip()))
    except ValueError:
        print("Error: The file contains non-numeric data.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
    return scores


def save_genenames_to_txt(genenames, file_path):
    """
    Save a list of gene names to a .txt file.

    Parameters:
        genenames (list): List of gene names to save.
        file_path (str): Path to the .txt file where the gene names will be saved.

    Returns:
        None
    """
    try:
        with open(file_path, 'w') as file:
            for name in genenames:
                file.write(name + '\n')
        print(f"Gene names successfully saved to {file_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")


def load_feature_names(file_path):
    """
    Load feature names from a .txt or .pkl file.

    Parameters:
    - file_path (str): Path to the file containing feature names.

    Returns:
    - list: List of feature names.
    """
    if file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            return [line.strip() for line in f]
    else:
        raise ValueError("Unsupported file format. Only .txt and .pkl are supported.")


# convert pkl files to txt files
def convert_pkl_to_txt(input_dir, output_dir):
    with open(input_dir, 'rb') as pkl_file:
        my_list = pickle.load(pkl_file)

    # Save the content to a .txt file
    with open(output_dir, 'w') as txt_file:
        for item in my_list:
            txt_file.write(str(item) + '\n')