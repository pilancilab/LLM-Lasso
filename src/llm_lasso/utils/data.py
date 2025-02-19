"""
This script implements data processing functions for microarray genetic data used 
in large-scale experiments as well as in the score saving and loading process.
"""

import numpy as np
import pickle
import json
import pandas as pd
import os

def remove_suffix(s):
    """
    Remove the suffix (anything after and including the first underscore) from a string.

    Args:
        s (str): Input string that may contain an underscore and suffix.

    Returns:
        str: The string with any trailing underscore and its suffix removed.
    """
    return s.split('_')[0]


def process_npz_file(input_file_path, output_file_path):
    """
    Process an NPZ file of microarray gene data to eliminate duplicated measurements 
    by taking the average of columns with the same (suffix-stripped) gene name.

    Steps performed:
        1. Alter gene names to eliminate subscripts (suffixes).
        2. Group columns by the updated gene names and calculate the mean across duplicates.
        3. Create new arrays for xall and genenames based on these combined columns.

    Args:
        input_file_path (str): Path to the input NPZ file.
        output_file_path (str): Path to save the modified NPZ file.

    Returns:
        dict: A dictionary containing the updated 'xall', 'yall', 'yclass', 
              and 'genenames' arrays.
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
        'yall': data['yall'],    # Keep yall unchanged
        'yclass': data['yclass'],  # Keep yclass unchanged
        'genenames': new_genenames
    }

    # Save the new data as an NPZ file
    np.savez(output_file_path, **new_data)
    print(f"Processed data saved to: {output_file_path}")

    return new_data


def save_dictionary_as_json(gene_importance_dict, file_path):
    """
    Save a dictionary to a JSON file.

    Args:
        gene_importance_dict (dict): Dictionary containing gene importance data.
        file_path (str): Path to the output JSON file.
    """
    with open(file_path, 'w') as json_file:
        json.dump(gene_importance_dict, json_file, indent=4)


def save_list_as_pkl(gene_importance_ls, file_path):
    """
    Save a list to a pickle file.

    Args:
        gene_importance_ls (list): A list containing gene importance data.
        file_path (str): Path to the output pickle file.
    """
    with open(file_path, 'wb') as pkl_file:
        pickle.dump(gene_importance_ls, pkl_file)


def load_dictionary_from_json(file_path):
    """
    Load a dictionary from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: The dictionary loaded from the JSON file.
    """
    with open(file_path, 'r') as json_file:
        dictionary = json.load(json_file)
    return dictionary


def save_genenames_to_txt(genenames, file_path):
    """
    Save a list of gene names to a .txt file.

    Args:
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


def load_dictionary_from_pkl(file_path):
    """
    Load a dictionary (or other Python object) from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        Any: The object loaded from the pickle file (commonly a dictionary).
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_scores_from_dict(importance_dict):
    """
    Load scores from a dictionary by extracting its values into a list.

    Args:
        importance_dict (dict): Dictionary where values represent scores.

    Returns:
        list: List of scores extracted from the dictionary's values.
    """
    return list(importance_dict.values())


def load_scores_from_txt(file_path):
    """
    Load scores (floats) from a .txt file into a list.

    Each line in the .txt file should contain a single numeric value.

    Args:
        file_path (str): Path to the .txt file containing scores.

    Returns:
        list: A list of scores (floats). If a line is not numeric, it will print an error.
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


def load_feature_names(file_path):
    """
    Load feature names from a .txt or .pkl file.

    Args:
        file_path (str): Path to the file containing feature names.

    Returns:
        list: List of feature names.
    
    Raises:
        ValueError: If the file extension is not supported (not .txt or .pkl).
    """
    if file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            return [line.strip() for line in f]
    else:
        raise ValueError("Unsupported file format. Only .txt and .pkl are supported.")


def convert_pkl_to_txt(input_dir, output_dir):
    """
    Convert a pickle file (.pkl) containing a list into a .txt file.

    Args:
        input_dir (str): Path to the input .pkl file.
        output_dir (str): Path to the output .txt file.

    Returns:
        None
    """
    with open(input_dir, 'rb') as pkl_file:
        my_list = pickle.load(pkl_file)

    # Save the content to a .txt file
    with open(output_dir, 'w') as txt_file:
        for item in my_list:
            txt_file.write(str(item) + '\n')
