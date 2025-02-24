"""
Utils for processing small-scale data for 3 classification and 2 regression.

**Classification**:
1. Glioma Grading Clinical and Mutation Features (2022): (839,23)
2. Bank (2012): Predict whether a client at a bank will subscribe to a term deposit.
3. Diabetes (1988): Predict whether a female adult patient of Pima Indian heritage has diabetes. (768, 9); (0,1)

**Regression**:
1. Spotify (2024) *: Predict the number of Spotify streams based on quantitative features like Track Score and All Time Rank. (4600, 28) | (565, 22).
2. Wine Quality (2009): Predict whether a wine is high or low quality. (6497,11).
"""

# Import uci datasets
import os
import json
from ucimlrepo import fetch_ucirepo
import pickle
import pandas as pd
import kagglehub

def process_glioma():
    """
    Returns feature and label data for the Glioma UCI dataset, as a `pandas`
    `DataFrame` and `Series`, respectively.
    """
    X, y, feat_names, _ = load_uci_data("Glioma")
    X = pd.DataFrame(X, columns=feat_names)
    y = pd.Series(y)
    X["Race"] = pd.Categorical(X["Race"]).codes

    columns_list = X.columns.tolist()
    save_path = "small_scale/data/Glioma_feature_names.pkl"
    with open(save_path, 'wb') as file:
        pickle.dump(columns_list, file)
    print(f"Feature names saved to: {save_path}")

    return X, y

def process_bank():
    """
    Returns feature and label data for the Bank UCI dataset, as a `pandas`
    `DataFrame` and `Series`, respectively.
    """
    X, y, feat_names, _ = load_uci_data("Bank")
    X = pd.DataFrame(X, columns=feat_names)
    X.drop(["poutcome"], axis=1, inplace=True)
    X["y"] = y
    X = X.dropna()
    X.reset_index(inplace=True)

    for col in ["job", "marital", "education", "default", "month", "housing", "y", "loan", "contact"]:
        X[col] = pd.Categorical(X[col]).codes
    
    y = pd.Series(X["y"].to_numpy())
    X.drop(["y", "index"], axis=1, inplace=True)
    columns_list = X.columns.tolist()
    save_path = "small_scale/data/Bank_feature_names.pkl"
    with open(save_path, 'wb') as file:
        pickle.dump(columns_list, file)
    print(f"Feature names saved to: {save_path}")

    return X, y

def process_spotify_csv():
    """
    Returns feature and label data for the Spotify Kaggle dataset, as a `pandas`
    `DataFrame` and `Series`, respectively.
    """
    # read csv
    save_path = save_kaggle_data("Spotify")
    data = pd.read_csv(save_path,
                       encoding='utf-8', encoding_errors='ignore')
    data.drop(['Track', 'Album Name', 'Artist', 'Release Date', 'ISRC', 'TIDAL Popularity'], axis=1, inplace=True)  # dropped
    # Loop through all columns in the DataFrame
    for col in data.columns:
        if data[col].dtype == 'object':
            # Remove commas and convert to numeric (if applicable)
            data[col] = data[col].str.replace(',', '').astype(float, errors='ignore')
    data = data.dropna()
    data.reset_index(inplace=True)
    y = pd.Series(data['Spotify Streams'].to_numpy())
    # Drop the target column from the features
    X = data.drop(columns=['Spotify Streams', "index"])

    columns_list = X.columns.tolist()
    save_path = "small_scale/data/Spotify_feature_names.pkl"
    with open(save_path, 'wb') as file:
        pickle.dump(columns_list, file)
    print(f"Feature names saved to: {save_path}")

    return X, y


def get_uciid_map():
    return {"Bank": 222, "Wine": 186, "Glioma": 759}

def get_kaggle_map():
    return {"Diabetes": "uciml/pima-indians-diabetes-database", "Spotify": "nelgiriyewithana/most-streamed-spotify-songs-2024"}

def save_kaggle_data(data_name, save_dir="small_scale/data/"):
    """
    Saves data for a Kaggle dataset `data_name` as a CSV file and returns the
    corresponding filename.
    """
    # Get the kaggle map
    kaggle_map = get_kaggle_map()

    # Validate data_name
    if data_name not in kaggle_map:
        raise TypeError(f"Invalid data name: '{data_name}'. Expected one of {list(kaggle_map.keys())}.")

    save_path = os.path.join(save_dir, f"{data_name}.csv")
    if os.path.exists(save_path):
        return save_path

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Get the dataset identifier and download it
    download_dir = kaggle_map[data_name]
    path = kagglehub.dataset_download(download_dir)

    # Load the dataset (assuming the downloaded path contains a CSV file)
    csv_file = None
    for file in os.listdir(path):  # Check files in the downloaded directory
        if file.endswith(".csv"):
            csv_file = os.path.join(path, file)
            break

    if csv_file is None:
        raise FileNotFoundError(f"No CSV file found in the downloaded dataset at {path}.")

    # Read the CSV file with the specified encoding
    try:
        df = pd.read_csv(csv_file, encoding="ISO-8859-1")  # Use ISO-8859-1 for non-UTF-8 files
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            f"Failed to read the file '{csv_file}' with encoding 'ISO-8859-1'. Check the file's encoding or content."
        ) from e

    df.to_csv(save_path, index=False)  # Save it to the specified save_dir without an index

    print(f"Dataset saved to: {save_path}")
    return save_path

def load_kaggle_data(data_name, save_dir="small_scale/data/"):
    """
    Load Kaggle data, preprocess it, and save feature names to a specified directory.

    Parameters:
        data_name (str): The name of the dataset to load.
        save_dir (str): The directory where the feature names .pkl file will be saved.

    Returns:
        tuple: A NumPy array of features (X), a list of labels (y), a list of feature names,
               and the path to the saved .pkl file with feature names.
    """
    if data_name == "Spotify":
        kag_data, y = process_spotify_csv()
    else:
        save_path = save_kaggle_data(data_name)
        kag_data = pd.read_csv(save_path, encoding='utf-8', encoding_errors='ignore')
        kag_data = kag_data.dropna()
        y = kag_data['Outcome']
        kag_data = kag_data.drop(columns=['Outcome'])

    # Get feature names
    feat_names = list(kag_data.columns)

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the file path for the pickle file
    pkl_file_path = os.path.join(save_dir, f"{data_name}_feature_names.pkl")

    # Save feature names to the specified directory
    with open(pkl_file_path, 'wb') as pkl_file:
        pickle.dump(feat_names, pkl_file)

    # Convert data to NumPy arrays
    X = kag_data.to_numpy()
    y = y.squeeze().tolist()

    print(f"Feature names saved to: {pkl_file_path}")
    return X, y, feat_names, pkl_file_path


def get_uci_feature_names(matrix, data_name, save_dir="small_scale/data/"):
    """
    Extract and process feature names along with their descriptions from a metadata matrix from fetch_ucirepo with .variables command.

    Parameters:
        matrix (pd.DataFrame): A DataFrame containing feature metadata, including
                               columns 'name', 'role', and 'description'.
        data_name (str): The name of the dataset for naming the output file.
        save_dir (str): The directory where the .json file will be saved.

    Returns:
        tuple: A dictionary of processed feature names with descriptions and the path to the saved .json file.
    """
    # Filter rows where 'role' is 'Feature'
    features = matrix[matrix['role'] == 'Feature']

    # Create a dictionary of feature names and descriptions
    feature_dict = {}
    for _, row in features.iterrows():
        # Process the feature name
        feature_name = row['name'].replace('_', ' ')

        # Add description if it exists and is not None
        if row['description'] and row['description'] != "None":
            feature_dict[feature_name] = row['description']
        else:
            feature_dict[feature_name] = f"{feature_name}"

    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the file path
    save_path = os.path.join(save_dir, f"{data_name}_feature_names.json")

    # Save the feature dictionary to a .json file
    with open(save_path, 'w') as file:
        json.dump(feature_dict, file, indent=4)

    print(f"Processed feature names and descriptions saved to: {save_path}")
    return feature_dict, save_path



def load_uci_data(data_name):
    """
    Loads an UCI dataset, returning:
    - `X`: a numpy array of the feature data.
    - `y`: a list of the labels.
    - `feat_names`: feature names corresponding to `X`.
    - `feat_dir`: filename where the feature names were stored.
    """
    id_map = get_uciid_map()
    if data_name not in id_map:
        raise TypeError(f"Invalid data name: '{data_name}'. Expected one of {list(id_map.keys())}.")
    id_num = id_map[data_name]
    uci_data = fetch_ucirepo(id=id_num)
    X = uci_data.data.features.to_numpy()
    y = uci_data.data.targets.squeeze().tolist()
    feat_names, feat_dir = get_uci_feature_names(uci_data.variables, data_name)
    return X, y, feat_names, feat_dir

