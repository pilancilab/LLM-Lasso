"""
Implements data-driven feature selector. K is the number of features to be selected.

Description: this is a feature Selection Script with Command-Line Interface.

Supports four feature selection methods:
- Mutual Information (MI)
- Recursive Feature Elimination (RFE)
- Minimum Redundancy Maximum Relevance (MRMR)
- Random Feature Selection

"""

import os
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LogisticRegression
import random
from Expert_RAG.data_processing import load_feature_names
import pickle
import json
from LLasso.splits import read_train_test_splits

## Helpers ##
def load_x_from_txt(file_path, gene_names):
    """
    Load feature matrix (X) from a .txt file and assign gene names as column names.

    Each line in the file represents a p-dimensional vector (floats), with n lines total.

    Parameters:
    - file_path (str): Path to the .txt file containing the feature matrix.
    - gene_names (list of str): List of gene names corresponding to each feature (column).

    Returns:
    - pd.DataFrame: n x p feature matrix (n samples, p features) with gene names as columns.
    """

    # Load data where each row is a p-dimensional vector
    try:
        data = pd.read_csv(file_path, header=None, delimiter=",", dtype=float)
    except Exception as e:
        raise ValueError(f"Error reading feature file {file_path}: {e}")

    # Check if data is empty or improperly formatted
    if data.empty:
        raise ValueError(f"The feature file {file_path} is empty or improperly formatted.")

    # Check if the number of gene names matches the number of columns
    if len(gene_names) != data.shape[1]:
        raise ValueError("The number of gene names does not match the number of features (columns) in the file.")

    # Assign gene names as column names
    data.columns = gene_names

    return data


def load_y_from_txt(file_path):
    """
    Load target variable (y) from a .txt file.

    Each line in the file contains a single integer, with n lines total.

    Parameters:
    - file_path (str): Path to the .txt file containing the target variable.

    Returns:
    - pd.Series: n-dimensional target vector.
    """
    # Load data as a single-column DataFrame and ensure it’s an integer
    try:
        data = pd.read_csv(file_path, header=None, delimiter=",", dtype=int).squeeze("columns")
    except Exception as e:
        raise ValueError(f"Error reading target file {file_path}: {e}")

    # Check if data is empty or improperly formatted
    if data.empty:
        raise ValueError(f"The target file {file_path} is empty or improperly formatted.")

    # Check if data is single-dimensional
    if len(data.shape) != 1:
        raise ValueError(f"The target file {file_path} must contain a single integer per line.")

    return data


# 1. Filtering by Mutual Information (MI)
def mi_filter_method(X, y, k):
    mi = mutual_info_regression(X, y, random_state=42, discrete_features=False)
    mi_scores = pd.Series(mi, index=X.columns)
    selected_features = mi_scores.sort_values(ascending=False).head(k).index.tolist()
    return X[selected_features], selected_features


# 2. Recursive Feature Elimination (RFE)
def rfe_method(X, y, n_features_to_select):
    model = LogisticRegression()
    selector = RFE(estimator=model, n_features_to_select=n_features_to_select, step=10)
    selector.fit(X, y)
    selected_features = X.columns[selector.support_].tolist()
    return X[selected_features], selected_features


def mrmr_method(X:pd.DataFrame, y, k):
    mi = mutual_info_regression(X, y, random_state=42, discrete_features=False)
    mi_scores = pd.Series(mi, index=X.columns)
    selected_features = []
    remaining_features = list(X.columns)

    # mutual information between Xi and Xj
    # matrix where each row is a selected feature and each column is a feature
    cross_mi = pd.DataFrame(columns=X.columns)

    for i in range(k):
        relevance = mi_scores.loc[remaining_features]

        redundancy = pd.Series(
            [cross_mi[feat].mean()
                if selected_features else 0 for feat in remaining_features],
            index=remaining_features
        )
        mrmr_scores = relevance - redundancy
        next_feature = mrmr_scores.idxmax()

        cross_mi.loc[-1] = mutual_info_regression(X, X[next_feature], discrete_features=False, random_state=42)

        selected_features.append(next_feature)
        remaining_features.remove(next_feature)

    return X[selected_features], selected_features



# 4. Random Feature Selector
def random_feature_selector(X, k, random_state=42):
    random.seed(random_state)
    selected_features = random.sample(list(X.columns), k)
    return X[selected_features], selected_features


def feature_selector(X, y, method, k, random_state=42):
    """
    General feature selection function to call specific feature selection methods.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (array-like): Target variable.
    - method (str): Feature selection method ('mi', 'rfe', 'mrmr', 'random').
    - k (int): Number of features to select.
    - random_state (int): Random seed for reproducibility (used in 'random' method).

    Returns:
    - X_selected (pd.DataFrame): Selected features (subset of X).
    - selected_features (list): List of selected feature names.
    """
    if method == 'mi':
        return mi_filter_method(X, y, k)
    elif method == 'rfe':
        return rfe_method(X, y, n_features_to_select=k)
    elif method == 'mrmr':
        return mrmr_method(X, y, k)
    elif method == 'random':
        return random_feature_selector(X, k, random_state=random_state)
    else:
        raise ValueError("Invalid method. Choose from 'mi', 'rfe', 'mrmr', 'random'.")



def run_all_baselines(X, y, save_dir, min=0, max=161, step=160, random_state=42):
    """
    Runs all baselines for feature selection across a range of k values and saves results in a JSON file.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (array-like): Target variable.
    - feature_names (list): List of feature names.
    - save_dir (str): Directory to save results.
    - min (int): Minimum k value for feature selection.
    - max (int): Maximum k value for feature selection.
    - step (int): Step size for k values.
    - random_state (int): Random seed for reproducibility.
    """
    # Baseline methods
    # methods = ['mi', 'rfe', 'random', 'mrmr']
    methods = ["random", "mrmr"]

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Dictionary to store results
    results = {}

    # Calculate the total number of iterations for the progress bar
    total_iterations = len(range(min, max, step)) * len(methods)

    # Use tqdm progress bar
    with tqdm(total=total_iterations, desc="Running Baselines") as pbar:
        for k in range(min, max, step):
            if k == 0:
                continue  # Skip k=0, as selecting zero features isn't meaningful

            for method in methods:
                # Perform feature selection
                X_selected, selected_features = feature_selector(X, y, method=method, k=k, random_state=random_state)

                # Store results for this method and k value
                if method not in results:
                    results[method] = {}
                results[method][k] = selected_features  # Use selected features directly (already names)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(method=method, k=k)

    # Save the results to a JSON file
    results_path = os.path.join(save_dir, "selected_features.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")


def run_all_baselines_for_splits(
    x_train: list[pd.DataFrame], y_train: list[pd.Series],
    save_dir="baselines/results", feature_files=None,
    min=0, max=161, step=160, random_state=42
):
    """
    Runs all baselines for feature selection on 10 splits for each dataset using `run_all_baselines`.

    Parameters:
    - save_dir (str): Directory to save results.
    - feature_files (list): List of file paths containing feature names for each dataset.
    - min (int): Minimum k value for feature selection.
    - max (int): Maximum k value for feature selection.
    - step (int): Step size for k values.
    - random_state (int): Random seed for reproducibility.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Datasets in the data directory
    for split in tqdm(range(len(x_train))):
        split_save_dir = os.path.join(save_dir, f"split{split}")
        os.makedirs(split_save_dir, exist_ok=True)
        run_all_baselines(
            x_train[split], y_train[split], save_dir=split_save_dir,
            min=min, max=max, step=step, random_state=random_state
        )

    print(f"All results saved in {save_dir}")


if __name__ == "__main__":
    #main()

    # Create argument parser
    parser = argparse.ArgumentParser(description="Run feature selection baselines across a range of k values.")

    # Input arguments
    parser.add_argument("--split_dir", type=str, required=True, help="Path to the all of the train_test_splits")
    parser.add_argument("--n_splits", type=int, required=True, help="Number of splits in the dir")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results.")
    # Optional arguments
    parser.add_argument("--min", type=int, default=0, help="Minimum k value for feature selection.")
    parser.add_argument("--max", type=int, default=161, help="Maximum k value for feature selection.")
    parser.add_argument("--step", type=int, default=160, help="Step size for k values.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")

    # Parse the arguments
    args = parser.parse_args()

    (x_train, _, y_train, _) = read_train_test_splits(args.split_dir, args.n_splits)

    # Run the baseline function
    run_all_baselines_for_splits(
        x_train, y_train, args.save_dir, min=args.min, max=args.max, step=args.step, random_state=args.random_state)

    # Example Command
    """
    PYTHONPATH=$(pwd) python baselines/data-driven.py \
    --split_dir data/train_test_splits/FL \
    --n_splits 10 \
    --save_dir baselines/results/FL \
    --random_state 42
    """









