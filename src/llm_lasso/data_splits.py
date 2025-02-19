import numpy as np
import pandas as pd
import os
import json


# Helper function to create balanced folds
def balanced_folds(data_labels, nfolds=None):
    """
    Creates equal-size folds such that different folds an equal proportion
    of each class.
    """
    totals = np.bincount(data_labels)
    fmax = np.max(totals)
    if nfolds is None:
        nfolds = min(min(np.bincount(data_labels)), 10)
    nfolds = max(min(nfolds, fmax), 2)

    # Group indices by class
    y_groups = {label: np.where(data_labels == label)[0] for label in np.unique(data_labels)}

    # Shuffle indices within each class
    for label in y_groups:
        np.random.shuffle(y_groups[label])

    # Distribute indices into folds
    folds = [[] for _ in range(nfolds)]
    for label, indices in y_groups.items():
        for i, idx in enumerate(indices):
            folds[i % nfolds].append(idx)

    return [np.array(fold) for fold in folds]


def save_train_test_splits(X: pd.DataFrame, y: pd.Series, save_dir: str, n_splits=10, seed=0):
    """
    Divides a dataset into 50/50 train-test folds, for `n_splits` differet
    random seeds. Saves all splits to CSV files

    Parameter:
    - `X`: datapoints, as a `pandas.DataFrame`.
    - `y`: labels, as a `pandas.Series`.
    - `save_dir`: directory to save the CSV splits
    - `n_splits`: number of times to run the split-generation process (with
        different random seeds)
    - `seed`: initial random seed
    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(n_splits):
        np.random.seed(seed + i)

        train_idxs, test_idxs = balanced_folds(y, nfolds=2)
        x_train = X.loc[train_idxs]
        x_test = X.loc[test_idxs]

        y_train = y.loc[train_idxs]
        y_test = y.loc[test_idxs]

        
        x_train.to_csv(f"{save_dir}/x_train_{i}.csv", index=False)
        x_test.to_csv(f"{save_dir}/x_test{i}.csv", index=False)
        y_train.to_csv(f"{save_dir}/y_train{i}.csv", index=False)
        y_test.to_csv(f"{save_dir}/y_test{i}.csv", index=False)


def read_train_test_splits(dir: str, n_splits: int) -> tuple[
    list[pd.DataFrame], list[pd.Series], list[pd.DataFrame], list[pd.Series]
]:
    """
    Reads the output of `save_train_test_splits` into the following four lists:
    - `x_train`: list of training datapoints (as pandas dataframes) for each different split
    - `x_test`: list of test datapoints (as pandas dataframes) for each different split
    - `y_train`: list of training labels (as pandas series) for each different split
    - `y_test`: list of test labels (as pandas series) for each different split

    Parameters:
    - `dir`: save directory of `save_train_test_splits`
    - `n_splits`: number of splits to load.
    """
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in range(n_splits):
        x_train.append(pd.read_csv(f"{dir}/x_train_{i}.csv"))
        x_test.append(pd.read_csv(f"{dir}/x_test{i}.csv"))
        y_train.append(pd.read_csv(f"{dir}/y_train{i}.csv")["0"])
        y_test.append(pd.read_csv(f"{dir}/y_test{i}.csv")["0"])

    return x_train, x_test, y_train, y_test


def read_baseline_splits(dir: str, n_features: str, n_splits: int) -> dict[str, list[list[str]]]:
    """
    Reads the features selected by the data-driven baselines from the output of
    `llm_lasso.baselines.data_driven.run_all_baselines_for_splits`.

    Parameters:
    - `dir`: save directory of `run_all_baselines_for_splits`.
    - `n_features`: number of features selected by each baseline (this corresponds
        to the key of the baseline result CSV we read) .
    - `n_splits`: number of splits to load.

    Returns: dictionary mapping baseline name to a list of selected feature
        names for each split.
    """
    n_features = str(n_features)
    feature_baseline = {}
    with open(f'{dir}/split0/selected_features.json') as f:
        data = json.load(f)
    for x in data.keys():
        feature_baseline[x] = [data[x][n_features]]

    for i in range(n_splits):
        with open(f'{dir}/split{i}/selected_features.json') as f:
            data = json.load(f)
        for x in data.keys():
            feature_baseline[x].append(data[x][n_features])

    return feature_baseline