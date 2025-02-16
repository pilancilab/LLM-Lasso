import numpy as np
import pandas as pd
from LLasso.helper import balanced_folds
import os
import json
import rpy2.robjects as ro
from sklearn.preprocessing import StandardScaler
import pickle


def save_train_test_splits(X: pd.DataFrame, y: pd.Series, save_dir: str, n_splits=10, seed=0):
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


def read_train_test_splits(dir: str, n_splits):
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

def read_baseline_splits(dir: str, key="160", n_splits=10):
    feature_baseline = {}
    with open(f'{dir}/split0/selected_features.json') as f:
        data = json.load(f)
    for x in data.keys():
        feature_baseline[x] = [data[x][key]]

    for i in range(n_splits):
        with open(f'{dir}/split{i}/selected_features.json') as f:
            data = json.load(f)
        for x in data.keys():
            feature_baseline[x].append(data[x][key])

    return feature_baseline


def read_baseline_splits_old(dir: str, task: str, key="160"):
    feature_baseline = {}
    with open(f'{dir}/split1/{task}_split1/selected_features.json') as f:
        data = json.load(f)
    for x in data.keys():
        feature_baseline[x] = [data[x][key]]

    for i in range(2, 11):
        with open(f'{dir}/split{i}/{task}_split{i}/selected_features.json') as f:
            data = json.load(f)
        for x in data.keys():
            feature_baseline[x].append(data[x][key])

    return feature_baseline


def read_csv_splits(dir: str, n_splits: int, names):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in range(n_splits):
        x_train.append(pd.read_csv(f"{dir}/x_train{i+1}", names=names))
        x_test.append(pd.read_csv(f"{dir}/x_test{i+1}.csv", names=names))
        y_train.append(pd.read_csv(f"{dir}/y_train{i+1}", header=None)[0])
        y_test.append(pd.read_csv(f"{dir}/y_test{i+1}.csv", header=None)[0])
    return x_train, x_test, y_train, y_test


