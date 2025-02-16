import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from adelie import grpnet
import adelie as ad
from adelie.diagnostic import predict, coefficient
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse
import re

# Helper function to create balanced folds
def balanced_folds(y, nfolds=None):
    totals = np.bincount(y)
    fmax = np.max(totals)
    if nfolds is None:
        nfolds = min(min(np.bincount(y)), 10)
    nfolds = max(min(nfolds, fmax), 2)

    # Group indices by class
    y_groups = {label: np.where(y == label)[0] for label in np.unique(y)}

    # Shuffle indices within each class
    for label in y_groups:
        np.random.shuffle(y_groups[label])

    # Distribute indices into folds
    folds = [[] for _ in range(nfolds)]
    for label, indices in y_groups.items():
        for i, idx in enumerate(indices):
            folds[i % nfolds].append(idx)

    return [np.array(fold) for fold in folds]

# Function to plot error bars
def error_bars(x, upper, lower, color, width=0.02, x_offset=0):
    bar_width = width * (np.max(x) - np.min(x))
    x += x_offset * (np.max(x) - np.min(x))
    plt.vlines(x, lower, upper, colors=color)
    plt.hlines(upper, x - bar_width, x + bar_width, colors=color)
    plt.hlines(lower, x - bar_width, x + bar_width, colors=color)

def assign_color_group(row):
    return assign_color_group_model(row['methodModel'])
    
def assign_color_group_model(method_model):
    if re.match(r"^1/imp", method_model):
        return "red4"
    elif re.match(r"^ReLU", method_model):
        return "blue4"
    elif method_model == "LLMselect":
        return "purple4"
    elif method_model == "Lasso":
        return "darkgray"
    else:
        return "black"

def scale_cols(x, center=None, scale=None):
    if center is None:
        center = np.mean(x, axis=0)
    if scale is None:
        scale = np.std(x, axis=0)
    
    return (x - center) / scale

# Permute rows of a matrix
def permute_rows(matrix):
    n = matrix.shape[0]
    shuffled_indices = np.random.permutation(n)
    return matrix[shuffled_indices, :]

# Function to calculate variance for a matrix
def varr(matrix, meanx=None):
    if meanx is None:
        meanx = np.mean(matrix, axis=1, keepdims=True)
    x_diff = matrix - meanx
    return np.mean(x_diff**2, axis=1)

# Function to compute multiclass contrasts
def multiclass_func(x, y, s0=0):
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    class_means = np.zeros((x.shape[0], n_classes))
    variances = np.zeros_like(class_means)

    for i, cls in enumerate(unique_classes):
        class_data = x[:, y == cls]
        class_means[:, i] = np.mean(class_data, axis=1)
        variances[:, i] = np.var(class_data, axis=1, ddof=1)

    overall_mean = np.mean(x, axis=1)
    mean_diffs = class_means - overall_mean[:, np.newaxis]

    # Calculate scores and standard deviations
    scores = np.sqrt(np.sum(mean_diffs**2, axis=1))
    standard_devs = np.sqrt(np.sum(variances, axis=1))

    t_values = scores / (standard_devs + s0)
    return {
        "tt": t_values,
        "numer": scores,
        "sd": standard_devs,
        "stand_contrasts": mean_diffs / standard_devs[:, np.newaxis]
    }


# Function to generate penalty factors based on scores
def pffun_relu(imp_scores, scorcut, pfmax, impmin, seed=None):
    if seed is not None:
        np.random.seed(seed)

    p = len(imp_scores)
    sorted_indices = np.argsort(imp_scores)
    sorted_scores = np.sort(imp_scores)

    penalty_factors = np.zeros(p)

    penalty_factors[sorted_scores >= scorcut] = 1
    slope = 1 / (1 - scorcut)
    below_cut = sorted_scores < scorcut
    penalty_factors[below_cut] = slope * (1 - sorted_scores[below_cut])

    if impmin < scorcut:
        factor = (pfmax - 1) * (1 - scorcut) / (scorcut - impmin)
    else:
        factor = 1

    penalty_factors = factor * (penalty_factors - 1) + 1
    return penalty_factors[np.argsort(sorted_indices)]