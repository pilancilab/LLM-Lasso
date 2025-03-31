from enum import IntEnum
import pandas as pd
import numpy as np


class PenaltyType(IntEnum):
    IMP = 0
    PF = 1


def scale_cols(x: pd.DataFrame, center=None, scale=None) -> pd.DataFrame:
    if center is None:
        center = np.mean(x, axis=0)
    if scale is None:
        scale = np.std(x, axis=0)
    
    return (x - center) / scale


def count_feature_usage(betas: np.array, multinomial, n_features, tolerance=1e-10):
    """
    Counts the number of times each feature is used in a multinomial glmnet model across all lambdas.

    Parameters:
    - `multinomial`: whether classification is multinomial.
    - `n_features`: number of features
    - `tolerance`: numerical tolerance for counting nonzero features.

    Returns:
    - A DataFrame where each row corresponds to a lambda value and each column indicates
      whether a feature is nonzero (True/False).
    """
    # Get the number of features (exclude intercept)
    if multinomial:
        feature_inclusion_matrix =  np.abs(betas.reshape((betas.shape[0], -1, n_features))).mean(axis=1) > tolerance
        sign_mtx = np.argmax(betas.reshape((betas.shape[0], -1, n_features)), axis=1)
        magnitude_mtx = np.max(betas.reshape((betas.shape[0], -1, n_features)), axis=1)
    else:
        feature_inclusion_matrix = np.abs(betas) > tolerance
        sign_mtx = np.sign(betas)
        magnitude_mtx = np.abs(betas)

    # Convert to a DataFrame for easier interpretation
    feature_inclusion_df = pd.DataFrame(feature_inclusion_matrix, columns=[f"Feature_{j+1}" for j in range(n_features)])
    sign_df = pd.DataFrame(sign_mtx, columns=[f"Feature_Sign_{j+1}" for j in range(n_features)])
    magnitude_df = pd.DataFrame(magnitude_mtx, columns=[f"Feature_Magnitude{j+1}" for j in range(n_features)])
    return feature_inclusion_df, sign_df, magnitude_df