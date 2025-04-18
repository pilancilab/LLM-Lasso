from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from adelie.cv import cv_grpnet
from adelie.solver import grpnet
import adelie as ad
from adelie.diagnostic import auc_roc, test_error_hamming, test_error_mse, predict
from enum import IntEnum
import xgboost as xgb
from dataclasses import dataclass
from llm_lasso.task_specific_lasso.utils import standardize, PenaltyType, count_feature_usage
from scipy.special import expit
from tqdm import tqdm
import scipy.linalg

"""
TODO: clean up and document this file!
"""

@dataclass
class TrainTestAugmented:
    X_train: np.array
    X_val: np.array
    y_train: np.array
    y_val: np.array
    X_train_aug: np.array # with a col of ones added
    X_val_aug: np.array # with a col of ones added
    glm_train: ad.glm.glm_base
    n: int
    nv: int


def get_balanced_splits(data_labels: pd.Series, ratio_test=0.5) -> tuple[list[int], list[int]]:
    """
    Creates equal-size folds such that different folds an equal proportion
    of each class.
    """
    # Group indices by class
    y_groups = {label: np.where(data_labels == label)[0] for label in np.unique(data_labels)}

    # Shuffle indices within each class
    for label in y_groups:
        np.random.shuffle(y_groups[label])

    # Distribute indices into folds
    train = []
    test = []
    for label, indices in y_groups.items():
        for i, idx in enumerate(indices):
            if i <= np.ceil(len(indices) * ratio_test):
                test.append(idx)
            else:
                train.append(idx)

    np.random.shuffle(train)
    np.random.shuffle(test)
    return (train, test)


def get_train_and_test_splits(
    X: pd.DataFrame, y: pd.Series,
    ratio_test: float, n_splits: int, seed=0
) -> list[TrainTestAugmented]:
    splits = []
    for i in range(n_splits):
        np.random.seed(seed + i)
        (train, test) = get_balanced_splits(y, ratio_test)

        X_train = X.loc[train]
        X_test = X.loc[test]
        y_train = y.loc[train].to_numpy()
        y_test = y.loc[test].to_numpy()

        X_test = np.asfortranarray(
            standardize(X_test, center=X_train.mean(axis=0), scale=X_train.std(axis=0)).to_numpy(),
        )
        X_train = np.asfortranarray(
            standardize(X_train).to_numpy(),
        )
        X_train_aug = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        X_val_aug = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

        glm_train = ad.glm.binomial(y=y_train, dtype=np.float64)

        splits.append(
            TrainTestAugmented(
                X_train, X_test, y_train, y_test,
                X_train_aug, X_val_aug, glm_train,
                X_train.shape[0], X_test.shape[0]
            )
        )
    return splits


def penalty_descent_llm_lasso(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    score: np.array,
    lambda_min_ratio = 0.01,
    score_type: int = PenaltyType.PF,
    n_splits = 5,
    seed=0,
    tolerance=1e-10,
    n_threads=4,
    val_proportion = 0.5,
    default_step_size=0.1,
    use_scaled_polyak=True,
    polyak_scale=0.5,
    max_iter=10,
):
    if score_type == PenaltyType.IMP:
        penalty = 1 / score
    elif score_type == PenaltyType.PF:
        penalty = score
    else:
        assert False, "score_type must be either PenaltyType.IMP or PenaltyType.PF"

    assert len(y_train.unique()) == 2

    splits = get_train_and_test_splits(
        x_train, y_train, val_proportion, n_splits, seed
    )
    penalty = penalty / np.sum(penalty) * x_train.shape[1]

    ## Initial run to get lambda
    models = [
        grpnet(
            X=split.X_train,
            glm=split.glm_train,
            ddev_tol=0,
            early_exit=True,
            n_threads=n_threads,
            min_ratio=lambda_min_ratio,
            progress_bar=False,
            alpha=1,
            penalty=penalty,
        ) for split in splits
    ]

    lmdas = [model.lmda_path[-1] for model in models]
    est_opt_mse = 0

    best_val_mse = float('inf')
    best_penalties = None

    for i in tqdm(range(max_iter)):
        # Normalize penalty
        penalty = penalty / np.sum(penalty) * x_train.shape[1]
        
        # Train a GLM
        try:
            models = [
                grpnet(
                    X=split.X_train,
                    glm=split.glm_train,
                    ddev_tol=0,
                    early_exit=False,
                    n_threads=n_threads,
                    progress_bar=False,
                    alpha=1,
                    penalty=penalty,
                    lmda_path=[lmda]
                ) for (split, lmda) in zip(splits, lmdas)
            ]
        except:
            print("Solver error! Giving up")
            break

        avg_val_error = 0
        avg_auroc = 0
        avg_n_features = 0
        val_mse = 0

        if any([model.betas.shape[0] == 0 for model in models]):
            print("Solver error! Giving up")
            break

        ## Run eval
        etas_list = []
        etas_tr_list = []
        for (split, model) in zip(splits, models):
            betas = model.betas[0, :].toarray()

            etas_list.append(predict(
                X=split.X_val,
                betas=model.betas[0, :],
                intercepts=model.intercepts[0],
                n_threads=8,
            ))

            etas_tr_list.append(predict(
                X=split.X_train,
                betas=model.betas[0, :],
                intercepts=model.intercepts[0],
                n_threads=8,
            ))

            avg_val_error += test_error_hamming(etas_list[-1], split.y_val, False)[0] / n_splits
            avg_auroc += auc_roc(etas_list[-1], split.y_val, False)[0] / n_splits
            active_set = np.where(np.abs(betas) > 1e-10)[0]
            avg_n_features += len(active_set) / n_splits

            val_mse += (np.sum(np.square(split.y_val - expit(etas_list[-1]))) / (split.nv * 2)) / n_splits
        # print(val_mse, avg_val_error, avg_n_features)
        if val_mse <= best_val_mse:
            best_val_mse = val_mse
            best_penalties = np.array(list(penalty))

        ### Compute Gradient Step for Penalties
        grad = np.zeros(x_train.shape[1])
        for (split, model, etas, etas_tr, lmda) in zip(splits, models, etas_list, etas_tr_list, lmdas):
            betas = model.betas[0, :].toarray()
            intercept = model.intercepts[0]

            betas_aug = np.hstack(([intercept], betas[0]))
            active_set = np.where(np.abs(betas_aug) > 1e-10)[0]
        
            s = np.sign(betas_aug)
            sA = s[active_set]
            XA = split.X_train_aug[:, active_set]
            XVA = split.X_val_aug[:, active_set]

            exp_eta_tr = np.exp(etas_tr[0])
            exp_neg_eta = np.exp(-etas[0])

            diag_one = exp_eta_tr / (1 + exp_eta_tr)**2
            diag_two = 1 / (1 + exp_neg_eta)**2
            RHS = split.y_val - 1/(1 + exp_neg_eta)

            model_grad = split.n * lmda / split.nv * np.diag(sA) @ scipy.linalg.solve(
                XA.T @ np.diag(diag_one) @ XA, XVA.T @ np.diag(diag_two) @ RHS,
                assume_a="sym"
            )

            if 0 in active_set:
                active_set = active_set[1:]
                model_grad = model_grad[1:]
            active_set -= 1
            
            grad[active_set] += model_grad / n_splits

        if use_scaled_polyak and est_opt_mse < val_mse:
            step_size = (val_mse - est_opt_mse) / np.sum(np.square(grad)) * polyak_scale
        else:
            step_size = default_step_size

        penalty -= grad * step_size
        penalty = np.maximum(penalty, 0)

    penalty = best_penalties

    x_train_scaled = standardize(x_train)
    x_test_scaled = standardize(x_test, center=x_train.mean(axis=0), scale=x_train.std(axis=0))
    glm_train = ad.glm.binomial(y=y_train.to_numpy(), dtype=np.float64)

    model = grpnet(
        X=x_train_scaled.to_numpy(),
        glm = glm_train,
        alpha=1,
        penalty=penalty,
        n_threads=n_threads,
        progress_bar=False,
        early_exit=True,
        min_ratio=lambda_min_ratio / 10
    )

    y = y_test.to_numpy()
    etas = predict(
        X=x_test_scaled.to_numpy(),
        betas=model.betas,
        intercepts=model.intercepts,
        n_threads=n_threads,
    )
        
    test_error_raw = test_error_hamming(etas, y, False)
    roc_auc_raw = auc_roc(etas, y, False)
    non_zero_coefs_raw = [
        np.count_nonzero(np.abs(coeffs) > tolerance)
            for coeffs in model.betas.toarray()
    ]

    (feature_count_raw, signs_raw, magnitudes_raw) = count_feature_usage(model.betas.toarray(),  False, x_train.shape[1], tolerance=tolerance)

    df = pd.DataFrame({
        'n_features': non_zero_coefs_raw,
        'test_error': test_error_raw,
        "auroc": roc_auc_raw
    })
    df = pd.concat([df, feature_count_raw, signs_raw, magnitudes_raw], axis=1)
    df["best_method_model"] = "penalty_descent"
    df["method"] = "penalty_descent"

        # Group by 'non_zero_coefs' and filter rows where 'metric' is the minimum for each group
    return (
        df.loc[df.groupby('n_features')['test_error'].idxmin()]
        .reset_index(drop=True)
    )