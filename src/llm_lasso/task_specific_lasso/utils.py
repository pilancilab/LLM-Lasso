from enum import IntEnum
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
import adelie as ad
from adelie.solver import grpnet
from adelie.diagnostic import auc_roc, test_error_hamming, test_error_mse, predict
from scipy.interpolate import interp1d
import xgboost as xgb
import regex as re
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold


@dataclass
class TrainTest:
    """
    Stores training and test splits for a classification or regression
    problem.
    """
    x_train: pd.DataFrame
    x_test:pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    
    def to_tuple(self):
        return self.x_train, self.x_test, self.y_train, self.y_test


class PenaltyType(IntEnum):
    """
    Specifies whether scores input into LLM-Lasso are importance scores (IMP)
    or penalty factors (PF).
    """
    IMP = 0
    PF = 1


def standardize_array(x: pd.DataFrame, center=None, scale=None) -> pd.DataFrame:
    """
    Perform standardization on a dataframe, with the option to manually pass
    in the center and scale (e.g., for standardizing a test set using the 
    mean and stdev from the training set).
    """
    if center is None:
        center = np.mean(x, axis=0)
    if scale is None:
        scale = np.std(x, axis=0)
    
    return (x - center) / scale


def count_feature_usage(
    betas: np.array, multinomial, feature_names: list[str], tolerance=1e-10
):
    """
    Counts when each feature is used in a multinomial adelie grpnet model
    across all lambdas, as well as the magnitude and sign of the coefficients
    for the selected features.

    Parameters:
    - `multinomial`: whether classification is multinomial.
    - `feature_names`: list of features
    - `tolerance`: numerical tolerance for counting nonzero features.

    Returns:
    - A DataFrame where each row corresponds to a lambda value and columns
        indicating feature selection (as a boolean), feature weight magnitude,
        and feature weight sign (+1 or -1 for binary or index of the feature
        that is most heavily weighted for multinomial).

    """
    n_features = len(feature_names)
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
    feature_inclusion_df = pd.DataFrame(feature_inclusion_matrix, columns=[f"{feat}_Selected" for feat in feature_names])
    sign_df = pd.DataFrame(sign_mtx, columns=[f"{feat}_Sign" for feat in feature_names])
    magnitude_df = pd.DataFrame(magnitude_mtx, columns=[f"{feat}_Magnitude" for feat in feature_names])
    return feature_inclusion_df, sign_df, magnitude_df


def count_feature_usage_subset(
    coeffs: np.array, features: list[str], all_features: list[str]
):
    """
    Computes feature selection (which features are selected, and the magnitude
    and sign of the corresponding weights) for `coeffs`, where `coeffs` is
    `sklearn.LogisticRegressionCV.coef_` or `sklearn.LinearRegression.coef_`,
    for a model run on a subset of the features.

    Parameters:
    - coeffs: `sklearn.LogisticRegressionCV.coef_` or
        `sklearn.LinearRegression.coef_`
    - features: subset of the feature names used to train the logistic or
        linear regression model.
    - all_features: list of all feature names in the original datset.

    Returns:
    - One-row dataframe in the same format as `count_feature_usage`
    """
    selected = [feat in features for feat in all_features]
    pred_to_magnitude_sign = defaultdict(lambda: (0, 0))
    if len(coeffs.shape) == 1:
        coeffs = coeffs[np.newaxis, :]
    for (i, feat) in enumerate(features):
        if coeffs.shape[0] == 1:
            sign = int(np.sign(coeffs[0, i]))
        else:
            sign = np.argmax(coeffs[:, i])
        pred_to_magnitude_sign[feat] = (np.max(np.abs(coeffs[:, i])), sign)

    magnitude = [pred_to_magnitude_sign[feat][0] for feat in all_features]
    sign = [pred_to_magnitude_sign[feat][1] for feat in all_features]
    return selected, sign, magnitude


def run_l2_regression(
    train_test: TrainTest,
    features: list[str],
    model_name: str,
    regression=False,
    standardize=True,
    folds_cv = 10,
    seed = 0,
    n_threads = 4,
):
    """
    Runs l2-regularized regression (logistic for classification and linear for
    regression) for a subset of the features, with cross-validation to choose the
    regularization strength.

    Parameters:
    - train_test: training and test splits, as a TrainTest object
    - features: list of the features to select
    - model_name: name of the model used to select the features (used to 
        populate the returned DataFrame such that it matches the format of
        that returned by `llm_lasso_cv`)
    - regression: whether this is a regression problem (as opposed to 
        classification)
    - standardize: whether to standardize each feature
    - folds_cv: number of cross-validation folds to use when choosing the
        regularization strength
    - seed: random seed (for selecting cross-validation folds)
    - n_threads: number of threads to use for computation

    Returns: one-row dataframe with the `model_name`, test error, AUROC (for
        classification), number of features, cross-validation error, and the
        feature selection data from `count_feature_usage_subset`
    """
    (x_train, x_test, y_train, y_test) = train_test.to_tuple()
    
    np.random.seed(seed)
    if standardize:
        x_train_scaled = standardize_array(x_train)
        x_test_scaled = standardize_array(x_test, center=x_train.mean(axis=0), scale=x_train.std(axis=0))
    else:
        x_train_scaled = x_train
        x_test_scaled = x_test

    # Columns of the resulting dataframe
    df_cols = ["method", "test_error", "auroc", "n_features", "cvm"]
    df_cols += [f"{feat}_Selected" for feat in x_train.columns]
    df_cols += [f"{feat}_Sign" for feat in x_train.columns]
    df_cols += [f"{feat}_Magnitude" for feat in x_train.columns]

    # Select features
    x_subset_train = x_train_scaled[features]
    x_subset_test = x_test_scaled[features]

    if not regression: # classification
        model = LogisticRegressionCV(
            Cs=[0.1, 0.5, 1, 5, 10, 50, 100, 1000, 1e4],
            penalty="l2",
            solver="lbfgs",
            random_state=seed,
            n_jobs=n_threads,
            scoring="accuracy",
            refit=True,
            cv=StratifiedKFold(n_splits=folds_cv, shuffle=True, random_state=seed)
        ).fit(x_subset_train.to_numpy(), y_train.to_numpy())   

        preds = model.predict_proba(x_subset_test.to_numpy())
        
        test_error = 1 - accuracy_score(y_test.to_numpy(), np.argmax(preds, axis=1))
        auroc = roc_auc_score(y_test, preds[:, 1] if preds.shape[1] == 2 else preds, multi_class='ovr')
        cvm = 1 - model.scores_[next(iter(model.scores_.keys()))].mean(axis=0).max()
    else: # regression
        model = RidgeCV(
            alphas=[10, 2, 1, 0.2, 0.1, 0.02, 0.01, 0.001, 1e-4],
            cv=folds_cv,
        ).fit(x_subset_train.to_numpy(), y_train.to_numpy())
        preds = model.predict(x_subset_test.to_numpy())
        test_error = mean_squared_error(y_test.to_numpy(), preds)
        auroc = None # AUROC is undefined for regression
        cvm = model.best_score_

    # Get dataframe of selected features, magnitude, and sign
    selected, sign, magnitude = count_feature_usage_subset(
        model.coef_, features, x_test.columns
    )

    return pd.DataFrame([[
        model_name, test_error, auroc, len(features), cvm
    ] + selected + sign + magnitude], columns=df_cols)


def cve(cvm, non_zero, ref_cvm, ref_non_zero):
    """
    Calculate the area under the cross-validation error curve, defined as the signed area 
    under the reference curve.

    Parameters:
    - cvm: Cross-validation errors (list or array)
    - non_zero: Number of non-zero features (list or array)
    - ref_cvm: Cross-validation errors of reference (list or array)
    - ref_non_zero: Number of non-zero features of reference (list or array)

    Returns:
    - area: Signed area under the reference curve
    """
    # Create data frames and group by unique values of non_zero
    df1 = pd.DataFrame({'x1': ref_non_zero, 'y1': ref_cvm})
    df1 = df1.groupby('x1', as_index=False)['y1'].min()

    df2 = pd.DataFrame({'x2': non_zero, 'y2': cvm})
    df2 = df2.groupby('x2', as_index=False)['y2'].min()

    # Extract x and y values
    x1 = df1['x1'].values
    y1 = df1['y1'].values
    x2 = df2['x2'].values
    y2 = df2['y2'].values

    # Interpolate y1 values to match x2
    interp_func = interp1d(x1, y1, bounds_error=False, fill_value='extrapolate')
    y1_interp = interp_func(x2)

    # Calculate area using the trapezoidal rule
    area = 0
    for i in range(len(x2) - 1):
        width = x2[i+1] - x2[i]
        height = ((y1_interp[i] - y2[i]) + (y1_interp[i+1] - y2[i+1])) / 2
        area += width * height

    return area


def eval_grpnet_model(
    grpnet_model,
    x_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    multinomial: bool,
    regression: bool,
    n_threads: int = 8,
    tolerance=1e-10
):
    """
    Given a trained adelie grpnet object, compute test error and AUROC, as well
    as the number of non-zero coefficients, for each value of lambda (l1
    regularization strength)

    Parameters:
    - grpnet_model: trained grpnet object
    - x_test_scaled: standardized test data
    - y_test: test labels
    - multinomial: whether the problem is multinomial classification
    - regression: whether the problem is regression
    - n_threads: number of threads to use for prediction
    - tolerance: tolerance for determining whether a feature is selected

    Returns:
    - test_error_raw: array of test errors (misclassification or MSE, depending
        on whether the problem is classification or regression) for each lambda
    - auroc_raw: array of area under the ROC curve for classification, or None
        for regression
    - non_zero_coefs_raw: array of number of features selected for each lambda
    """
    if multinomial:
        one_hot_encoder = OneHotEncoder(sparse_output=False)  # Use dense array
        y_one_hot = one_hot_encoder.fit_transform(y_test.to_numpy().reshape(-1, 1))
        y = y_one_hot
    else:
        y = y_test.to_numpy()

    etas = predict(
        X=x_test_scaled.to_numpy(),
        betas=grpnet_model.betas,
        intercepts=grpnet_model.intercepts,
        n_threads=n_threads,
    )
    
    if regression:
        test_error_raw = test_error_mse(etas, y)
    else:
        test_error_raw = test_error_hamming(etas, y, multinomial)

    if not regression:
        auroc_raw = auc_roc(etas, y, multinomial)
    else:
        auroc_raw = None

    if multinomial:
        non_zero_coefs_raw = [
            np.count_nonzero(np.mean(np.abs(coeffs), axis=0) > tolerance)
            for coeffs in grpnet_model.betas.toarray().reshape((grpnet_model.betas.shape[0], -1, x_test_scaled.shape[1]))
        ]
    else:
        non_zero_coefs_raw = [
            np.count_nonzero(np.abs(coeffs) > tolerance)
                for coeffs in grpnet_model.betas.toarray()
        ]

    return test_error_raw, auroc_raw, non_zero_coefs_raw

def run_one_lasso(
    x_train_scaled: pd.DataFrame, y_train: pd.Series,
    x_test_scaled: pd.DataFrame, y_test: pd.Series,
    pf: np.array,
    lambda_min_ratio: float,
    lmda_path_size: int = 200,
    multinomial=False,
    regression=False,
    n_threads=4,
    tolerance=1e-10,
    alpha=1
):
    """
    Runs weighted Lasso and computes the test error, AUROC, and feature
    selection for each lambda.

    Parameters:
    - x_train_scaled: standardized training data
    - y_train: training labels
    - x_test_scaled: standardized test data
    - y_test: test labels
    - pf: penalty factors to use
    - lambda_min_ratio: the maximum lambda (l1 regularization strength) is 
        automatically chosen such that the model chooses zero features.
        `lambda_min_ratio` is multiplied by that value to get the minimum
        lambda used
    - lmda_path_size: number of different lambda values to try
    - multinomial: whether the problem is multinomial classification
    - regression: whether the problem is regression
    - n_threads: number of threads to use for computation
    - tolerance: tolerance for determining what features are selected
    - alpha: elasticnet parameter (0 means all l2 regularization, 1 means all
        l1 regularization)

    Returns: DataFrame with test error, AUROC, and feature selection
    """
    if multinomial:
        one_hot_encoder = OneHotEncoder(sparse_output=False)  # Use dense array
        y_one_hot = one_hot_encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))
        glm_train = ad.glm.multinomial(y=y_one_hot, dtype=np.float64)
    elif not regression:
        glm_train = ad.glm.binomial(y=y_train.to_numpy(), dtype=np.float64)
    else:
        glm_train = ad.glm.gaussian(y=y_train.to_numpy(), dtype=np.float64)

    # assess model
    model = grpnet(
        X=x_train_scaled.to_numpy(),
        glm=glm_train,
        ddev_tol=0,
        early_exit=False,
        n_threads=n_threads,
        min_ratio=lambda_min_ratio,
        progress_bar=False,
        lmda_path_size=lmda_path_size,
        alpha=alpha,
        penalty=pf / np.sum(pf) * x_train_scaled.shape[1],
    )

    (test_error_raw, roc_auc_raw, non_zero_coefs_raw) = eval_grpnet_model(
        model, x_test_scaled, y_test, multinomial, regression, n_threads
    )

    (feature_count_raw, signs_raw, magnitudes_raw) = count_feature_usage(
        model.betas.toarray(),  multinomial, x_train_scaled.columns,
        tolerance=tolerance
    )

    df = pd.DataFrame({
        'n_features': non_zero_coefs_raw,
        'test_error': test_error_raw,
        "auroc": roc_auc_raw,
        "lmda": model.lmda_path
    })
    return pd.concat([df, feature_count_raw, signs_raw, magnitudes_raw], axis=1)


def run_baseline_xgboost(
    train_test: TrainTest,
    ordered_features: list[str],
    regression=False,
    n_points: int = 40,
    max_features: int = None,
):
    """
    For a list of features ordered by importance, runs the XGBoost model on
    increasingly-large subsets of the most important features. Computes test
    error and AUROC for each subset.

    Parameters:
    - train_test: training and test splits, as a TrainTest object
    - ordered_features: ordered list of features, with the most important first
    - regression: whether this is a regression problem
    - n_points: number of different feature subsets to evaluate
    - max_features: maximum number of features in an evaluated subset

    Returns: DataFrame with test error, AUROC, and feature selected.
    """
    (x_train, x_test, y_train, y_test) = train_test.to_tuple()

    model_name = "pure_xgboost"

    if len(y_train.unique()) <= 2:
        params =  {'objective': 'binary:logistic'}
    elif not regression:
        params =  {'objective': 'multi:softmax'}
    else:
        params =  {'objective': 'reg:squarederror'}
    params["eval_metric"] = ['error', 'auc']
    
    results = None
    
    df_cols = ["method", "test_error", "auroc", "n_features"]
    df_cols += [f"{feat}_Selected" for feat in x_train.columns]
    df_cols += [f"{feat}_Sign" for feat in x_train.columns]
    df_cols += [f"{feat}_Magnitude" for feat in x_train.columns]
        
    n_features = len(ordered_features) if max_features is None else max_features
    n_features = min(n_features, len(ordered_features))
    step = int(np.ceil(n_features / n_points))

    for i in np.arange(1, n_features+1, step):
        top_features = ordered_features[:i]

        x_subset_train = x_train[top_features]
        x_subset_test = x_test[top_features]

        data = xgb.DMatrix(x_subset_train, label=y_train.to_numpy())
        data_eval = xgb.DMatrix(x_subset_test, label=y_test.to_numpy())

        bst = xgb.train(params, data)
        eval = bst.eval(data_eval)

        test_error = float(re.match(".*error:([0-9.]*)", eval).group(1))
        test_auc = float(re.match(".*auc:([0-9.]*)", eval).group(1))

        selected, sign, magnitude = count_feature_usage_subset(
            xgb.XGBClassifier(objective=params['objective']).fit(
                x_subset_train, y_train
            ).feature_importances_[np.newaxis, :],
            top_features, x_test.columns
        )

        df = pd.DataFrame([
            [
                model_name, test_error, test_auc, i
            ] + selected + sign + magnitude
        ], columns=df_cols)

        if results is not None:
            results = pd.concat([df, results], ignore_index=True)
        else:
            results = df

    return results