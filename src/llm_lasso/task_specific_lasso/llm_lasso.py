from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from adelie.cv import cv_grpnet
from adelie import grpnet
import adelie as ad
from adelie.diagnostic import auc_roc, test_error_hamming, test_error_mse, predict
from enum import IntEnum


class PenaltyType(IntEnum):
    IMP = 0
    PF = 1


def run_repeated_llm_lasso_cv(
    x_train_splits: list[pd.DataFrame],
    y_train_splits: list[pd.DataFrame],
    x_test_splits: list[pd.DataFrame],
    y_test_splits: list[pd.DataFrame],
    scores: dict[str, np.array],
    feature_baseline: dict[str, list[list[str]]],
    regression=False,
    n_splits=10,
    score_type: int = PenaltyType.IMP,
    n_threads = 4,
    folds_cv = 5,
    lambda_min_ratio = 0.01,
    seed = 7235,
    elasticnet_alpha=1,
    max_imp_pow=5
):
    """
      LLM-lasso comparison across models, with error bars for different test-train splits
  
        Parameters:
         - `x_train_splits`: training data, as a list of `pandas.DataFrame`
            objects per split (where the dataframe column names are the feature
            names).
        - `y_train`: output labels, as a list of `pandas.Series` objects per split.
        - `x_test`: testing data, in the same format as the training data.
        - `y_test`: testing labels, in the same format as the training labels.
        - `scores`: penalty factors or importance scores for each LLM-Lasso model.
        - `feature_baseline`: mapping between baseline name and a list of
            selected features for each split.
        - `regression`: whether this is a regression problem (as opposed to
            classification).
        - `n_splits`: number of splits to test, default 10
        - `score_type`: whether the scores are penalty factors or importance scores.
        - `folds_cv`: number of cross-validation folds.
        - `lambda_min_ratio`: lambda-min to lambda-max ratio.
        - `seed`: random seed.
        - `n_threads`: number of threads to use for model fitting.
        - `elasticnet_alpha`: elasticnet parameter (1 = pure l1 regularization,
            0 = pure l2) for LLM-Lasso.
        - `max_imp_pow`: maximum power to use for 1/imp model.
    """

    all_results = pd.DataFrame()
    model_names = scores.keys()

    for split_idx in range(n_splits):
        print(f"Processing split {split_idx} of {n_splits}")

        # Iterate over each model (Plain and RAG)
        for model in model_names:
            print(f"\tRunning model: {model}")
            impotance_scores = scores[model]

            # run_repeated_llm_lasso_cv
            res = llm_lasso_cv(
                x_train=x_train_splits[split_idx],
                x_test=x_test_splits[split_idx],
                y_train=y_train_splits[split_idx],
                y_test=y_test_splits[split_idx],
                score=impotance_scores,
                score_type=score_type,
                folds_cv=folds_cv,
                seed=seed + split_idx,
                lambda_min_ratio=lambda_min_ratio,
                n_threads=n_threads,
                regression=regression,
                alpha=elasticnet_alpha,
                max_imp_pow=max_imp_pow
            )
            res["split"] = split_idx
            res["model"] = model
            res["is_baseline"] = False
            all_results = pd.concat([all_results, res], ignore_index=True)

        # iterate over each baseline
        split_baseline = {}
        for name in feature_baseline.keys():
            split_baseline[name] = feature_baseline[name][split_idx]

        print(f"Running baselines")
        res = run_baselines(
            x_train=x_train_splits[split_idx],
            x_test=x_test_splits[split_idx],
            y_train=y_train_splits[split_idx],
            y_test=y_test_splits[split_idx],
            max_features=all_results['n_features'].max(),
            feature_baseline=split_baseline, folds_cv=folds_cv,
            seed=seed + split_idx, n_threads=n_threads,
            regression=regression
        )
        res["split"] = split_idx
        res["model"] = "Baseline"
        res["is_baseline"] = True
        all_results = pd.concat([all_results, res], ignore_index=True)

    baseline_names = all_results[all_results["is_baseline"] == True]["method"].unique()
    all_results['method_model'] = all_results.apply(
        lambda row: "Lasso" if row['method'] == "Lasso" else
                   row['method'] if row['method'] in baseline_names else
                    f"{row['method']} - {row['model']}",
        axis=1
    )
    return all_results


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



def count_feature_usage(model, multinomial, n_features, tolerance=1e-10):
    """
    Counts the number of times each feature is used in a multinomial glmnet model across all lambdas.

    Parameters:
    - `model`: Output of cv_grpnet from adelie.
    - `multinomial`: whether classification is multinomial.
    - `n_features`: number of features
    - `tolerance`: numerical tolerance for counting nonzero features.

    Returns:
    - A DataFrame where each row corresponds to a lambda value and each column indicates
      whether a feature is nonzero (True/False).
    """
    # Get the number of features (exclude intercept)
    if multinomial:
        feature_inclusion_matrix =  np.abs(model.betas.toarray().reshape((model.betas.shape[0], -1, n_features))).mean(axis=1) > tolerance
        sign_mtx = np.argmax(model.betas.toarray().reshape((model.betas.shape[0], -1, n_features)), axis=1)
    else:
        feature_inclusion_matrix = np.abs(model.betas.toarray()) > tolerance
        sign_mtx = np.sign(model.betas.toarray())

    # Convert to a DataFrame for easier interpretation
    feature_inclusion_df = pd.DataFrame(feature_inclusion_matrix, columns=[f"Feature_{j+1}" for j in range(n_features)])
    sign_df = pd.DataFrame(sign_mtx, columns=[f"Feature_Sign_{j+1}" for j in range(n_features)])
    return feature_inclusion_df, sign_df
    

def llm_lasso_cv(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    score: np.array,
    regression: bool = False,
    lambda_min_ratio = 0.01,
    score_type: int = PenaltyType.PF,
    folds_cv = 5,
    seed=0,
    tolerance=1e-10,
    n_threads=4,
    alpha=1,
    max_imp_pow=5
):
    """
    Creates LLM-lasso model and chooses the optimal i for 1/imp^i.
    
    Does multinomial/binomial classification, with test error as the cross-
    validation metric, and regression, with MSE as the cross-validation metric.
    
    Parameters:
    - `x_train`: training data, as a `pandas.DataFrame` where the column names
        are the feature names.
    - `y_train`: output labels, as a `pandas.Series`.
    - `x_test`: testing data, in the same format as the training data.
    - `y_test`: testing labels, in the same format as the training labels.
    - `score`: penalty factors or importance scores.
    - `regression`: whether this is a regression problem (as opposed to
        classification).
    - `score_type`: whether the scores are penalty factors or importance scores.
    - `folds_cv`: number of cross-validation folds.
    - `seed`: random seed.
    - `tolerance`: numerical tolerance when computing number of features chosen.
    - `n_threads`: number of threads to use for model fitting.
    - `alpha`: elasticnet parameter (1 = pure l1 regularization, 0 = pure l2).
    - `max_imp_pow`: maximum power to use for 1/imp model.
    """

    multinomial = not regression and len(y_train.unique()) > 2

    if score_type == PenaltyType.IMP:
        importances = score
    elif score_type == PenaltyType.PF:
        importances = 1 / score
    else:
        assert False, "score_type must be either PenaltyType.IMP or PenaltyType.PF"

    x_train_scaled = scale_cols(x_train)
    x_test_scaled = scale_cols(x_test, center=x_train.mean(axis=0), scale=x_train.std(axis=0))

    # penalty factors
    pf_list = [np.ones(x_train.shape[1])]
    pf_type = ["Lasso"]

    for i in range(0, max_imp_pow+1):
        pf_list.append(1 / importances ** i)
        pf_type.append(f"1/imp^{i}")

    score_names = ["Lasso", "1/imp"]
    results = pd.DataFrame(columns=["best_method_model", "method", "test_error", "auroc", "n_features"])

    ref_cvm = None
    ref_nonzero = None

    # Initialize an Adelie GLM for the corresponding type of problem
    if multinomial:
        one_hot_encoder = OneHotEncoder(sparse_output=False)  # Use dense array
        y_one_hot = one_hot_encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))
        glm_train = ad.glm.multinomial(y=y_one_hot, dtype=np.float64)
    elif not regression:
        glm_train = ad.glm.binomial(y=y_train.to_numpy(), dtype=np.float64)
    else:
        glm_train = ad.glm.gaussian(y=y_train.to_numpy(), dtype=np.float64)

    for score_name in score_names:
        indices = np.nonzero([score_name in pf for pf in pf_type])[0]

        # Perform cross-validation
        best_cv_area = -float('inf')
        best_model = None
        best_model_pf = None

        for i in indices:
            print(f"Running pf_type {pf_type[i]}")
            pf = pf_list[i]
            if np.all(np.isnan(pf)):
                continue
            try:
                fit = cv_grpnet(
                    X=x_train_scaled.to_numpy(),
                    glm = glm_train,
                    seed=seed,
                    n_folds=folds_cv,
                    min_ratio=lambda_min_ratio,
                    alpha=alpha,
                    penalty=pf / np.sum(pf) * x_train_scaled.shape[1],
                    n_threads=n_threads,
                    progress_bar=False
                )                
            except Exception as e:
                print("ERROR", e)
                continue

            # cross-validation metric
            cvm = fit.test_error

            non_zero = [
                np.count_nonzero(np.mean(np.abs(clss), axis=0) > tolerance)
                    for clss in fit.betas[0]
            ]

            if ref_cvm is None:
                ref_cvm = cvm
                ref_nonzero = non_zero

            cv_area = cve(cvm, non_zero, ref_cvm, ref_nonzero)
            if cv_area > best_cv_area:
                best_cv_area = cv_area
                best_model = pf_type[i]
                best_model_pf = pf

        # assess best model
        model = grpnet(
            X=x_train_scaled.to_numpy(),
            glm=glm_train,
            ddev_tol=0,
            early_exit=False,
            n_threads=n_threads,
            min_ratio=lambda_min_ratio,
            progress_bar=False,
            alpha=alpha,
            penalty=best_model_pf / np.sum(best_model_pf) * x_train_scaled.shape[1],
        )

        if multinomial:
            one_hot_encoder = OneHotEncoder(sparse_output=False)  # Use dense array
            y_one_hot = one_hot_encoder.fit_transform(y_test.to_numpy().reshape(-1, 1))
            y = y_one_hot
        else:
            y = y_test.to_numpy()

        etas = predict(
            X=x_test_scaled.to_numpy(),
            betas=model.betas,
            intercepts=model.intercepts,
            n_threads=n_threads,
        )
        
        if regression:
            test_error_raw = test_error_mse(etas, y)
        else:
            test_error_raw = test_error_hamming(etas, y, multinomial)

        if not regression:
            roc_auc_raw = auc_roc(etas, y, multinomial)
        else:
            roc_auc_raw = None

        if multinomial:
            non_zero_coefs_raw = [
                np.count_nonzero(np.mean(np.abs(coeffs), axis=0) > tolerance)
                for coeffs in model.betas.toarray().reshape((model.betas.shape[0], -1, x_test.shape[1]))
            ]
        else:
            non_zero_coefs_raw = [
                np.count_nonzero(np.abs(coeffs) > tolerance)
                    for coeffs in model.betas.toarray()
            ]

        (feature_count_raw, signs_raw) = count_feature_usage(model,  multinomial, x_train.shape[1], tolerance=tolerance)

        df = pd.DataFrame({
            'n_features': non_zero_coefs_raw,
            'test_error': test_error_raw,
            "auroc": roc_auc_raw
        })
        df = pd.concat([df, feature_count_raw, signs_raw], axis=1)
        df["best_method_model"] = best_model
        df["method"] = score_name

        # Group by 'non_zero_coefs' and filter rows where 'metric' is the minimum for each group
        df = (
            df.loc[df.groupby('n_features')['test_error'].idxmin()]
            .reset_index(drop=True)
        )
        results = pd.concat([results, df], ignore_index=True)

    return results


def run_baselines(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_baseline: dict[str, list[str]],
    regression=False,
    n_points: int = 20,
    max_features: int = None,
    folds_cv = 5,
    seed = 0,
    n_threads = 4,
    tolerance=1e-10
):
    """
    Runs downstream model for baseline feature selectors, i.e., l2-regularized
    logitic regression for classification and linear regression for
    regression. Sweeps the number of features chosen from 1 to
    `max_features`, with a total of `n_points`.
      
    Parameters:
    - `x_train`: training data, as a `pandas.DataFrame` where the column names
        are the feature names.
    - `y_train`: output labels, as a `pandas.Series`.
    - `x_test`: testing data, in the same format as the training data.
    - `y_test`: testing labels, in the same format as the training labels.
    - `feature_baseline`: dictionary mapping the name of the baseline feature
        selection model to a list of feature names, in reverse order of
        importance.
    - `regression`: whether this is a regression problem (as opposed to
        classification).
    - `n_points`: determines granularity of sweep from 1 feature to
        `max_features`.
    - `max_features`: maximum number of features to choose.
    - `folds_cv`: number of cross-validation folds.
    - `seed`: random seed.
    - `n_threads`: number of threads to use for model fitting.
    - `tolerance`: numerical tolerance when computing number of features chosen.
    """
    model_names = feature_baseline.keys()

    x_train_scaled = scale_cols(x_train)
    x_test_scaled = scale_cols(x_test, center=x_train.mean(axis=0), scale=x_train.std(axis=0))

    results = pd.DataFrame(columns=["method", "test_error", "auroc", "n_features"])
    
    for model_name in model_names:
        ordered_features = feature_baseline[model_name]

        n_features = len(ordered_features) if max_features is None else max_features
        n_features = min(n_features, len(ordered_features))
        step = int(np.ceil(n_features / n_points))

        for i in np.arange(1, n_features+1, step):
            top_features = ordered_features[:i]
            x_subset_train = x_train_scaled[top_features]
            x_subset_test = x_test_scaled[top_features]

            if not regression: # classigication
                model = LogisticRegressionCV(
                    Cs=[0.1, 0.5, 1, 5, 10, 50, 100],
                    multi_class='multinomial',
                    penalty="l2",
                    random_state=seed,
                    n_jobs=n_threads,
                    scoring="accuracy",
                    refit=True,
                    cv=folds_cv
                ).fit(x_subset_train.to_numpy(), y_train.to_numpy())
                preds = model.predict_proba(x_subset_test.to_numpy())

                if not len(model.coef_.shape) > 1:
                    n_nonzero = np.count_nonzero(np.mean(np.abs(model.coef_), axis=1) > tolerance)
                else:
                    n_nonzero = np.count_nonzero(np.abs(model.coef_) > tolerance)

                results = pd.concat([pd.DataFrame([
                    [
                        model_name,
                        1 - accuracy_score(y_test.to_numpy(), np.argmax(preds, axis=1)), # test error
                        roc_auc_score(y_test, preds[:, 1] if preds.shape[1] == 2 else preds, multi_class='ovr'), # AUROC
                        n_nonzero # feature count
                    ]
                ], columns=results.columns), results], ignore_index=True)
            else: # regression
                model = LinearRegression(
                    n_jobs=n_threads,
                ).fit(x_subset_train.to_numpy(), y_train.to_numpy())
                preds = model.predict(x_subset_test.to_numpy())
                results = pd.concat([pd.DataFrame([
                        [
                        model_name,
                        mean_squared_error(y_test.to_numpy(), preds), # test error
                        None, # AURIC is undefined for regression,
                        np.count_nonzero(np.abs(model.coef_) > tolerance) # feature count
                    ]
                ], columns=results.columns), results], ignore_index=True)

    return results

def scale_cols(x: pd.DataFrame, center=None, scale=None) -> pd.DataFrame:
    if center is None:
        center = np.mean(x, axis=0)
    if scale is None:
        scale = np.std(x, axis=0)
    
    return (x - center) / scale