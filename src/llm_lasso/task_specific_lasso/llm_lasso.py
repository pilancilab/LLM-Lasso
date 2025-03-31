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
import xgboost as xgb
from llm_lasso.task_specific_lasso.penalty_descent import penalty_descent_llm_lasso
from llm_lasso.task_specific_lasso.utils import PenaltyType, scale_cols, count_feature_usage
import regex as re


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
    max_imp_pow=5,
    score_trial_list: dict[str, list[np.array]] = None,
    bagging=False,
    bagging_m=5,
    run_llm_xgboost=False,
    penalty_descent=False,
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
            importance_scores = scores[model]
            scores_list = score_trial_list[model] if (
                score_trial_list is not None and model in score_trial_list
            ) else None

            # run_repeated_llm_lasso_cv
            res = llm_lasso_cv(
                x_train=x_train_splits[split_idx],
                x_test=x_test_splits[split_idx],
                y_train=y_train_splits[split_idx],
                y_test=y_test_splits[split_idx],
                score=importance_scores,
                score_type=score_type,
                folds_cv=folds_cv,
                seed=seed + split_idx,
                lambda_min_ratio=lambda_min_ratio,
                n_threads=n_threads,
                regression=regression,
                alpha=elasticnet_alpha,
                max_imp_pow=max_imp_pow,
                scores_list=scores_list,
            )
            res["split"] = split_idx
            res["model"] = model
            res["is_baseline"] = False
            all_results = pd.concat([all_results, res], ignore_index=True)

            if bagging:
                print("Running bagging")
                res = llm_lasso_cv_bagged(
                    x_train=x_train_splits[split_idx],
                    x_test=x_test_splits[split_idx],
                    y_train=y_train_splits[split_idx],
                    y_test=y_test_splits[split_idx],
                    score=importance_scores,
                    score_type=score_type,
                    folds_cv=folds_cv,
                    seed=seed + split_idx,
                    lambda_min_ratio=lambda_min_ratio,
                    n_threads=n_threads,
                    regression=regression,
                    bagging_m=bagging_m,
                    num_features=res['n_features'].max()
                )
                res["split"] = split_idx
                res["model"] = model
                res["is_baseline"] = False
                all_results = pd.concat([all_results, res], ignore_index=True)

            if run_llm_xgboost:
                print("Runing LLM XGBoost")
                res = llm_xgboost(
                    x_train=x_train_splits[split_idx],
                    x_test=x_test_splits[split_idx],
                    y_train=y_train_splits[split_idx],
                    y_test=y_test_splits[split_idx],
                    score=importance_scores,
                    regression=regression,
                    score_type=score_type,
                    folds_cv=folds_cv,
                    seed=seed,
                    n_threads=n_threads,
                    num_features=res['n_features'].max()
                )
                res["split"] = split_idx
                res["model"] = model
                res["is_baseline"] = False
                all_results = pd.concat([all_results, res], ignore_index=True)

            if penalty_descent:
                print("Running penalty descent Lasso")
                res = penalty_descent_llm_lasso(
                    x_train=x_train_splits[split_idx],
                    x_test=x_test_splits[split_idx],
                    y_train=y_train_splits[split_idx],
                    y_test=y_test_splits[split_idx],
                    score=importance_scores,
                    score_type=score_type,
                    n_splits=folds_cv,
                    seed=seed,
                    n_threads=n_threads,
                    lambda_min_ratio=lambda_min_ratio,
                    val_proportion=0.5,
                    max_iter=10,
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

        if "xgboost" in split_baseline.keys():
            res = run_baseline_xgboost(
                 x_train=x_train_splits[split_idx],
                x_test=x_test_splits[split_idx],
                y_train=y_train_splits[split_idx],
                y_test=y_test_splits[split_idx],
                max_features=all_results['n_features'].max(),
                feature_baseline=split_baseline["xgboost"], folds_cv=folds_cv,
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


def llm_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    score: np.array,
    regression: bool,
    score_type: int = PenaltyType.PF,
    folds_cv = 5,
    seed=0,
    n_threads=4,
    num_features=30,
):
    if score_type == PenaltyType.PF:
        imp = np.square(1 / score)
    elif score_type == PenaltyType.IMP:
        imp = score
    else:
        assert False, "score_type must be either PenaltyType.IMP or PenaltyType.PF"
    imp /= np.sum(imp)
    # imp = np.ones_like(score)

    data = xgb.DMatrix(x_train, label=y_train.to_numpy(), feature_weights=imp)
    unique = y_train.unique()
    if len(unique) <= 2:
        params =  {'objective': 'binary:logistic'}
    elif not regression:
        params =  {'objective': 'multi:softmax'}
    else:
        params =  {'objective': 'reg:squarederror'}

    bst = xgb.train(params, data)
    ranked_genes = [x[0] for x in sorted(bst.get_score(importance_type='weight').items(), key=lambda x:x[1], reverse=True)]

    df = run_baselines(
        x_train, y_train,
        x_test, y_test,
        feature_baseline={"LLM_XGBoost": ranked_genes},
        regression=regression,
        folds_cv=folds_cv,
        n_threads=n_threads,
        seed=seed,
        max_features=num_features
    )
    df["best_method_model"] = "bagging"
    return df


def llm_lasso_cv_bagged(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    score: np.array,
    bagging_m: int,
    regression: bool = False,
    lambda_min_ratio = 0.01,
    score_type: int = PenaltyType.PF,
    folds_cv = 5,
    seed=0,
    tolerance=1e-10,
    n_threads=4,
    num_features=30
):
    multinomial = not regression and len(y_train.unique()) > 2

    if score_type == PenaltyType.IMP:
        penalty = 1 / score
    elif score_type == PenaltyType.PF:
        penalty = score
    else:
        assert False, "score_type must be either PenaltyType.IMP or PenaltyType.PF"

    x_train_scaled = scale_cols(x_train)
    x_test_scaled = scale_cols(x_test, center=x_train.mean(axis=0), scale=x_train.std(axis=0))

    # penalty factors
    pf_list = [np.ones(x_train.shape[1]), penalty]
    score_names = ["Lasso", "1/imp"]
    # pf_list = [penalty]
    # score_names = ["1/imp"]
    results = pd.DataFrame(columns=["best_method_model", "method", "test_error", "auroc", "n_features"])

    for (score_name, pf) in zip(score_names, pf_list):
        pf = pf / np.sum(pf) * x_train_scaled.shape[1]
        # models = []
        magnitudes = np.zeros(len(pf))
        train_idxs = np.random.random_integers(0, x_train.shape[0] - 1, (bagging_m, x_train.shape[0]))
        for i in range(bagging_m):
            idxs = train_idxs[i, :]
            Xi =  np.asfortranarray(x_train_scaled.to_numpy()[idxs, :])
            yi = y_train.to_numpy()[idxs]

            # Initialize an Adelie GLM for the corresponding type of problem
            if multinomial:
                one_hot_encoder = OneHotEncoder(sparse_output=False)  # Use dense array
                y_one_hot = one_hot_encoder.fit_transform(yi.reshape(-1, 1))
                glm_train = ad.glm.multinomial(y=y_one_hot, dtype=np.float64)
            elif not regression:
                glm_train = ad.glm.binomial(y=yi, dtype=np.float64)
            else:
                glm_train = ad.glm.gaussian(y=yi, dtype=np.float64)

            if np.all(np.isnan(pf)):
                continue
            try:
                fit = grpnet(
                    X=Xi,
                    glm = glm_train,
                    alpha=1,
                    penalty=pf,
                    n_threads=n_threads,
                    progress_bar=False,
                    early_exit=False,
                    min_ratio=lambda_min_ratio / 10
                )
            except Exception as e:
                print("ERROR", e)
                continue

            (_, _, magnitudes_raw) = count_feature_usage(fit.betas.toarray(),  multinomial, x_train.shape[1], tolerance=tolerance)
            magnitudes += (magnitudes_raw).sum(axis=0)
            # models.append(fit)
        ranked_genes = x_train.columns[np.argsort(-magnitudes)]
        df = run_baselines(
            x_train_scaled, y_train,
            x_test_scaled, y_test,
            feature_baseline={score_name + "_bagging": ranked_genes},
            regression=regression,
            folds_cv=folds_cv,
            n_threads=n_threads,
            seed=seed,
            max_features=num_features
        )
        df["best_method_model"] = "bagging"
        if len(results) == 0:
            results = df
        else:
            results = pd.concat([results, df], ignore_index=True)

    return results


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
    max_imp_pow=5,
    scores_list: list[np.array] = None,
    logistic_regression_after=False,
    rerun=False
    # select_top_ratio_by_penalty: float = 0.5
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
        penalty = 1 / score
    elif score_type == PenaltyType.PF:
        penalty = score
    else:
        assert False, "score_type must be either PenaltyType.IMP or PenaltyType.PF"

    x_train_scaled = scale_cols(x_train)
    x_test_scaled = scale_cols(x_test, center=x_train.mean(axis=0), scale=x_train.std(axis=0))

    # penalty factors
    pf_list = [np.ones(x_train.shape[1])]
    pf_type = ["Lasso"]

    for i in range(0, max_imp_pow+1):
        pf_list.append(penalty ** i)
        pf_type.append(f"1/imp^{i}")

    if scores_list:
        for trial in range(len(scores_list)):
            for i in range(0, max_imp_pow+1):
                pf_list.append(scores_list[trial] ** i)
                pf_type.append(f"1/imp^{i}-trial{trial}")

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
                    lmda_path_size=200,
                    alpha=alpha,
                    penalty=pf / np.sum(pf) * x_train.shape[1],
                    n_threads=n_threads,
                    progress_bar=False,
                    early_exit=False
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
        df = run_one_lasso(
            x_train_scaled, y_train,
            x_test_scaled, y_test,
            best_model_pf, lambda_min_ratio,
            multinomial, regression,
            n_threads, tolerance
        )

        if logistic_regression_after:
            features = []
            for row in df[[f"Feature_{i+1}" for i in range(x_train.shape[1])]].iterrows():
                features.append(list(x_train.columns[np.where(row[1])[0]]))

            best_model_pf = best_model_pf / np.sum(best_model_pf) * x_train.shape[1]
            df = run_baselines(
                x_train_scaled / best_model_pf, y_train,
                x_test_scaled / best_model_pf, y_test,
                feature_baseline={score_name: features},
                regression=regression,
                folds_cv=folds_cv,
                n_threads=n_threads,
                seed=seed,
                normalize=False
            )
            df["best_method_model"] = best_model

        if rerun:
            idxs = np.where(df[[f"Feature_{i+1}" for i in range(x_train.shape[1])]].loc[df.shape[0]-1])[0]
            features = x_train.columns[idxs]
            print(features)

            df = run_one_lasso(
                x_train_scaled[features], y_train,
                x_test_scaled[features], y_test,
                best_model_pf[idxs], lambda_min_ratio,
                multinomial, regression,
                n_threads, tolerance,
            )

        df["best_method_model"] = best_model
        df["method"] = score_name
        

        # Group by 'non_zero_coefs' and filter rows where 'metric' is the minimum for each group
        df = (
            df.loc[df.groupby('n_features')['test_error'].idxmin()]
            .reset_index(drop=True)
        )
        
        if len(results) == 0:
            results = df
        else:
            results = pd.concat([results, df], ignore_index=True)
            

    return results


def run_one_lasso(
    x_train_scaled: pd.DataFrame, y_train: pd.Series,
    x_test_scaled: pd.DataFrame, y_test: pd.Series,
    pf: np.array,
    lambda_min_ratio: float,
    multinomial=False,
    regression=False,
    n_threads=4,
    tolerance=1e-10,
    alpha=1
):
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
        lmda_path_size=200,
        alpha=alpha,
        penalty=pf / np.sum(pf) * x_train_scaled.shape[1],
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
            for coeffs in model.betas.toarray().reshape((model.betas.shape[0], -1, x_train_scaled.shape[1]))
        ]
    else:
        non_zero_coefs_raw = [
            np.count_nonzero(np.abs(coeffs) > tolerance)
                for coeffs in model.betas.toarray()
        ]

    (feature_count_raw, signs_raw, magnitudes_raw) = count_feature_usage(model.betas.toarray(),  multinomial, x_train_scaled.shape[1], tolerance=tolerance)

    df = pd.DataFrame({
        'n_features': non_zero_coefs_raw,
        'test_error': test_error_raw,
        "auroc": roc_auc_raw,
        "lmda": model.lmda_path
    })
    return pd.concat([df, feature_count_raw, signs_raw, magnitudes_raw], axis=1)
    
    

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
    tolerance=1e-10,
    normalize=True
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

    if normalize:
        x_train_scaled = scale_cols(x_train)
        x_test_scaled = scale_cols(x_test, center=x_train.mean(axis=0), scale=x_train.std(axis=0))
    else:
        x_train_scaled = x_train
        x_test_scaled = x_test

    np.random.seed(seed)

    results = pd.DataFrame(columns=["method", "test_error", "auroc", "n_features"])
    
    for model_name in model_names:
        ordered_features = feature_baseline[model_name]
        
        n_features = len(ordered_features) if max_features is None else max_features
        n_features = min(n_features, len(ordered_features))
        step = int(np.ceil(n_features / n_points))

        for i in np.arange(1, n_features+1, step):
            if type(ordered_features[0]) == str:
                top_features = ordered_features[:i]
            else:
                if i == 1:
                    continue
                top_features = ordered_features[i-1]

            x_subset_train = x_train_scaled[top_features]
            x_subset_test = x_test_scaled[top_features]

            if not regression: # classigication
                model = LogisticRegressionCV(
                    Cs=[0.1, 0.5, 1, 5, 10, 50, 100],
                    # multi_class='multinomial',
                    penalty="l2",
                    random_state=seed,
                    n_jobs=n_threads,
                    scoring="accuracy",
                    refit=True,
                    cv=folds_cv
                ).fit(x_subset_train.to_numpy(), y_train.to_numpy())
                preds = model.predict_proba(x_subset_test.to_numpy())
                n_nonzero = i

                df = pd.DataFrame([
                    [
                        model_name,
                        1 - accuracy_score(y_test.to_numpy(), np.argmax(preds, axis=1)), # test error
                        roc_auc_score(y_test, preds[:, 1] if preds.shape[1] == 2 else preds, multi_class='ovr'), # AUROC
                        n_nonzero # feature count
                    ]
                ], columns=results.columns)
                if results.shape[0] > 0:
                    results = pd.concat([df, results], ignore_index=True)
                else:
                    results = df
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
    

def run_baseline_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_baseline: list[str],
    regression=False,
    n_points: int = 40,
    max_features: int = None,
    folds_cv = 5,
    seed = 0,
    n_threads = 4,
    tolerance=1e-10,
    normalize=True
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
    model_name = "pure_xgboost"

    if len(y_train.unique()) <= 2:
        params =  {'objective': 'binary:logistic'}
    elif not regression:
        params =  {'objective': 'multi:softmax'}
    else:
        params =  {'objective': 'reg:squarederror'}
    params["eval_metric"] = ['error', 'auc']
    

    results = pd.DataFrame(columns=["method", "test_error", "auroc", "n_features"])
    
    ordered_features = feature_baseline
    
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

        df = pd.DataFrame([
            [
                model_name, test_error, test_auc, i
            ]
        ], columns=results.columns)
        if results.shape[0] > 0:
            results = pd.concat([df, results], ignore_index=True)
        else:
            results = df

    return results