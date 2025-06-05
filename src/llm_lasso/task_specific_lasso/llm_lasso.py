from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from adelie.cv import cv_grpnet
from adelie.solver import grpnet
from llm_lasso.task_specific_lasso.utils import *
from tqdm import tqdm
from dataclasses import dataclass
from enum import IntEnum


class CrossValMetric(IntEnum):
    ERROR = 0
    AUROC = 1
    # LOSS = 2


@dataclass
class LLMLassoExperimentConfig:
    """
    Configuration parameters passed into LLM-Lasso experiments.

    TODO: document individual parameters
    """
    folds_cv: int = 10
    regression: bool = False
    max_features_for_baselines: int = 50
    max_feature_granularity: int = 50
    n_threads: int = 8
    seed: int = 7235
    model_name: str = None
    tolerance: float = 1e-10

    # Lasso config
    lambda_min_ratio: float = 0.001
    lambda_path_size: int = 200
    relaxed_lasso: bool = False
    lasso_downstream_l2: bool = False
    run_pure_lasso_after: int = None
    max_imp_power: int = 3
    score_type: int = PenaltyType.PF
    adaptive_lasso_relative_imp_base: float = 0.2
    cross_val_metric: int = CrossValMetric.ERROR

    remove_correlated_features: bool = False
    correlation_thresh: float = 0.9


def run_downstream_baselines_for_splits(
    splits: list[TrainTest],
    feature_baseline: dict[str, list[list[str]]],
    config: LLMLassoExperimentConfig
):
    """
    [LLM-LASSO TOP-LEVEL EXPERIMENT]

    Runs baseline methods for each training/test split provided by `splits`.

    Parameters:
    - splits: list of training and test splits, as TrainTest objects
    - feature_baseline: mapping of baseline model name to a list of importance-
        ordered features, for each split. e.g.,
        ```
        feature_baseline={
            "xgboost": [
                [split1_important_features...],
                [split2_important_features...],
                ...
            ], ...
        }
        ```
        where the ordered features have the most important feature first.
    - config: `LLMLassoExperimentConfig` object

    Returns: DataFrame with test error, AUROC, selected features, and other
        metadata for each split. 
    """
    all_results = None
    for split_idx in tqdm(range(len(splits))):
        split_baseline = {}
        for name in feature_baseline.keys():
            ordered_features = feature_baseline[name][split_idx]
            n_features = len(ordered_features) \
                if config.max_features_for_baselines is None \
                    else config.max_features_for_baselines
            n_features = min(n_features+1, len(ordered_features))
            step = int(np.ceil(n_features / config.max_feature_granularity))

            split_baseline[name] = [
                ordered_features[:i] for i in np.arange(1, n_features, step)
            ]
        
        res = pd.concat([
            pd.concat([
                run_l2_regression(
                    train_test=splits[split_idx],
                    features=features,
                    model_name=model_name,
                    regression=config.regression,
                    folds_cv=config.folds_cv,
                    seed=config.seed,
                    n_threads=config.n_threads
                ) for features in split_baseline[model_name]
            ], ignore_index=True) for model_name in split_baseline
        ], ignore_index=True).copy()
        res["split"] = split_idx
        res["model"] = "Baseline"
        res["is_baseline"] = True
        res["method_model"] = res["method"]

        if all_results is None:
            all_results = res
        else:
            all_results = pd.concat([all_results, res], ignore_index=True)   
        
    return all_results


def run_lasso_baseline_for_splits(
    splits: list[TrainTest],
    config: LLMLassoExperimentConfig,
):
    """
    [LLM-LASSO TOP-LEVEL EXPERIMENT]

    Runs the Lasso baseline for each training/test split provided by `splits`.

    Parameters:
    - splits: list of training and test splits, as TrainTest objects
    - config: `LLMLassoExperimentConfig` object

    Returns: DataFrame with test error, AUROC, selected features, and other
        metadata for each split. 
    """
    model_name = config.model_name if config.model_name is not None else "Lasso"
    all_results = None
    for split_idx in tqdm(range(len(splits))):
        res = run_llm_lasso_and_maybe_remove_correlated_features(
            train_test=splits[split_idx],
            scores=np.ones(splits[split_idx].x_train.shape[1]),
            config=config,
            verbose=False,
            max_imp_pow=0
        )
        res["model"] = model_name
        res["is_baseline"] = False
        res["split"] = split_idx
        res["method_model"] = model_name

        if all_results is None:
            all_results = res
        else:
            all_results = pd.concat([all_results, res], ignore_index=True)   
        
    return all_results


def run_adaptive_lasso_for_splits(
    splits: list[TrainTest],
    config: LLMLassoExperimentConfig,
):
    """
    [LLM-LASSO TOP-LEVEL EXPERIMENT]

    Runs the adaptive Lasso for each training/test split provided by `splits`.
    The weights for adaptive Lasso are found by taking the feature weight
    magnitudes for a l2-regularized model, run with all features.

    Parameters:
    - splits: list of training and test splits, as TrainTest objects
    - config: `LLMLassoExperimentConfig` object

    Returns: DataFrame with test error, AUROC, selected features, and other
        metadata for each split. 
    """
    model_name = config.model_name if config.model_name is not None else "Adaptive_Lasso"
    all_results = None
    for split_idx in tqdm(range(len(splits))):
        weights = run_l2_regression(
            train_test=splits[split_idx],
            features=splits[split_idx].x_train.columns,
            model_name=model_name,
            regression=config.regression,
            folds_cv=config.folds_cv,
            seed=config.seed,
            n_threads=config.n_threads
        )[[f"{feat}_Magnitude" for feat in splits[split_idx].x_train.columns]].to_numpy()[0, :]
        weights += weights.max() * config.adaptive_lasso_relative_imp_base

        if config.score_type == PenaltyType.PF:
            weights = 1/weights
        
        res = run_llm_lasso_and_maybe_remove_correlated_features(
            train_test=splits[split_idx],
            scores=weights,
            config=config,
            verbose=False,
            max_imp_pow=config.max_imp_power
        )

        res["split"] = split_idx
        res["model"] = model_name
        res["is_baseline"] = False
        res["method_model"] = model_name

        if all_results is None:
            all_results = res
        else:
            all_results = pd.concat([all_results, res], ignore_index=True)

    return all_results


def run_xgboost_for_splits(
    splits: list[TrainTest],
    ordered_features: list[list[str]],
    config: LLMLassoExperimentConfig
):
    """
    [LLM-LASSO TOP-LEVEL EXPERIMENT]

    Runs the XGBoost model baseline (as opposed to the XGBoost feature selector
    with downstream logistic regression) for each training/test split provide
    by `splits`.

    Parameters:
    - splits: list of training and test splits, as TrainTest objects
    - ordered_featurs: list of importance-ordered features for each split, e.g.
        ```
        [
            [split1_important_features...],
            [split2_important_features...],
                ...
        ]
        ```
        where the ordered features have the most important feature first.
    - config: `LLMLassoExperimentConfig` object

    Returns: DataFrame with test error, AUROC, selected features, and other
        metadata for each split. 
    """
    model_name = config.model_name if config.model_name is not None else "XGBoost_Model"
    all_results = None
    for split_idx in tqdm(range(len(splits))):
        res = run_baseline_xgboost(
            splits[split_idx],
            ordered_features=ordered_features[split_idx],
            regression=config.regression,
            n_points=config.max_feature_granularity,
            max_features=config.max_features_for_baselines
        )
        res["model"] = model_name
        res["is_baseline"] = True
        res["split"] = split_idx
        res["method_model"] = model_name

        if all_results is None:
            all_results = res
        else:
            all_results = pd.concat([all_results, res], ignore_index=True)   
        
    return all_results


def run_llm_lasso_cv_for_splits(
    splits: list[TrainTest],
    scores: dict[str, np.array],
    config: LLMLassoExperimentConfig,
    preselected_genes: list[list[str]] = None,
    score_trial_list: dict[str, list[np.array]] = None,
    verbose: bool = False
):
    """
    [LLM-LASSO TOP-LEVEL EXPERIMENT]

    Runs LLM-Lasso for each training/test split provided by `splits`, with
    cross-validation to choose the form of the penalty factors (i.e., power
    of 1/importance).

    Parameters:
    - splits: list of training and test splits, as TrainTest objects
    - scores: mapping of the model used to produce the scores to a list of
        penalty factors or importance scores for each feature
    - config: `LLMLassoExperimentConfig` object
    - score_trial_list: optional argument; if scores were collected using
        multiple trials, passing in this argument enables use of cross-
        validation to select the best set of scores
    - verbose: whether to print some debugging statements

    Returns: DataFrame with test error, AUROC, selected features, and other
        metadata for each split. 
    """
    all_results = None
    model_names = scores.keys()
    for split_idx in tqdm(range(len(splits))):
        for model in model_names:
            if verbose:
                print(f"Running model: {model}")
            pf = scores[model]
            this_model_score_trials = score_trial_list[model] \
                if (score_trial_list is not None and model in score_trial_list) \
                    else None
            if len(pf.shape) == 2:
                pf = pf[split_idx, :]
                if this_model_score_trials is not None:
                    this_model_score_trials = this_model_score_trials[:, split_idx]

            curr_split = splits[split_idx]
            excluded_features = None
            if preselected_genes is not None:
                genes = preselected_genes[split_idx]
                all_genes = {x: i for i, x in enumerate(splits[split_idx].x_train.columns)}
                gene_idxs = [all_genes[x] for x in genes]
                curr_split = TrainTest(
                    splits[split_idx].x_train[genes],
                    splits[split_idx].x_test[genes],
                    splits[split_idx].y_train,
                    splits[split_idx].y_test,
                )
                if this_model_score_trials is not None:
                    this_model_score_trials = this_model_score_trials[genes, :]
                pf = pf[gene_idxs]
                excluded_features = list(set(splits[split_idx].x_train.columns) - set(genes))

            res = run_llm_lasso_and_maybe_remove_correlated_features(
                train_test=curr_split,
                scores=pf,
                config=config,
                score_trial_list=this_model_score_trials,
                verbose=verbose,
                max_imp_pow=config.max_imp_power
            )
            res["split"] = split_idx
            res["model"] = model
            res["is_baseline"] = False
            if excluded_features is not None:
                res = pd.concat([res, pd.DataFrame({
                    f"{feat}_Selected": [False] * res.shape[0] for feat in excluded_features
                })], axis=1).copy()
                res = pd.concat([res, pd.DataFrame({
                    f"{feat}_Sign": [0] * res.shape[0] for feat in excluded_features
                })], axis=1).copy()
                res = pd.concat([res, pd.DataFrame({
                    f"{feat}_Magnitude": [0] * res.shape[0] for feat in excluded_features
                })], axis=1).copy()

            if all_results is None:
                all_results = res
            else:
                all_results = pd.concat([all_results, res], ignore_index=True)
    all_results['method_model'] = all_results.apply(
        lambda row: f"{row['method']} - {row['model']}",
        axis=1
    )
    return all_results


def run_llm_lasso_and_maybe_remove_correlated_features(
    train_test: TrainTest,
    scores: np.array,
    config: LLMLassoExperimentConfig,
    score_trial_list: list[np.array] = None,
    verbose: bool = False,
    max_imp_pow: int = 5
):
    """
    Runs LLM-Lasso for a specified training/test split. If specified, removes
    features that have correlations above a certain threshold, replacing them
    with a representative feature (chosen based on which has the highest
    importance score/lowest penalty).

    Parameters:
    - train_test: training and test split, as a TrainTest object
    - scores: a list of penalty factors or importance scores for each feature
    - config: `LLMLassoExperimentConfig` object
    - score_trial_list: optional argument; if scores were collected using
        multiple trials, passing in this argument enables use of cross-
        validation to select the best set of scores
    - verbose: whether to print some debugging statements
    - max_imp_power: maximum power of (1/importance) to consider when using
        cross-validation to determine the best form of the penalty factors

    Returns: DataFrame with test error, AUROC, selected features, and other
        metadata for each split. 
    """

    if not config.remove_correlated_features:
        # Just run LLM-Lasso as normal
        return llm_lasso_cv(
            train_test=train_test,
            score=scores,
            regression=config.regression,
            lambda_min_ratio=config.lambda_min_ratio,
            score_type=config.score_type,
            folds_cv=config.folds_cv,
            seed=config.seed,
            tolerance=config.tolerance,
            n_threads=config.n_threads,
            max_imp_pow=max_imp_pow,
            scores_list=score_trial_list,
            relaxed=config.relaxed_lasso,
            downstream_l2=config.lasso_downstream_l2,
            lmda_path_size=config.lambda_path_size,
            verbose=verbose,
            run_pure_lasso_after=config.run_pure_lasso_after,
            cross_val_metric=config.cross_val_metric
        )
    
    # Change importance scores to penalties, if relevant
    penalties = scores
    if config.score_type == PenaltyType.IMP:
        penalties = 1/penalties

    # Compute groups of correlated features
    groups = []
    X = train_test.x_train
    feature_to_group = {}
    corr = X.corr().abs()

    all_groups = set()
    for feat in X.columns:
        # Find all features that are correlated with the current feature,
        # and are not already in a group
        group = set(
            corr.index[corr[feat] > config.correlation_thresh].tolist()
        ) - all_groups
        if len(group) == 0:
            continue
        
        # Populate list of groups and feature_to_group mapping
        if feat in feature_to_group:
            group_id = feature_to_group[feat]
            groups[group_id] = groups[group_id].union(group)
        else:
            group_id = len(groups)
            groups.append(group)
        for group_feat in group:
            feature_to_group[group_feat] = group_id
        all_groups = all_groups.union(group)
    
    # mapping of feature name to penalty factor
    feat_to_penalty = {
        X.columns[i]: penalties[i] for i in range(X.shape[1])
    }

    # Choose a representative feature for each group, and record the penalty
    # factor of that feature
    new_features = []
    penalties_for_new_features = []
    for group in groups:
        group = list(group)
        idx = np.argmin([feat_to_penalty[feat] for feat in group])
        new_features.append(group[idx])
        penalties_for_new_features.append(feat_to_penalty[group[idx]])
    
    # Keep track of which features were excluded
    excluded_features = list(set(X.columns) - set(new_features))
    assert len(excluded_features) + len(new_features) == len(X.columns)

    # Select representative features in training/test data
    train_test = TrainTest(
        x_train=train_test.x_train[new_features],
        x_test=train_test.x_test[new_features],
        y_train=train_test.y_train,
        y_test=train_test.y_test
    )

    # Run LLM-Lasso with correlated features removed
    df = llm_lasso_cv(
        train_test=train_test,
        score=np.array(penalties_for_new_features),
        regression=config.regression,
        lambda_min_ratio=config.lambda_min_ratio,
        score_type=PenaltyType.PF,
        folds_cv=config.folds_cv,
        seed=config.seed,
        tolerance=config.tolerance,
        n_threads=config.n_threads,
        max_imp_pow=max_imp_pow,
        scores_list=score_trial_list,
        relaxed=config.relaxed_lasso,
        downstream_l2=config.lasso_downstream_l2,
        lmda_path_size=config.lambda_path_size,
        verbose=verbose,
        cross_val_metric=config.cross_val_metric
    )

    # Add feature selection data for the removed features to the dataframe,
    # so it has the same columns as other dataframes returned by LLM-Lasso
    # experiments
    df = pd.concat([df, pd.DataFrame({
        f"{feat}_Selected": [False] * df.shape[0] for feat in excluded_features
    })], axis=1).copy()
    df = pd.concat([df, pd.DataFrame({
        f"{feat}_Sign": [0] * df.shape[0] for feat in excluded_features
    })], axis=1).copy()
    df = pd.concat([df, pd.DataFrame({
        f"{feat}_Magnitude": [0] * df.shape[0] for feat in excluded_features
    })], axis=1).copy()
    return df
    


def llm_lasso_cv(
    train_test: TrainTest,
    score: np.array,
    regression: bool = False,
    lambda_min_ratio = 0.01,
    lmda_path_size=200,
    score_type: int = PenaltyType.PF,
    folds_cv = 5,
    seed=0,
    tolerance=1e-10,
    n_threads=4,
    max_imp_pow=5,
    scores_list: list[np.array] = None,
    relaxed=False,
    downstream_l2=False,
    run_pure_lasso_after=None,
    verbose=False,
    cross_val_metric=CrossValMetric.ERROR
):
    """
    Runs LLM-Lasso for a specified training/test split, using cross-validation
    to choose the form of the penalty factors (which power of 1/importance, or 
    which set of penalties from `scores_list` to choose, if `scores_list` is
    provided).

    Optionally, uses LLM-Lasso selected features to run downstream l2-
    regularized regression or relaxed Lasso.

    Parameters:
    - train_test: training and test split, as a TrainTest object
    - score: a list of penalty factors or importance scores for each feature

    Optional Parameters:
    - lambda_min_ratio: the maximum lambda (l1 regularization strength) is 
        automatically chosen such that the model chooses zero features.
        `lambda_min_ratio` is multiplied by that value to get the minimum
        lambda used
    - lmda_path_size: number of different lambda values to try
    - score_type: whether the scores provided are importance scores or penalty
        factors
    - folds_cv: number of cross-validation folds to use
    - seed: random seed for choosing cross-validation folds
    - tolerance: numerical tolerance for determining feature usage
    - n_threads: number of threads to use for computation
    - max_imp_power: maximum power of (1/importance) to consider when using
        cross-validation to determine the best form of the penalty factors
    - scores_list: if scores were collected using multiple trials, passing in
        this argument enables use of cross-validation to select the best set
        of scores
    - relaxed: whether to apply relaxed Lasso downstream
    - downstream_l2: whether to apply an l2-regularized model downstream. This
        is mutually exclusive with relaxed Lasso
    - run_pure_lasso_after: if an integer is passed in and we have downstream
        relaxed Lasso or l2-regularized regression, we omit the downstream model
        for more than `run_pure_lasso_after` features
    - verbose: whether to print some debugging statements

    Returns: DataFrame with test error, AUROC, selected features, and other
        metadata for each split. 
    """
    (x_train, x_test, y_train, y_test) = train_test.to_tuple()

    multinomial = not regression and len(y_train.unique()) > 2

    if score_type == PenaltyType.IMP:
        penalty = 1 / score
    elif score_type == PenaltyType.PF:
        penalty = score
    else:
        assert False, "score_type must be either PenaltyType.IMP or PenaltyType.PF"

    x_train_scaled = standardize_array(x_train)
    x_test_scaled = standardize_array(x_test, center=x_train.mean(axis=0), scale=x_train.std(axis=0))

    # Compute the forms of the penalty factors to try
    penalty = penalty / np.sum(penalty) * x_train.shape[1]
    pf_list = []
    pf_types = []
    for i in range(0, max_imp_pow+1):
        pf_list.append(penalty ** i)
        pf_types.append(f"1/imp^{i}")

    if scores_list is not None:
        for trial in range(len(scores_list)):
            for i in range(1, max_imp_pow+1):
                trial_penalty = scores_list[trial] \
                    if score_type == PenaltyType.PF \
                    else 1 / scores_list[trial]
                pf_list.append(trial_penalty ** i)
                pf_types.append(f"1/imp^{i}-trial{trial}")

    # Cross-validation variables
    ref_cvm = None
    ref_nonzero = None
    best_cvm = None
    best_cv_area = -float('inf')
    best_model = None
    best_model_pf = None

    # Initialize an Adelie GLM for the corresponding type of problem
    if multinomial:
        one_hot_encoder = OneHotEncoder(sparse_output=False)  # Use dense array
        y_one_hot = one_hot_encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))
        glm_train = ad.glm.multinomial(y=y_one_hot, dtype=np.float64)
    elif not regression:
        glm_train = ad.glm.binomial(y=y_train.to_numpy(), dtype=np.float64)
    else:
        glm_train = ad.glm.gaussian(y=y_train.to_numpy(), dtype=np.float64)

    # Try out all of the penalty factor forms and choose the best one using
    # cross-validation
    for (pf, pf_type) in zip(pf_list, pf_types):
        if verbose:
            print(f"Running pf_type {pf_type}")
        if np.any(np.isnan(pf)):
            continue

        pf = pf / np.sum(pf) * x_train.shape[1]
        try:
            fit = cv_grpnet(
                X=x_train_scaled.to_numpy(),
                glm = glm_train,
                seed=seed,
                n_folds=folds_cv,
                min_ratio=lambda_min_ratio,
                lmda_path_size=lmda_path_size,
                penalty=pf,
                n_threads=n_threads,
                progress_bar=False,
                stratified_split=not regression,
                early_exit=False
            )                
        except Exception as e:
            print("ERROR", e)
            continue

        # cross-validation metric
        if cross_val_metric == CrossValMetric.ERROR:
            cvm = fit.test_error
        else:
            cvm = -fit.roc_auc

        non_zero = [
            np.count_nonzero(np.mean(np.abs(clss), axis=0) > tolerance)
                for clss in fit.betas[0]
        ]

        if ref_cvm is None:
            ref_cvm = cvm
            ref_nonzero = non_zero

        cv_area = cve(cvm, non_zero, ref_cvm, ref_nonzero)
        if cv_area > best_cv_area:
            if verbose:
                print(pf_type, cv_area, best_cv_area)
            best_cv_area = cv_area
            best_model = pf_type[i]
            best_model_pf = pf
            best_cvm = cvm

    # evaluate best model
    df = run_one_lasso(
        x_train_scaled, y_train,
        x_test_scaled, y_test,
        best_model_pf,
        lambda_min_ratio,
        lmda_path_size,
        multinomial, regression,
        n_threads, tolerance
    )
    df.insert(len(df.columns), "cvm", best_cvm)

    # Group by 'non_zero_coefs' and filter rows where the cross validation
    # metric is the minimum for each group
    df = (
        df.loc[df.groupby('n_features')['cvm'].idxmin()]
        .reset_index(drop=True)
    )
    df = df.sort_values(axis=0, by=["n_features"]).reset_index(drop=True)

    if relaxed:
        # Run downstream relaxed Lasso
        df2 = df.copy()
        test_errors = df["test_error"].tolist()
        aurocs = df["auroc"].tolist()
        n_features = df["n_features"].tolist()

        for (i, row) in enumerate(df[[f"{feat}_Selected" for feat in x_train.columns]].iterrows()):
            idxs = np.where(row[1])[0]
            feat = list(x_train.columns[idxs])
            relaxed_pf = np.copy(best_model_pf)
            relaxed_pf[idxs] /= 10

            try:
                model = grpnet(
                    X=x_train_scaled.to_numpy(),
                    glm=glm_train,
                    ddev_tol=0,
                    early_exit=False,
                    lmda_path=[df["lmda"][i]],
                    penalty=relaxed_pf / np.sum(relaxed_pf) * x_train.shape[1],
                    n_threads=n_threads,
                    alpha=1 - 1/(i+1),
                    progress_bar=False,
                )
                (test_error_raw, auroc_raw, non_zero_coefs_raw) = eval_grpnet_model(
                    model, x_test_scaled, y_test, multinomial, regression, n_threads
                )
            except:
                continue

            test_errors[i] = test_error_raw[0]
            aurocs[i] = auroc_raw[0]
            n_features[i] = non_zero_coefs_raw[0]

            if run_pure_lasso_after is not None and i > run_pure_lasso_after:
                break
            
        df2["test_error"] = test_errors
        df2["auroc"] = aurocs
        df2["n_features"] = n_features
        df = df2

    elif downstream_l2:
        # Run downstream l2-regularized regression
        features = []
        for row in df[[f"{feat}_Selected" for feat in x_train.columns]].iterrows():
            feat = list(x_train.columns[np.where(row[1])[0]])
            features.append(feat)
            if run_pure_lasso_after is not None and len(features) == run_pure_lasso_after:
                break

        best_model_pf = best_model_pf / np.sum(best_model_pf) * x_train.shape[1]
        df2 = pd.concat([run_l2_regression(
            train_test=train_test,
            features=lst,
            model_name=pf_type,
            regression=regression,
            folds_cv=folds_cv,
            n_threads=n_threads,
            seed=seed,
            standardize=False
        ) for lst in features[1:]], ignore_index=True).copy()
        df2 = df2.sort_values(axis=0, by=["n_features"]).reset_index(drop=True)
        df2.index += 1

        idxs = np.where((df2["cvm"] < df["cvm"][1:len(features)]).tolist())[0] + 1
        # idxs = np.arange(1, len(features))
        if len(idxs) > 0:
            df.loc[idxs, ["test_error", "auroc"]] = df2.loc[idxs, ["test_error", "auroc"]]

    df["best_method_model"] = best_model
    df["method"] = "1/imp"
    
    return df