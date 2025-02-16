from LLasso.helper import *
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from adelie.cv import cv_grpnet
from adelie import grpnet
import adelie as ad
from adelie.diagnostic import auc_roc, test_error_hamming, test_error_mse, predict

LASSO_COLOR = ["#999999"]
BASELINE_COLORS = ["#56B4E9", "#009292", "#117733", "#490092", "#924900"]
LLM_LASSO_COLORS = ["#D55E00", "#CC79A7", "#E69F00"] + BASELINE_COLORS

# TODO: this should probably be moved to a diff. file because it runs both
# our method and the baselines?
def run_repeated_llm_lasso_cv(
    x_train_splits,
    y_train_splits,
    x_test_splits,
    y_test_splits,
    importance_scores_list,
    model_names,
    baseline_names,
    feature_baseline,
    regression=False,
    n_splits=10,
    score_type = "pf",
    n_threads = 4,
    pfmax = [],
    folds_cv = 5,
    lambda_min_ratio = 0.01,
    methods = None,
    seed = 7235,
    quantize_gene_counts = False,
    n_gene_count_bins=20,
    bolded_methods=[],
    plot_error_bars = True,
    elasticnet_alpha=1,
    max_imp_pow=5
):
    """
      LLM-lasso comparison across models, with error bars for different test-train splits
  
        Parameters:
        - `xall`: predictors
        - `yall`: target
        - `importance_scores_list`: list of importance scores
        - `model_names`: vector of model names
        - `baseline_names`: vector of baseline model names
        - `n_splits`: number of splits to test, default 10
        - `score_type`: "pf" if penalty factors, "imp" if importance scores
        - `pfmax`: the maximum penalty factor. If not provided, will estimate the optimal pfmax such that the least important feature is always 0.  
        - `folds_cv`: number of folds in CV. Default is 5.
        - `lambda.min.ratio`: lambda-min to lambda-max ratio. Default for glmnet is 0.01.
        - `methods`: which methods to plot. Plots all if NULL.
        - `seed`: seed; for each split evaluated, 1 will be added to this number 
        - `feature_baseline`: list of baselines
    """

    all_results = pd.DataFrame()
    best_models = []
    for split_idx in range(n_splits):
        best_splt_models = {}
        print(f"Processing split {split_idx} of {n_splits}")

        # Iterate over each model (Plain and RAG)
        for model in model_names:
            print(f"\tRunning model: {model}")
            # Retrieve the importance scores for the current model
            impotance_scores = importance_scores_list[model]

            # run_repeated_llm_lasso_cv
            res = llm_lasso_cv(
                x_train=x_train_splits[split_idx],
                x_test=x_test_splits[split_idx],
                y_train=y_train_splits[split_idx],
                y_test=y_test_splits[split_idx],
                score=impotance_scores,
                score_type=score_type,
                pfmax=pfmax,
                folds_cv=folds_cv,
                seed=seed + split_idx,
                lambda_min_ratio=lambda_min_ratio,
                n_threads=n_threads,
                regression=regression,
                alpha=elasticnet_alpha,
                max_imp_pow=max_imp_pow
            )

            # Loop through each method in res['score_names']
            for method_idx, method_name in enumerate(res['score_names']):
                # Extract corresponding metric, nGenes, and feature_count values
                metric_values = res['test_error'][method_idx]
                auc_values = res['roc_auc'][method_idx]
                n_genes_values = res['non_zero_coefs'][method_idx]
                feature_count_values = res['feature_count'][method_idx] 

                if method_name != "Lasso":
                    best_splt_models[(model, method_name)] = res["best_model"][method_idx]

                # Create a temporary DataFrame for the current method
                temp_df = pd.DataFrame({
                    'split': split_idx,
                    'model': model,
                    'method': method_name,
                    'n_genes': n_genes_values,
                    'test_error': metric_values,
                    'roc_auc': auc_values,
                })

                temp_df = pd.concat([temp_df, feature_count_values.reset_index(drop=True)], axis=1)

                # Append to all_results
                all_results = pd.concat([all_results, temp_df], ignore_index=True)
        best_models.append(best_splt_models)

        # iterate over each baseline
        split_baseline = {}
        for name in baseline_names:
            split_baseline[name] = feature_baseline[name][split_idx]
        print(f"Running baselines")
        res = run_baseline(
            x_train=x_train_splits[split_idx],
            x_test=x_test_splits[split_idx],
            y_train=y_train_splits[split_idx],
            y_test=y_test_splits[split_idx],
            max_features=all_results['n_genes'].max(),
            feature_baseline=split_baseline, folds_cv=folds_cv,
            seed=seed + split_idx, n_threads=n_threads,
            regression=regression
        )

        for method_idx, method_name in enumerate(res['model_names']):
            metric_values = res['test_error'][method_idx]
            auc_values = res['roc_auc'][method_idx]
            n_genes_values = res['non_zero_coefs'][method_idx]
            feature_count_values = res['feature_count'][method_idx]

            # Create a temporary DataFrame for the current method
            temp_df = pd.DataFrame({
                'split': split_idx,
                'model': model,
                'method': method_name,
                'n_genes': n_genes_values,
                'test_error': metric_values,
                'roc_auc': auc_values,
            })

            temp_df = pd.concat([temp_df, feature_count_values], axis=1)

            # Append to all_results
            all_results = pd.concat([all_results, temp_df], ignore_index=True)
    
    if quantize_gene_counts:
        quant_level = int(np.ceil(
            (all_results['n_genes'].max() - all_results['n_genes'].min()) / n_gene_count_bins
        ))
        all_results['n_genes'] = np.round(all_results['n_genes'] / quant_level) * quant_level


    aggregated_results = (
        all_results
        .groupby(['model', 'method', 'n_genes'], dropna=False)
        .agg(
            mean_metric=('test_error', 'mean'),
            sd_metric=('test_error', 'std'),
            mean_auc=('roc_auc', 'mean'),
            sd_auc=('roc_auc', 'std'),
            **{
                col: ('mean', col)
                for col in all_results.columns if col.startswith('feature')
            }
        ).reset_index() 
    )

    # Define color groups based on method types
    aggregated_results['methodModel'] = aggregated_results.apply(
        lambda row: "Lasso" if row['method'] == "Lasso" else
                   row['method'] if row['method'] in baseline_names else
                    f"{row['method']} - {row['model']}",
        axis=1
    )
   # Assign specific colors to each method group
    if methods is None:
        methods = aggregated_results['methodModel'].unique()

    our_methods  = [method for method in methods if re.match(r"^1/imp", method)]
    lasso_method = ["Lasso"] if "Lasso" in methods else []
    baseline_methods = baseline_names

    methods = methods[:] + list(baseline_names)
    colors = {}
    bolded = {}
    for (i, method) in enumerate(our_methods):
        colors[method] = LLM_LASSO_COLORS[i]
        bolded[method] = False
    for (i, method) in enumerate(lasso_method):
        colors[method] = LASSO_COLOR[i]
        bolded[method] = False
    for (i, method) in enumerate(baseline_methods):
        colors[method] = BASELINE_COLORS[i]
        bolded[method] = False
    
    for met in bolded_methods:
        bolded[met] = True
        
    
    # Filter the DataFrame for the desired methods
    filtered_data = aggregated_results[aggregated_results['methodModel'].isin(methods)]

    metrics = [
        ("mean_metric", "sd_metric", "Test Error"),
        ("mean_auc", "sd_auc", "ROC AUC")
    ]
    if regression:
        metrics = metrics[:1]

    for (mean, sd, label) in metrics:
        plt.figure(figsize=(10, 8))
        for (i, method) in enumerate(reversed(methods)):
            color = colors[method]
            data = filtered_data.where(filtered_data["methodModel"] == method)
            plt.plot(
                data["n_genes"], data[mean],
                "-D" if bolded[method] else '-o',
                linewidth=3 if bolded[method] else 2, color=color,
                markersize=8 if bolded[method] else 6,
                label=method
            )

            if plot_error_bars:
                error_bars(
                    x=data["n_genes"],
                    upper=data[mean] + data[sd],
                    lower=data[mean] - data[sd],
                    color=color,
                    width=0,
                    x_offset=i*0.005
                )
            plt.grid(True, alpha=0.5)
        plt.ylabel(label, fontdict={"size": 16})
        plt.xlabel("Number of Features", fontdict={"size": 16}) 
        plt.legend(fontsize=14, bbox_to_anchor=(1.02, 0.5), loc="center left")
        plt.tick_params(axis='both', labelsize=12)  # Change font size for both x and y axes
        plt.title(f"LLM-LASSO Performance across {n_splits} Splits", fontdict={"size": 18})
    return all_results, best_models


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



def count_feature_usage(model, multinomial, n_features, eps=1e-10):
    """
    Counts the number of times each feature is used in a multinomial glmnet model across all lambdas.

    Parameters:
    - model: Output of cv_grpnet from adelie

    Returns:
    - A DataFrame where each row corresponds to a lambda value and each column indicates
      whether a feature is nonzero (True/False).
    """
    # Get the number of features (exclude intercept)
    if multinomial:
        feature_inclusion_matrix =  np.abs(model.betas.toarray().reshape((model.betas.shape[0], -1, n_features))).mean(axis=1) > eps
        sign_mtx = np.argmax(model.betas.toarray().reshape((model.betas.shape[0], -1, n_features)), axis=1)
    else:
        feature_inclusion_matrix = np.abs(model.betas.toarray()) > eps
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
    score_type = "pf",
    pfmax = [],
    folds_cv = 5,
    seed=0,
    epsilon=1e-16,
    n_threads=4,
    alpha=1,
    max_imp_pow=5
):
    """
    LLM-lasso-cv
    Creates LLM-lasso model and chooses the optimal i for 1/imp^i and optimal gamma (threshold) for ReLU
    
    Does multinomial/binomial classification.
    For suvival analysis, metric is deviance. Otherwise, metric is test error.
    
    Parameters:
    - `xall`: predictors
    - `yall`: target
    - `pf`: penalty factors or importance scores
    - `score_type`: "pf" if penalty factors, "imp" if importance scores
    - `pfmax`: the maximum penalty factor. If not provided, will estimate the optimal pfmax such that the least important feature is always 0.  
    - `folds_cv`: number of folds in CV. default is 5 
    - `seed`: seed
    """

    multinomial = not regression and len(y_train.unique()) > 2

    np.random.seed(seed)

    if score_type == "imp":
        importances = score
    elif score_type == "pf":
        importances = 1 / score
    else:
        assert False, "score_type must be either imp or pf"

    x_train_scaled = scale_cols(x_train)
    x_test_scaled = scale_cols(x_test, center=x_train.mean(axis=0), scale=x_train.std(axis=0))

    # pfs
    pf_list = [np.ones(x_train.shape[1])]
    pf_type = ["Lasso"]

    for i in range(1, max_imp_pow+1):
        pf_list.append(1 / importances ** i)
        pf_type.append(f"1/imp^{i}")

    score_names = ["Lasso", "1/imp"]

    metric = [None for _ in score_names]
    non_zero_coefs = [None for _ in score_names]
    feature_count = [None for _ in score_names]
    aucs = [None for _ in score_names]

    ref_cvm = None
    ref_nonzero = None

    if multinomial:
        one_hot_encoder = OneHotEncoder(sparse_output=False)  # Use dense array
        y_one_hot = one_hot_encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))
        glm_train = ad.glm.multinomial(y=y_one_hot, dtype=np.float64)
    elif not regression:
        glm_train = ad.glm.binomial(y=y_train.to_numpy(), dtype=np.float64)
    else:
        glm_train = ad.glm.gaussian(y=y_train.to_numpy(), dtype=np.float64)

    best_models = []
    for (score_index, score_name) in enumerate(score_names):
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
                print("THERE WAS ERROR")
                print(e)
                continue

            cvm = fit.test_error

            non_zero = [
                np.count_nonzero(
                    np.mean(np.abs(clss), axis=0) > epsilon
                )
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
            metric_raw = test_error_mse(etas, y)
        else:
            metric_raw = test_error_hamming(etas, y, multinomial)

        if not regression:
            roc_auc_raw = auc_roc(etas, y, multinomial)
        else:
            roc_auc_raw = np.zeros_like(metric_raw)

        if multinomial:
            non_zero_coefs_raw = [
                np.count_nonzero(np.mean(np.abs(coeffs), axis=0) > epsilon)
                for coeffs in model.betas.toarray().reshape((model.betas.shape[0], -1, x_test.shape[1]))
            ]
        else:
            non_zero_coefs_raw = [
                np.count_nonzero(np.abs(coeffs) > epsilon)
                    for coeffs in model.betas.toarray()
            ]

        (feature_count_raw, signs_raw) = count_feature_usage(model,  multinomial, x_train.shape[1], eps=epsilon)

        df = pd.DataFrame({
            'non_zero_coefs': non_zero_coefs_raw,
            'metric': metric_raw,
            "roc_auc": roc_auc_raw
        })
        df = pd.concat((df, feature_count_raw, signs_raw), axis=1)

        best_models.append(best_model)

        # Group by 'non_zero_coefs' and filter rows where 'metric' is the minimum for each group
        df = (
            df.loc[df.groupby('non_zero_coefs')['metric'].idxmin()]
            .reset_index(drop=True)
        )

        ## give it a downstream l2 model
        # counts = df[[f"Feature_{i+1}" for i in range(x_train.shape[1])]].to_numpy()
        # if not multinomial:
        #     test_errors_this_model = []
        #     roc_aucs_this_model = []
            
        #     for row in counts:
        #         idxs = np.where(row)[0]
        #         features = [x_train.columns[idx] for idx in idxs]

        #         if len(features) == 0:
        #             test_errors_this_model.append(metric_raw[0])
        #             roc_aucs_this_model.append(roc_auc_raw[0])
        #             continue
        #         x_train_subset = x_train[features]
        #         x_test_subset = x_test[features]
        #         # penalty = (best_model_pf / np.sum(best_model_pf) * x_train_scaled.shape[1])[list(idxs)]
                
        #         model = perform_logistic_regression(
        #             x_train_subset.to_numpy(),
        #             y_train.to_numpy(),
        #             nfolds=folds_cv, seed=seed,
        #             n_threads=n_threads,
        #         )
            
        #         preds = model.predict_proba(x_test_subset.to_numpy())
        #         test_errors_this_model.append(1 - accuracy_score(
        #             y_test.to_numpy(), np.argmax(preds, axis=1)))
        #         # print(len(features), test_errors_this_model[-1])
        #         roc_aucs_this_model.append(roc_auc_score(
        #             y_test, preds[:, 1] if preds.shape[1] == 2 else preds, multi_class='ovr'))
                
        #     df['metric'] = test_errors_this_model
        #     df["roc_auc"] = roc_aucs_this_model
        # else:
        #     # TODO
        #     pass

        metric[score_index] = df['metric']
        non_zero_coefs[score_index] = df['non_zero_coefs']
        aucs[score_index] = df["roc_auc"]

        feature_count[score_index] = df.drop(columns=['metric', 'non_zero_coefs', 'roc_auc'])

    return {
        "score_names": score_names,
        "test_error": metric,
        "non_zero_coefs": non_zero_coefs,
        "feature_count": feature_count,
        "roc_auc":  aucs,
        "best_model": best_models
    }


def run_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    regression=False,
    max_features = None,
    feature_baseline = None,
    folds_cv = 5,
    seed = 0,
    n_threads = 4,
    epsilon=1e-16
):
    """
    Runs baseline model
  
    Does multinomial classification.
    
    Parameters:
    - `xall`: predictors
    - `yall`: target
    - `genenames`: gene names
    - `feature_baseline`: list of feature dataframes selected in baseline models
    - `folds_cv`: number of folds in CV. default is 5 
    - `seed`: seed. default is 0
    """
    np.random.seed(seed)

    model_names = feature_baseline.keys()

    x_train_scaled = scale_cols(x_train)
    x_test_scaled = scale_cols(x_test, center=x_train.mean(axis=0), scale=x_train.std(axis=0))

    test_errors = [None for _ in model_names]
    roc_aucs = [None for _ in model_names]
    non_zero_coefs = [None for _ in model_names]
    feature_count = [None for _ in model_names]
    

    for (model_index, model_name) in enumerate(model_names):
        feature_list = feature_baseline[model_name]

        test_errors_this_model = []
        roc_aucs_this_model = []
        non_zero_coefs_this_model = []

        n_features = len(feature_list) if max_features is None else min(max_features+1, len(feature_list))
        step = int(np.ceil(n_features / 20))
        for i in np.arange(1, n_features, step):
            i = int(i)
            features = feature_list[:i]
            x_subset_train = x_train_scaled[features]
            x_subset_test = x_test_scaled[features]

            # CV to get l2 penalty
            if not regression:
                model = perform_logistic_regression(
                    x_subset_train.to_numpy(),
                    y_train.to_numpy(),
                    nfolds=folds_cv, seed=seed,
                    n_threads=n_threads
                )
            
                preds = model.predict_proba(x_subset_test.to_numpy())
                test_errors_this_model.append(1 - accuracy_score(
                    y_test.to_numpy(), np.argmax(preds, axis=1)))
                roc_aucs_this_model.append(roc_auc_score(
                    y_test, preds[:, 1] if preds.shape[1] == 2 else preds, multi_class='ovr'))
                
                if not len(model.coef_.shape) > 1:
                    n_nonzero = np.count_nonzero(np.mean(np.abs(model.coef_), axis=1) > epsilon)
                else:
                    n_nonzero = np.count_nonzero(np.abs(model.coef_) > epsilon)
                non_zero_coefs_this_model.append(n_nonzero)
            else:
                model = LinearRegression(
                    n_jobs=n_threads,
                ).fit(x_subset_train.to_numpy(), y_train.to_numpy())
                preds = model.predict(x_subset_test.to_numpy())
                test_errors_this_model.append(mean_squared_error(
                    y_test.to_numpy(), preds
                ))
                roc_aucs_this_model.append(np.zeros_like(test_errors_this_model[-1]))
                non_zero_coefs_this_model.append(np.count_nonzero(np.abs(model.coef_) > epsilon))

        test_errors[model_index] = np.array(test_errors_this_model)
        roc_aucs[model_index] = np.array(roc_aucs_this_model)
        non_zero_coefs[model_index] = np.array(non_zero_coefs_this_model)

    return {
        "model_names": model_names,
        "test_error": test_errors,
        "non_zero_coefs": non_zero_coefs,
        "feature_count": feature_count,
        "roc_auc": roc_aucs
    }
    

# Function to perform logistic regression using custom balanced folds
def perform_logistic_regression(x, y, alpha_list=None, nfolds=7, seed=0, n_threads=4, penalty="l2"):
    if alpha_list is None:
        alpha_list = [0.1, 0.5, 1, 5, 10, 50, 100]

    log_reg = LogisticRegressionCV(
        Cs=alpha_list,
        multi_class='multinomial',
        penalty=penalty,
        random_state=seed,
        n_jobs=n_threads,
        scoring="accuracy",
        refit=True,
        cv=nfolds
    ).fit(x, y)

    return log_reg

# Function to plot cross-validation graph
def plot_cross_validation(log_reg, x):
    errors = []
    num_features = []

    for i, coef in enumerate(log_reg.coefs_paths_[np.unique(log_reg.classes_)[0]]):
        selected_features = np.sum(coef != 0, axis=1)
        misclassification_rates = 1 - np.mean(log_reg.scores_[np.unique(log_reg.classes_)[0]][:, i])

        num_features.append(selected_features)
        errors.append(misclassification_rates)

    plt.figure(figsize=(10, 6))
    plt.plot(num_features, errors, marker='o')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Misclassification Rate')
    plt.title('Cross-Validation: Features Selected vs. Error Rate')
    plt.grid(True)
    plt.show()

# Function to evaluate the model on test data using misclassification rate
def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    misclassification_rate = 1 - accuracy_score(y_test, predictions)
    print("Test Misclassification Rate:", misclassification_rate)
    return misclassification_rate
