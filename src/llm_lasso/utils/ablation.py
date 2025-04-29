"""
Implements the abalation experiment pipeline.
"""
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from llm_lasso.task_specific_lasso.llm_lasso import run_repeated_llm_lasso_cv, PenaltyType
from llm_lasso.llm_penalty.llm import LLMQueryWrapperWithMemory
from llm_lasso.data_splits import read_train_test_splits
from llm_lasso.llm_penalty.penalty_collection import collect_penalties, PenaltyCollectionParams
sys.path.insert(0, '..')
import constants

# define plotting color schemes
LASSO_COLOR = ["#999999"]
LLM_LASSO_COLORS = ["#D55E00", "#CC79A7", "#E69F00"]
N_SPLITS = 10

import os
import numpy as np
from tqdm import tqdm

def temp_scores(temp_ls, category, feature_names, prompt_filename, vectorstore, save_dir="ablation_scores/", model='gpt-4o'):
    """
    Score collection for ablating on temperature, with and without RAG.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---------- Without RAG ----------
    for temp in tqdm(temp_ls, desc="Scoring (No RAG)"):
        _, all_scores = collect_penalties(
            category=category,
            feature_names=feature_names,
            prompt_file=prompt_filename,
            save_dir=save_dir,
            vectorstore=vectorstore,
            model=LLMQueryWrapperWithMemory(llm_name=model, temperature=temp),
            params=PenaltyCollectionParams(),  # no RAG
            omim_api_key=constants.OMIM_KEYS[0],
            n_threads=10,
            parallel=True
        )
        sub_dir = os.path.join(save_dir, "plain")
        os.makedirs(sub_dir, exist_ok=True)
        filename = f"temp_{temp}.txt"
        full_path = os.path.join(sub_dir, filename)
        np.savetxt(full_path, all_scores)

    # ---------- With RAG ----------
    for temp in tqdm(temp_ls, desc="Scoring (With RAG)"):
        _, all_scores = collect_penalties(
            category=category,
            feature_names=feature_names,
            prompt_file=prompt_filename,
            save_dir=save_dir,
            vectorstore=vectorstore,
            model=LLMQueryWrapperWithMemory(llm_name=model, temperature=temp),
            params=PenaltyCollectionParams(omim_rag=True),  # with RAG
            omim_api_key=constants.OMIM_KEYS[0],
            n_threads=10,
            parallel=True
        )
        sub_dir = os.path.join(save_dir, "rag")
        os.makedirs(sub_dir, exist_ok=True)
        filename = f"temp_{temp}.txt"
        full_path = os.path.join(sub_dir, filename)
        np.savetxt(full_path, all_scores)

def create_score_dict(save_dir, methods, mode="rag"):
    """
    Loads scores from 'rag' or 'plain' subfolder under save_dir.
    
    Args:
        save_dir (str): Base directory containing 'rag' and 'plain' subdirs.
        methods (list): List of method names or temperature values (as in `temp_ls`).
        mode (str): Either 'rag' or 'plain'.
    
    Returns:
        dict: {method: loaded_score_array}
    """
    assert mode in ["rag", "plain"], "Mode must be 'rag' or 'plain'"
    sub_dir = os.path.join(save_dir, mode)
    
    score_dict = {}
    for method in methods:
        filename = f"{method}.txt"
        full_path = os.path.join(sub_dir, filename)

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Expected score file not found: {full_path}")

        scores = np.loadtxt(full_path)
        score_dict[method] = scores

    return score_dict
    
def get_comp_df(importance_scores_dict, train_test_filename):
    """
    Getting a pandas dataframe for LLM-Lasso performance on a dataset using penalty factors obtained 
    from a selection of methods.
    """

    # TODO @nsagan: fix this for new experimental framework
    raise NotImplementedError()
    splits = read_train_test_splits(train_test_filename, 10)
    method_names = list(importance_scores_dict.keys())
    methods = [f"1/imp - {name}" for name in method_names] + ["Lasso"]
    res_df = run_repeated_llm_lasso_cv(
        x_train_splits=x_train, x_test_splits=x_test,
        y_train_splits=y_train, y_test_splits=y_test,
        scores=importance_scores_dict,
        feature_baseline={},
        n_splits=N_SPLITS,
        folds_cv=10,
        score_type = PenaltyType.PF,
        lambda_min_ratio=0.001,
        n_threads=8,
        regression = False
    )
    return res_df

def get_means_at_pt(df: pd.DataFrame, models, methods, n_features=4, metric="test_error"):
    means = {}
    stds = {}
    for (model, method) in zip(models, methods):
        x = df[(df["model"] == model) & (df["method"] == method)]
        subset = None
        for i in reversed(range(1,n_features + 1)):  
            if i in x["n_features"].to_list():
                subset = x[x["n_features"] == i]
                break
        # print(model, method, subset)
        means[(model, method)] = subset[metric].mean()
        if metric == "roc_auc":
            means[(model, method)] = 1 - means[(model, method)]
        stds[(model, method)] = subset[metric].std()
    
    return means, stds

def plot_ablation(importance_scores_dict, comp_df, metric_type, metric_label, title):
    method_names = list(importance_scores_dict.keys())
    means0, stds0 = get_means_at_pt(
        comp_df, method_names, ["1/imp"] * len(method_names) + ["Lasso"],
        4, metric_type
    )
    means = {name: means0[(name, "1/imp")] for name in method_names}

    stds = {name: stds0[(name, "1/imp")] for name in method_names}


    # Extract feature names, mean values, and standard deviations
    features = list(means.keys())

    mean_values = np.array([means[f] for f in features])
    std_values = np.array([stds[f] for f in features])

    argsort = list(np.argsort(mean_values))
    mean_values = mean_values[argsort]
    std_values = std_values[argsort]
    features = [features[i] for i in argsort]

    # Use a colorblind-friendly palette (tableau-colorblind10)
    colors = LLM_LASSO_COLORS[:len(features)-1] + LASSO_COLOR
    colors = [colors[i] for i in argsort]

    # Create the bar plot with unique colors
    plt.figure(figsize=(16, 6))
    bars = plt.bar(
        features, mean_values, yerr=std_values,
        capsize=5, color=colors, edgecolor='black',
        alpha=0.8
    )

    # Labels and title
    # plt.xlabel('Models', fontdict={"size": 22})
    plt.ylabel(metric_label,  fontdict={"size": 22})
    plt.title(title,  fontdict={"size": 26})
    plt.xticks(rotation=30)
    plt.tick_params(labelsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)