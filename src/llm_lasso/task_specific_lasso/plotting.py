import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from llm_lasso.task_specific_lasso.gene_usage import get_top_genes_for_method
import re


LASSO_COLOR = ["#999999"]
BASELINE_COLORS = ["#56B4E9", "#009292", "#117733", "#490092", "#924900"]
LLM_LASSO_COLORS = ["#D55E00", "#CC79A7", "#E69F00"] + BASELINE_COLORS


def plot_heatmap(res: pd.DataFrame, models: list[str], methods: list[str], labels: list[str],
                 genenames: list[str], sort_by="RAG", top=10, task="", pos_marker="+", neg_marker="-"):
    raw_top_genes = [
        get_top_genes_for_method(res, model, method, genenames, top=top)
            for (model, method) in zip(models, methods)
    ]
    genes = list(set(sum([list(x.index) for x in raw_top_genes], start=[])))
    top_genes = [
        [x["imps"][gene] if gene in x.index else np.nan for gene in genes]
            for x in raw_top_genes
    ]
    gene_signs = [
        [x["sgns"][gene] if gene in x.index else np.nan for gene in genes]
            for x in raw_top_genes
    ]
    top_genes = pd.DataFrame(top_genes, columns=genes, index=labels).T
    gene_signs = pd.DataFrame(gene_signs, columns=genes, index=labels).T
    for l in labels:
        top_genes[f"sign_{l}"] = gene_signs[l].map(lambda x: pos_marker if x > 0 else (neg_marker if x < 0 else "0"))
    top_genes = top_genes.sort_values(by=sort_by, ascending=False, axis=0)
    plt.figure(figsize=(3, 12))
    ax = sns.heatmap(
        top_genes[labels],
        annot=top_genes[[f"sign_{l}" for l in labels]],
        cmap="Reds", fmt="")
    ax.collections[0].cmap.set_bad('lightgray')

    plt.xlabel("Model", fontdict={"size": 17})
    plt.ylabel("Gene", fontdict={"size": 17})
    plt.tick_params("both", labelsize=11)
    title = "Gene Scores Across Models"
    if task:
        title += f"\n({task})"
    plt.title(title, fontdict={"size": 19})


# Function to plot error bars
def error_bars(x, upper, lower, color, width=0.02, x_offset=0):
    bar_width = width * (np.max(x) - np.min(x))
    x += x_offset * (np.max(x) - np.min(x))
    plt.vlines(x, lower, upper, colors=color)
    plt.hlines(upper, x - bar_width, x + bar_width, colors=color)
    plt.hlines(lower, x - bar_width, x + bar_width, colors=color)


def plot_llm_lasso_result(
    all_results: pd.DataFrame,
    baseline_names: list[str],
    n_splits: int,
    methods = None,
    regression = False,
    quantize_gene_counts = False,
    n_gene_count_bins=20,
    bolded_methods=[],
    plot_error_bars = True,
):
    """
    Plot result of LLM-Lasso alongside baselines.

    Parameters:
    - `all_results`: output dataframe from `run_repeated_llm_lasso_cv`,
    - `baseline_names`: names of the baseline methods (baselines use a different
        color scheme).
    - `n_splits`: number of train/test splits, used for generating the plot title. 
    - `methods`: which methods to plot. For LLM-Lasso methods, in the format
        "model_name - method_name", e.g., "RAG - 1/imp". Plots all if None.
    - `regression`: whether it was a regression problem, in which case the AUROC
        cannot be plotted.
    - `quantize_gene_counts`: whether to quantize the x-axis (number of features).
    - `n_gene_count_bins`: quantization granularity if `quantize_gene_counts`
        is True. 
    - `bolded_methods`: list of methods to bold, in the same format as `methods`.
    - `plot_error_bars`: whether to plot standard deviation error bars.
    """
    
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