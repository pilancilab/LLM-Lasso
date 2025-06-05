import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from llm_lasso.task_specific_lasso.gene_usage import get_top_features_for_method


def plot_format_generator(colors: list[str]):
    marker_styles = ["o", "v", "s", "X", "d"]
    idx = 0

    while True:
        if idx >= len(marker_styles) * len(colors):
            idx = 0
        yield colors[idx % len(colors)], marker_styles[idx // len(colors)]
        idx += 1


LASSO_COLOR = ["black"]
BASELINE_COLORS = ["#56B4E9", "#009292", "#117733", "#490092", "#924900"]
LLM_LASSO_COLORS = ["#DC267F", "#FE6100", "#FFB000"]


def plot_heatmap(
    dataframes: list[pd.DataFrame],
    method_models: list[str],
    labels: list[str],
    feature_names: list[str],
    sort_by="RAG", top=10, task="",
    pos_marker="+", neg_marker="-"
):
    """
    Plots a feature usage heatmap for binary classification or regression.

    Parameters:
    - `res`: `DataFrame` output by `run_repeated_llm_lasso_cv`.
    - `method_models`: which values of the `method_model` column of the 
        dataframe should be plot, e.g., `["Lasso", "RAG - 1/imp"]`.
    - `labels`: x-axis label for each value of `method_models`, e.g.,
        in the above example, `["Lasso", "RAG"]`.
    - `feature_names`: list of feature names, in the same order as the
        dataframe inputs to `run_repeated_llm_lasso_cv`
    - `sort_by`: which value of `labels` to sort the features by when plotting.
    - `top`: how many features to select for each `method_model`.
    - `task`: optional task name to add to the title.
    - `pos_marker`: label for features with positive contribution.
    - `neg_marker`: label for features with negative contribution.
    """

    res = pd.concat(dataframes, ignore_index=True).copy()
    raw_top_genes = [
        get_top_features_for_method(res, method_model, feature_names, top=top)
            for method_model in method_models
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
    title = "Feature Usage Across Models"
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
    dataframes: list[pd.DataFrame],
    quantize_gene_counts = False,
    n_gene_count_bins=20,
    bolded_methods=[],
    plot_error_bars = True,
    x_lim=None,
    test_error_y_lim=None,
    auroc_y_lim=None,
    task=None,
    small_plot=False
):
    """
    Plot result of LLM-Lasso alongside baselines.

    Parameters:
    - `all_results`: output dataframe from `run_repeated_llm_lasso_cv`
    - `quantize_gene_counts`: whether to quantize the x-axis (number of features).
    - `n_gene_count_bins`: quantization granularity if `quantize_gene_counts`
        is True. 
    - `bolded_methods`: list of methods to bold. For LLM-Lasso methods, in the format
        "model_name - method_name", e.g., "RAG - 1/imp", following the "method_model"
        column of the `all_results` dataframe.
    - `plot_error_bars`: whether to plot standard deviation error bars.
    """
    

    all_results = pd.concat(dataframes, ignore_index=True).copy()

    our_method_format = plot_format_generator(LLM_LASSO_COLORS)
    baseline_format = plot_format_generator(BASELINE_COLORS)
    
    n_splits = all_results["split"].max() + 1
    if x_lim is None:
        x_lim = all_results["n_features"].max()

    # Fill in points where the model does not select that number of features
    # (e.g., when Lasso refuses to select more than a certain number of
    # features, depending on the penalty factors)

    for method_model in all_results["method_model"].unique():
        for split in range(n_splits):
            prev_row = None
            for nfeat in range(x_lim + 1):
                row = all_results[
                    np.bitwise_and(
                        all_results["method_model"] == method_model,
                        np.bitwise_and(
                            all_results["split"] == split,
                            all_results["n_features"] == nfeat))
                    ]
                if row.shape[0] == 1:
                    prev_row = row.copy()
                elif row.shape[0] == 0:
                    if prev_row is not None:
                        prev_row["n_features"] = nfeat
                        all_results = pd.concat([all_results, prev_row], ignore_index=True)
    all_results = all_results.copy()

    # Infer some properties from the dataframe
    baseline_names = all_results[all_results["is_baseline"] == True]["method_model"].unique()
    regression = all(all_results["auroc"].isnull())
    if quantize_gene_counts:
        quant_level = int(np.ceil(
            (all_results['n_features'].max() - all_results['n_features'].min()) / n_gene_count_bins
        ))
        all_results['n_features'] = np.round(all_results['n_features'] / quant_level) * quant_level

    # Get mean and standard deviation of the metrics at each point
    aggregated_results = (
        all_results
        .groupby(['method_model', 'n_features'], dropna=False)
        .agg(
            mean_metric=('test_error', 'mean'),
            sd_metric=('test_error', 'std'),
            qlow_metric=('test_error', lambda x: x.quantile(0.1)),
            qhigh_metric=('test_error', lambda x: x.quantile(0.9)),
            mean_auc=('auroc', 'mean'),
            qlow_auc=('auroc', lambda x: x.quantile(0.1)),
            qhigh_auc=('auroc', lambda x: x.quantile(0.9)),
            sd_auc=('auroc', 'std'),
            **{
                col: ('mean', col)
                for col in all_results.columns if col.startswith('feature')
            }
        ).reset_index()
    )
   # Assign specific colors to each method group
    methods = aggregated_results['method_model'].unique().tolist()
    lasso_method = ["Lasso"] if "Lasso" in methods else []
    baseline_methods = baseline_names

    # Anything that's not Lasso or a baseline defaults to being in "our_methods"
    our_methods = [method for method in methods if \
                   method not in lasso_method and method not in baseline_methods]
    if lasso_method[0] in methods:
        methods.remove(lasso_method[0])
    for method in our_methods:
        methods.remove(method)
    
    methods = methods + lasso_method + our_methods

    color_and_marker = {}
    bolded = {}
    for (i, method) in enumerate(our_methods):
        color_and_marker[method] = next(our_method_format)
        bolded[method] = False
    for (i, method) in enumerate(lasso_method):
        color_and_marker[method] = (LASSO_COLOR[0], "o")
        bolded[method] = False
    for (i, method) in enumerate(baseline_methods):
        color_and_marker[method] = next(baseline_format)
        bolded[method] = False
    
    for met in bolded_methods:
        bolded[met] = True
        
    
    # Filter the DataFrame for the desired methods
    filtered_data = aggregated_results[aggregated_results['method_model'].isin(methods)]

    metrics = [
        ("mean_metric", "sd_metric", "qlow_metric", "qhigh_metric", "Test Error"),
        ("mean_auc", "sd_auc", "qlow_auc", "qhigh_auc", "AUROC")
    ]
    if regression:
        metrics = metrics[:1] # no AUROC

    for (mean, sd, qlow, qhigh, label) in metrics:
        # fig = plt.figure(figsize=(7, 6))
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_axes([0, 0, 0.8, 1])
        for (i, method) in enumerate((methods)):
            color, marker = color_and_marker[method]
            data = filtered_data.where(filtered_data["method_model"] == method)

            xaxis_data = data["n_features"][:]
            data_mean = data[mean]
            data_sd = data[sd]
            data_qlow = data[qlow]
            data_qhigh = data[qhigh]
            if x_lim:
                idxs = [i for (i,x) in enumerate(xaxis_data) if x <= x_lim]
                xaxis_data = xaxis_data[idxs]
                # print(xaxis_data)
                data_mean = data_mean[idxs]
                data_sd = data_sd[idxs]
                data_qlow = data_qlow[idxs]
                data_qhigh = data_qhigh[idxs]

            if small_plot:
                linewidth = 4 if bolded[method] else 3
            else:
                linewidth = 3.5 if bolded[method] else 2.5

            ax.plot(
                xaxis_data, data_mean,
                "-D" if bolded[method] else f'-{marker}',
                linewidth=linewidth, color=color,
                markersize=12 if bolded[method] else 9,
                label=method
            )

            if plot_error_bars:
                error_bars(
                    x=xaxis_data,
                    upper=data_qhigh,
                    lower=data_qlow,
                    color=color,
                    width=0,
                    x_offset=i*0.005
                )
            plt.grid(True, alpha=0.7)

        if small_plot:
            plt.ylabel(label, fontdict={"size": 40})
            plt.xlabel("Number of Features", fontdict={"size": 40}) 
            if test_error_y_lim and label == "Test Error":
                plt.ylim(*test_error_y_lim)
            elif auroc_y_lim and label == "AUROC":
                plt.ylim(*auroc_y_lim)
            plt.legend(fontsize=34, bbox_to_anchor=(1.02, 0.5), loc="center left")
            # plt.tight_layout()
            plt.tick_params(axis='both', labelsize=30)  # Change font size for both x and y axes
            if task is None:
                plt.title(f"LLM-LASSO Performance ({n_splits} Splits)\n", fontdict={"size": 44})
            else:
                plt.title(f"LLM-LASSO Performance ({task}, {n_splits} Splits)\n", fontdict={"size": 44})

        else:
            plt.ylabel(label, fontdict={"size": 36})
            plt.xlabel("Number of Features", fontdict={"size": 36}) 
            if test_error_y_lim and label == "Test Error":
                plt.ylim(*test_error_y_lim)
            elif auroc_y_lim and label == "AUROC":
                plt.ylim(*auroc_y_lim)
            plt.legend(fontsize=32, bbox_to_anchor=(1.02, 0.5), loc="center left")
            # plt.tight_layout()
            plt.tick_params(axis='both', labelsize=28)  # Change font size for both x and y axes
            if task is None:
                plt.title(f"LLM-LASSO Performance ({n_splits} Splits)\n", fontdict={"size": 40})
            else:
                plt.title(f"LLM-LASSO Performance ({task}, {n_splits} Splits)\n", fontdict={"size": 40})

        
