import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_top_genes_for_method(res: pd.DataFrame, model: str, method: str, genenames: list[str], top=10):
    df = res[(res["model"] == model) & (res["method"] == method)]
    sign_df = res[(res["model"] == model) & (res["method"] == method)]
    renamer = {}
    sgn_renamer = {}
    for (i, gene) in enumerate(genenames):
        renamer[f"Feature_{i+1}"] = gene
        sgn_renamer[f"Feature_Sign_{i+1}"] = gene
    df = df.rename(columns=renamer)
    sign_df = sign_df.rename(columns=sgn_renamer)

    imps = df[genenames].mean(axis=0)
    signs = sign_df[genenames].sum()
    feat_imp = pd.DataFrame([imps, signs], index=["imps", "sgns"]).T
    feat_imp = feat_imp.sort_values(ascending=False, by="imps", axis=0)
    return feat_imp[:top]


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