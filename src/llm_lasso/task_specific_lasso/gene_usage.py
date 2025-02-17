import pandas as pd


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