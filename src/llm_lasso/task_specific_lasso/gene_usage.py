import pandas as pd


def get_top_features_for_method(
    res: pd.DataFrame, method_model: str,
    feature_names: list[str], top=10
):
    """
    Produces a dataframe of the features that are most often chosen for the
    provided method_model, as well as whether the features are most often
    chosen with positive or negative coefficients.

    Only implemeted for binary classification or regression.

    Parameters:
    - `res`: `DataFrame` output by `run_repeated_llm_lasso_cv`.
    - `method_modes`: value of the `method_model` column of the dataframe.
    - `feature_names`: list of feature names, in the same order as the
        dataframe inputs to `run_repeated_llm_lasso_cv`
    - `top`: how many features to select for each `method_model`.
    """
    df = res[(res["method_model"] == method_model)]
    sign_df = res[(res["method_model"] == method_model)]
    renamer = {}
    sgn_renamer = {}
    for (i, gene) in enumerate(feature_names):
        renamer[f"Feature_{i+1}"] = gene
        sgn_renamer[f"Feature_Sign_{i+1}"] = gene
    df = df.rename(columns=renamer)
    sign_df = sign_df.rename(columns=sgn_renamer)

    imps = df[feature_names].mean(axis=0)
    signs = sign_df[feature_names].sum()
    feat_imp = pd.DataFrame([imps, signs], index=["imps", "sgns"]).T
    feat_imp = feat_imp.sort_values(ascending=False, by="imps", axis=0)
    return feat_imp[:top]