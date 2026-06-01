from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:  # plotting helpers fail only if called
    plt = None
    sns = None


def one_hot_label(df, prefix, default="Unknown"):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.Series(default, index=df.index)

    labels = df[cols].idxmax(axis=1).str.replace(prefix, "", regex=False)
    empty_rows = df[cols].sum(axis=1) == 0
    return labels.mask(empty_rows, default)


def education_level_from_class_year(class_year):
    if pd.isna(class_year) or class_year == 0:
        return "Unknown"
    if class_year <= 2:
        return "Lower secondary"
    if class_year <= 4:
        return "Upper secondary"
    return "Other"


def school_type_from_school(school):
    school = str(school).lower()

    if school in {"unknown", "nan", "none"}:
        return "Unknown"
    if any(token in school for token in ["gym", "passerelle"]):
        return "Academic"
    if any(token in school for token in ["sek", "secondary"]):
        return "Secondary"
    if any(token in school for token in ["bm", "efz", "wms"]):
        return "Vocational"
    return "Other"


def extract_stage1_demographics(df):
    demographics = df[["user_id"]].copy()
    demographics["gender"] = one_hot_label(df, "gender_")
    demographics["canton"] = one_hot_label(df, "canton_")
    demographics["school"] = one_hot_label(df, "school_")
    demographics["class_year"] = df["class_year"] if "class_year" in df else np.nan
    demographics["education_level"] = demographics["class_year"].apply(
        education_level_from_class_year
    )
    demographics["school_type"] = demographics["school"].apply(school_type_from_school)
    return demographics


def outcome_by_cluster_and_group(
    df,
    group_col,
    min_n=30,
    cluster_col="spectral_eucl",
    outcome_col="is_dropout_point",
    user_col="user_id",
):
    table = (
        df.groupby([cluster_col, group_col], observed=False)
        .agg(
            n_rows=(user_col, "size"),
            n_unique_users=(user_col, "nunique"),
            dropout_point_rate=(outcome_col, "mean"),
        )
        .reset_index()
    )
    table = table[table["n_rows"] >= min_n].copy()

    cluster_rates = (
        df.groupby(cluster_col, observed=False)[outcome_col]
        .mean()
        .rename("cluster_rate")
    )
    table = table.merge(cluster_rates, on=cluster_col, how="left")
    table["rate_gap_vs_cluster"] = table["dropout_point_rate"] - table["cluster_rate"]
    table["disparity_ratio_vs_cluster"] = table["dropout_point_rate"] / table[
        "cluster_rate"
    ].replace(0, np.nan)
    table["support_note"] = np.where(
        table["n_unique_users"] < min_n,
        "low unique-user support",
        "",
    )
    return table


def plot_outcome_heatmap(
    table,
    group_col,
    title,
    cluster_col="spectral_eucl",
    save_path=None,
):
    if plt is None or sns is None:
        raise ModuleNotFoundError("plot_outcome_heatmap requires matplotlib/seaborn.")

    pivot = table.pivot(
        index=cluster_col,
        columns=group_col,
        values="dropout_point_rate",
    )

    plt.figure(figsize=(max(7, 1.1 * len(pivot.columns)), 4.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Dropout-point rate"},
    )
    plt.title(title)
    plt.xlabel(group_col)
    plt.ylabel("Spectral Euclidean cluster")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return pivot
