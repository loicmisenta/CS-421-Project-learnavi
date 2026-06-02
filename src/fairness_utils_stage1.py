import pandas as pd
import numpy as np

def compute_group_metrics(df, group_col):
    """
    Compute fairness metrics for each subgroup.
    """

    results = []

    groups = sorted(df[group_col].dropna().unique())

    for g in groups:

        sub = df[df[group_col] == g]

        if len(sub) == 0:
            continue

        # confusion matrix components
        tp = ((sub["y_true"] == 1) & (sub["y_pred"] == 1)).sum()
        tn = ((sub["y_true"] == 0) & (sub["y_pred"] == 0)).sum()
        fp = ((sub["y_true"] == 0) & (sub["y_pred"] == 1)).sum()
        fn = ((sub["y_true"] == 1) & (sub["y_pred"] == 0)).sum()

        # Equalized odds metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

        # Predictive value parity metrics
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

        results.append({
            "group": g,
            "n_samples": len(sub),
            "TPR": tpr,
            "FPR": fpr,
            "PPV": ppv,
            "NPV": npv,
        })

    return pd.DataFrame(results)


def compute_max_gap(metric_series):
    """
    Maximum absolute gap between any two groups.
    """
    values = metric_series.dropna().values

    if len(values) < 2:
        return np.nan

    return np.max(values) - np.min(values)


def print_fairness_report(metrics_df, demographic_name):

    print("=" * 70)
    print(f"FAIRNESS REPORT FOR: {demographic_name}")
    print("=" * 70)

    display(metrics_df)

    tpr_gap = compute_max_gap(metrics_df["TPR"])
    fpr_gap = compute_max_gap(metrics_df["FPR"])
    ppv_gap = compute_max_gap(metrics_df["PPV"])
    npv_gap = compute_max_gap(metrics_df["NPV"])

    print("\n--- MAXIMUM GAPS ACROSS GROUPS ---")
    print(f"Equalized Odds - TPR gap : {tpr_gap:.4f}")
    print(f"Equalized Odds - FPR gap : {fpr_gap:.4f}")

    print(f"\nPredictive Value Parity - PPV gap : {ppv_gap:.4f}")
    print(f"Predictive Value Parity - NPV gap : {npv_gap:.4f}")

    return {
        "TPR_gap": tpr_gap,
        "FPR_gap": fpr_gap,
        "PPV_gap": ppv_gap,
        "NPV_gap": npv_gap,
    }