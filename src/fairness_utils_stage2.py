import torch
import torch.nn as nn
import numpy as np
import pandas as pd



def get_predictions_basic(model, loader, device):
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)

            logits = model(X_batch)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())

    return (
        np.array(all_probs),
        np.array(all_preds),
        np.array(all_targets)
    )


def get_predictions_demo(model, loader, device):
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, gender, canton, class_level, y_batch in loader:

            X_batch = X_batch.to(device)
            gender = gender.to(device)
            canton = canton.to(device)
            class_level = class_level.to(device)

            logits = model(X_batch, gender, canton, class_level)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())

    return (
        np.array(all_probs),
        np.array(all_preds),
        np.array(all_targets)
    )

# ============================================================
# BUILD FAIRNESS DATAFRAME
# ============================================================

def create_fairness_df(
    df_full,
    users_clean,
    feature_cols,
    sequence_length=12
):

    rows = []

    for user_id, user_df in df_full.groupby("user_id"):

        user_df = user_df.sort_values("relative_week")

        target = user_df["is_dropout_point"].values

        for i in range(sequence_length, len(user_df)):

            if np.isnan(target[i]):
                continue

            rows.append({
                "user_id": user_id,
                "true_label": int(target[i])
            })

    fairness_df = pd.DataFrame(rows)

    demo_cols = users_clean[
        ["user_id", "gender", "canton"]
    ].copy()

    fairness_df = fairness_df.merge(
        demo_cols,
        on="user_id",
        how="left"
    )

    return fairness_df


def create_fairness_df_from_users(user_ids, users_clean):

    fairness_df = pd.DataFrame({
        "user_id": user_ids
    })

    fairness_df = fairness_df.merge(
        users_clean[["user_id", "gender", "canton"]],
        on="user_id",
        how="left"
    )

    return fairness_df

# ============================================================
# FAIRNESS METRICS
# ============================================================

def compute_group_metrics(df, group_col):

    results = []

    for group_value, gdf in df.groupby(group_col):

        TP = ((gdf["y_true"] == 1) & (gdf["y_pred"] == 1)).sum()
        TN = ((gdf["y_true"] == 0) & (gdf["y_pred"] == 0)).sum()
        FP = ((gdf["y_true"] == 0) & (gdf["y_pred"] == 1)).sum()
        FN = ((gdf["y_true"] == 1) & (gdf["y_pred"] == 0)).sum()

        # Equalized odds
        TPR = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        FPR = FP / (FP + TN) if (FP + TN) > 0 else np.nan

        # Predictive value parity
        PPV = TP / (TP + FP) if (TP + FP) > 0 else np.nan
        NPV = TN / (TN + FN) if (TN + FN) > 0 else np.nan

        results.append({
            group_col: group_value,
            "n_samples": len(gdf),
            "TPR": TPR,
            "FPR": FPR,
            "PPV": PPV,
            "NPV": NPV
        })

    return pd.DataFrame(results)


def compute_parity_gaps(metrics_df, group_col):

    fairness_summary = {
        "TPR_gap": metrics_df["TPR"].max() - metrics_df["TPR"].min(),
        "FPR_gap": metrics_df["FPR"].max() - metrics_df["FPR"].min(),
        "PPV_gap": metrics_df["PPV"].max() - metrics_df["PPV"].min(),
        "NPV_gap": metrics_df["NPV"].max() - metrics_df["NPV"].min(),
    }

    print(f"\n===== FAIRNESS GAPS ({group_col}) =====")

    for k, v in fairness_summary.items():
        print(f"{k}: {v:.4f}")

    return fairness_summary
    



def create_sequences(df, feature_cols, SEQUENCE_LENGTH, drop_summer_labels=False):
    sequences = []
    labels = []
    user_ids = []

    for user_id, user_df in df.groupby("user_id"):
        user_df = user_df.sort_values("relative_week")

        data = user_df[feature_cols].values
        target = user_df["is_dropout_point"].values
        is_summer = user_df["is_summer"].values

        for i in range(SEQUENCE_LENGTH, len(user_df)):
            if np.isnan(target[i]):
                continue

            if drop_summer_labels and is_summer[i]:
                continue

            X = data[i - SEQUENCE_LENGTH:i]
            y = float(target[i])

            sequences.append(X)
            labels.append(y)
            user_ids.append(user_id)

    return np.array(sequences), np.array(labels), np.array(user_ids)