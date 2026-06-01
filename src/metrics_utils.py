from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ModuleNotFoundError:  # helpers fail only if called
    f1_score = None
    precision_score = None
    recall_score = None
    roc_auc_score = None


def require_sklearn():
    if f1_score is None:
        raise ModuleNotFoundError("This helper requires scikit-learn.")


def best_threshold_by_f1(y_true, probabilities, thresholds=None):
    require_sklearn()

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 50)

    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in thresholds:
        score = f1_score(y_true, np.asarray(probabilities) > threshold)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return best_threshold, best_f1


def binary_metrics(y_true, probabilities, threshold):
    require_sklearn()

    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "f1": f1_score(y_true, predictions),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
        "threshold": threshold,
    }


def validation_tuned_binary_metrics(y_val, val_prob, y_test, test_prob):
    threshold, val_f1 = best_threshold_by_f1(y_val, val_prob)
    metrics = binary_metrics(y_test, test_prob, threshold)
    metrics["val_f1"] = val_f1
    return metrics


def print_metrics(metrics, prefix=""):
    for key, value in metrics.items():
        print(f"{prefix}{key}: {value:.4f}")
