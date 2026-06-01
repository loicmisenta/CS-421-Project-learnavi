from __future__ import annotations

from collections import Counter

import numpy as np


RANDOM_STATE = 42


def random_oversample_minority(X, y, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    sampled_indices = []
    for cls, count in zip(classes, counts):
        indices = np.flatnonzero(y == cls)
        extra = rng.choice(indices, size=max_count - count, replace=True)
        sampled_indices.append(np.concatenate([indices, extra]))

    sampled_indices = np.concatenate(sampled_indices)
    rng.shuffle(sampled_indices)
    return X[sampled_indices], y[sampled_indices]


def stratified_subset_indices(y, max_samples=200_000, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)

    if len(y) <= max_samples:
        return np.arange(len(y))

    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    sample_counts = np.maximum(1, np.floor(proportions * max_samples).astype(int))

    while sample_counts.sum() > max_samples:
        sample_counts[np.argmax(sample_counts)] -= 1
    while sample_counts.sum() < max_samples:
        sample_counts[np.argmin(sample_counts)] += 1

    indices = []
    for cls, n_samples in zip(classes, sample_counts):
        cls_indices = np.flatnonzero(y == cls)
        indices.append(rng.choice(cls_indices, size=n_samples, replace=False))

    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return indices


def smote_on_subset(
    X,
    y,
    max_smote_samples=200_000,
    random_state=RANDOM_STATE,
    smote_cls=None,
):
    subset_idx = stratified_subset_indices(
        y,
        max_samples=max_smote_samples,
        random_state=random_state,
    )
    X_subset = X[subset_idx]
    y_subset = y[subset_idx]
    print(f"SMOTE subset: {X_subset.shape}, {Counter(y_subset)}")

    if smote_cls is not None:
        k_neighbors = min(5, Counter(y_subset).most_common()[-1][1] - 1)
        if k_neighbors >= 1:
            return smote_cls(
                random_state=random_state,
                k_neighbors=k_neighbors,
            ).fit_resample(X_subset, y_subset)

    return random_oversample_minority(X_subset, y_subset, random_state=random_state)


def mixup_augmentation(
    X,
    y,
    alpha=0.2,
    augmentation_ratio=0.5,
    random_state=RANDOM_STATE,
):
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    n_aug = int(n_samples * augmentation_ratio)

    idx_1 = rng.integers(0, n_samples, size=n_aug)
    idx_2 = rng.integers(0, n_samples, size=n_aug)
    lambdas = rng.beta(alpha, alpha, size=n_aug).reshape(-1, 1)

    X_aug = lambdas * X[idx_1] + (1 - lambdas) * X[idx_2]
    y_mix = lambdas.ravel() * y[idx_1] + (1 - lambdas.ravel()) * y[idx_2]
    y_aug = (y_mix >= 0.5).astype(int)
    return np.vstack([X, X_aug]), np.concatenate([y, y_aug])


def continuous_noise_augmentation(
    X,
    y,
    noise_level=0.01,
    augmentation_ratio=0.5,
    min_unique_values=50,
    random_state=RANDOM_STATE,
):
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    n_aug = int(n_samples * augmentation_ratio)
    indices = rng.integers(0, n_samples, size=n_aug)

    X_aug = X[indices].copy().astype(float)
    y_aug = y[indices].copy()

    continuous_cols = []
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        unique_vals = np.unique(col[~np.isnan(col)])
        if len(unique_vals) >= min_unique_values:
            continuous_cols.append(col_idx)

    print(f"continuous-noise columns: {len(continuous_cols)} / {X.shape[1]}")

    for col_idx in continuous_cols:
        std = np.nanstd(X[:, col_idx])
        if std > 0:
            X_aug[:, col_idx] += rng.normal(
                loc=0,
                scale=noise_level * std,
                size=n_aug,
            )

    return np.vstack([X, X_aug]), np.concatenate([y, y_aug])


def masking_augmentation(
    X,
    y,
    mask_rate=0.02,
    augmentation_ratio=0.5,
    random_state=RANDOM_STATE,
):
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    n_aug = int(n_samples * augmentation_ratio)
    indices = rng.integers(0, n_samples, size=n_aug)

    X_aug = X[indices].copy()
    y_aug = y[indices].copy()
    mask = rng.random(X_aug.shape) < mask_rate
    col_medians = np.nanmedian(X, axis=0)
    X_aug[mask] = np.take(col_medians, np.where(mask)[1])
    return np.vstack([X, X_aug]), np.concatenate([y, y_aug])
