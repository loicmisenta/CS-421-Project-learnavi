from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # plotting helpers fail only if called
    plt = None

try:
    from scipy import linalg
    from scipy.sparse.csgraph import laplacian
except ModuleNotFoundError:  # spectral helpers fail only if called
    linalg = None
    laplacian = None

try:
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.manifold import spectral_embedding
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.metrics.pairwise import pairwise_kernels
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:  # clustering helpers fail only if called
    KMeans = None
    SimpleImputer = None
    spectral_embedding = None
    adjusted_rand_score = None
    silhouette_score = None
    pairwise_kernels = None
    train_test_split = None
    Pipeline = None
    StandardScaler = None


RANDOM_STATE = 42


def require_clustering_deps():
    if Pipeline is None:
        raise ModuleNotFoundError("This helper requires scikit-learn.")


def require_spectral_deps():
    require_clustering_deps()
    if linalg is None or laplacian is None:
        raise ModuleNotFoundError("This helper requires scipy.")


def preprocessing_pipeline():
    require_clustering_deps()

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def preprocess_features(df, feature_cols):
    pipe = preprocessing_pipeline()
    X = pipe.fit_transform(df[feature_cols])
    return X, pipe


def build_trajectory_sequences(
    df,
    feature_cols,
    sequence_length,
    user_col="user_id",
    time_col="relative_week",
    label_col="is_dropout_point",
):
    df = df.sort_values([user_col, time_col]).dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)

    X_weekly, weekly_preprocess = preprocess_features(df, feature_cols)
    y_all = df[label_col].to_numpy()
    weeks_all = df[time_col].to_numpy()
    row_index_all = df.index.to_numpy()

    sequences = []
    labels = []
    metadata = []

    for user_id, user_positions in df.groupby(user_col).indices.items():
        user_positions = np.asarray(user_positions)
        if len(user_positions) < sequence_length:
            continue

        for end_pos in range(sequence_length - 1, len(user_positions)):
            seq_positions = user_positions[
                end_pos - sequence_length + 1 : end_pos + 1
            ]
            end_position = user_positions[end_pos]
            sequences.append(X_weekly[seq_positions])
            labels.append(int(y_all[end_position]))
            metadata.append(
                {
                    user_col: user_id,
                    "end_week": weeks_all[end_position],
                    "row_index": row_index_all[end_position],
                    label_col: int(y_all[end_position]),
                }
            )

    return (
        np.asarray(sequences),
        np.asarray(labels),
        pd.DataFrame(metadata),
        weekly_preprocess,
    )


def stratified_subsample(
    X_seq,
    y_seq,
    metadata,
    sample_size,
    random_state=RANDOM_STATE,
):
    require_clustering_deps()

    n_seq = X_seq.shape[0]
    n_sample = min(sample_size, n_seq)

    if n_sample < n_seq:
        sample_idx, _ = train_test_split(
            np.arange(n_seq),
            train_size=n_sample,
            stratify=y_seq,
            random_state=random_state,
        )
    else:
        sample_idx = np.arange(n_seq)

    sample_idx = np.sort(sample_idx)
    return (
        X_seq[sample_idx],
        y_seq[sample_idx],
        metadata.iloc[sample_idx].reset_index(drop=True).copy(),
    )


def rbf_similarity(X, base_gamma=1.0):
    require_clustering_deps()

    gamma = base_gamma / max(X.shape[1], 1)
    return pairwise_kernels(X, metric="rbf", gamma=gamma)


def adjacency_from_similarity(S, connectivity="full"):
    if connectivity == "full":
        return S
    if connectivity == "epsilon":
        return np.where(S > 0.5, 1, 0)
    raise ValueError(f"Unknown connectivity: {connectivity}")


def per_sequence_l2_normalize(X_seq):
    norms = np.linalg.norm(X_seq, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X_seq / norms


def run_kmeans_clustering(X, n_clusters, random_state=RANDOM_STATE, n_init=10):
    require_clustering_deps()

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(X)
    return labels, model


def run_spectral_clustering(W, n_clusters, random_state=RANDOM_STATE):
    require_spectral_deps()

    L = laplacian(W, normed=True)
    eigenvals, _ = linalg.eig(L)
    eigenvals = np.real(eigenvals)
    eigenvals_sorted = eigenvals[np.argsort(eigenvals)]

    rs = np.random.RandomState(random_state)
    embedding = spectral_embedding(
        W,
        n_components=n_clusters,
        random_state=rs,
        drop_first=False,
    )
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(embedding)
    return labels, embedding, eigenvals_sorted


def spectral_k_diagnostics(W, k_values, random_state=RANDOM_STATE):
    rows = []
    labels_by_k = {}
    last_eigenvals = None

    for k in k_values:
        labels, embedding, eigenvals = run_spectral_clustering(
            W,
            k,
            random_state=random_state,
        )
        labels_by_k[k] = labels
        last_eigenvals = eigenvals
        rows.append({"k": k, "silhouette": silhouette_score(embedding, labels)})

    return labels_by_k, pd.DataFrame(rows), last_eigenvals


def plot_k_diagnostics(diagnostics, eigenvals, title):
    if plt is None:
        raise ModuleNotFoundError("plot_k_diagnostics requires matplotlib.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(diagnostics["k"], diagnostics["silhouette"], "-o", color="#2563eb")
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("Silhouette")
    axes[0].set_title("Silhouette")
    axes[0].set_xticks(diagnostics["k"])

    n_show = max(len(diagnostics) * 2, 6)
    shown = eigenvals[:n_show]
    axes[1].scatter(range(1, len(shown) + 1), shown, color="#dc2626")
    axes[1].set_xlabel("Eigenvalue index")
    axes[1].set_ylabel("Eigenvalue")
    axes[1].set_title("Eigengap")
    axes[1].set_xticks(range(1, len(shown) + 1))

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def trajectory_overview(
    traj_df,
    cluster_col,
    user_col="user_id",
    label_col="is_dropout_point",
):
    overview = (
        traj_df.groupby(cluster_col, observed=False)
        .agg(
            n_sequences=(user_col, "count"),
            n_users=(user_col, "nunique"),
            dropout_point_rate=(label_col, "mean"),
        )
        .reset_index()
        .sort_values(cluster_col)
    )
    overview["share_sequences"] = (
        overview["n_sequences"] / overview["n_sequences"].sum()
    )
    return overview


def method_adjusted_rand_table(df, methods):
    require_clustering_deps()

    rows = []
    for i, left in enumerate(methods):
        for right in methods[i + 1 :]:
            rows.append(
                {
                    "method_a": left,
                    "method_b": right,
                    "adjusted_rand": adjusted_rand_score(df[left], df[right]),
                }
            )
    return pd.DataFrame(rows).sort_values("adjusted_rand", ascending=False)


def trajectory_dimension_curves(X_sequences, feature_cols, feature_groups):
    curves = {}
    for group_name, cols in feature_groups.items():
        indices = [feature_cols.index(c) for c in cols if c in feature_cols]
        if indices:
            curves[group_name] = X_sequences[:, :, indices].mean(axis=2)
    return curves


def plot_trajectory_profiles(X_seq, labels, feature_cols, feature_groups, title):
    if plt is None:
        raise ModuleNotFoundError("plot_trajectory_profiles requires matplotlib.")

    curves = trajectory_dimension_curves(X_seq, feature_cols, feature_groups)
    if not curves:
        return

    fig, axes = plt.subplots(
        1,
        len(curves),
        figsize=(4 * len(curves), 4),
        sharey=False,
    )
    if len(curves) == 1:
        axes = [axes]

    for ax, (dimension, values) in zip(axes, curves.items()):
        for cluster_id in sorted(np.unique(labels)):
            mask = labels == cluster_id
            ax.plot(values[mask].mean(axis=0), marker="o", label=f"C{cluster_id}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(dimension)
        ax.set_xlabel("Week")

    axes[0].set_ylabel("Avg standardized score")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
