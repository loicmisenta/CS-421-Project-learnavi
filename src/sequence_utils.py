from __future__ import annotations

import numpy as np

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:  # helpers fail only if called
    train_test_split = None
    StandardScaler = None


RANDOM_STATE = 42


def require_sklearn():
    if train_test_split is None or StandardScaler is None:
        raise ModuleNotFoundError("This helper requires scikit-learn.")


def split_users(
    df,
    user_col="user_id",
    test_size=0.15,
    val_size=0.20,
    random_state=RANDOM_STATE,
):
    require_sklearn()

    users = df[user_col].dropna().unique()
    train_val_users, test_users = train_test_split(
        users,
        test_size=test_size,
        random_state=random_state,
    )
    train_users, val_users = train_test_split(
        train_val_users,
        test_size=val_size,
        random_state=random_state,
    )
    return train_users, val_users, test_users


def split_by_users(
    df,
    user_col="user_id",
    test_size=0.15,
    val_size=0.20,
    random_state=RANDOM_STATE,
):
    train_users, val_users, test_users = split_users(
        df,
        user_col=user_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    return (
        df[df[user_col].isin(train_users)].copy(),
        df[df[user_col].isin(val_users)].copy(),
        df[df[user_col].isin(test_users)].copy(),
    )


def split_and_scale_by_users(
    df,
    feature_cols,
    user_col="user_id",
    test_size=0.15,
    val_size=0.20,
    random_state=RANDOM_STATE,
):
    require_sklearn()

    train_df, val_df, test_df = split_by_users(
        df,
        user_col=user_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    return train_df, val_df, test_df, scaler


def pad_sequence(seq, target_length):
    if len(seq) == target_length:
        return seq
    pad_size = target_length - len(seq)
    pad_values = np.repeat(seq[0:1], pad_size, axis=0)
    return np.vstack([pad_values, seq])


def make_sequences(
    df,
    feature_cols,
    sequence_length,
    user_col="user_id",
    time_col="relative_week",
    label_col="is_dropout_point",
    summer_col="is_summer",
    drop_summer_labels=False,
    return_user_ids=False,
):
    sequences = []
    labels = []
    user_ids = []

    for user_id, user_df in df.groupby(user_col):
        user_df = user_df.sort_values(time_col)
        data = user_df[feature_cols].to_numpy(dtype=np.float32)
        target = user_df[label_col].to_numpy()
        if summer_col in user_df:
            is_summer = user_df[summer_col].fillna(False).to_numpy(dtype=bool)
        else:
            is_summer = np.zeros(len(user_df), dtype=bool)

        for idx in range(sequence_length, len(user_df)):
            if np.isnan(target[idx]):
                continue
            if drop_summer_labels and is_summer[idx]:
                continue
            sequences.append(data[idx - sequence_length : idx])
            labels.append(float(target[idx]))
            user_ids.append(user_id)

    arrays = (np.asarray(sequences), np.asarray(labels))
    if return_user_ids:
        return (*arrays, np.asarray(user_ids))
    return arrays


def make_sequences_multi_window(
    df,
    feature_cols,
    sequence_length,
    min_window_length=2,
    user_col="user_id",
    time_col="relative_week",
    label_col="is_dropout_point",
):
    sequences = []
    labels = []

    for _, user_df in df.groupby(user_col):
        user_df = user_df.sort_values(time_col)
        data = user_df[feature_cols].to_numpy(dtype=np.float32)
        target = user_df[label_col].to_numpy()

        for idx in range(min_window_length, len(user_df)):
            if np.isnan(target[idx]):
                continue
            max_length = min(sequence_length, idx)
            for window_length in range(min_window_length, max_length + 1):
                seq = data[idx - window_length : idx]
                sequences.append(pad_sequence(seq, sequence_length))
                labels.append(float(target[idx]))

    return np.asarray(sequences), np.asarray(labels)
