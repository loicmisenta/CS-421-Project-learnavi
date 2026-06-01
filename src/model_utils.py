from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .metrics_utils import best_threshold_by_f1, binary_metrics
from .sequence_utils import make_sequences

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # helpers fail only if called
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


class SequenceDataset(Dataset):
    def __init__(self, x, y):
        if torch is None:
            raise ModuleNotFoundError("SequenceDataset requires torch.")
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SequenceWithDemoDataset(Dataset):
    def __init__(self, x, y, user_ids, demo_lookup):
        if torch is None:
            raise ModuleNotFoundError("SequenceWithDemoDataset requires torch.")
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.user_ids = list(user_ids)
        self.demo_lookup = demo_lookup

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        gender, canton, class_level = self.demo_lookup[user_id]
        return self.x[idx], gender, canton, class_level, self.y[idx]


def make_demo_lookup(users):
    if torch is None:
        raise ModuleNotFoundError("make_demo_lookup requires torch.")

    demo = users[["user_id", "gender", "canton", "class_level"]].copy()
    gender_ohe = pd.get_dummies(demo["gender"], prefix="gender").astype(float)
    demo = pd.concat([demo.drop(columns=["gender"]), gender_ohe], axis=1)

    canton_values = sorted(demo["canton"].astype(str).unique())
    class_values = sorted(demo["class_level"].astype(str).unique())
    canton_to_idx = {value: idx for idx, value in enumerate(canton_values)}
    class_to_idx = {value: idx for idx, value in enumerate(class_values)}
    gender_cols = list(gender_ohe.columns)

    lookup = {}
    for _, row in demo.iterrows():
        lookup[row["user_id"]] = (
            torch.tensor(row[gender_cols].to_numpy(dtype=np.float32)),
            torch.tensor(canton_to_idx[str(row["canton"])]).long(),
            torch.tensor(class_to_idx[str(row["class_level"])]).long(),
        )

    sizes = {
        "n_gender": len(gender_cols),
        "n_canton": len(canton_values),
        "n_class": len(class_values),
    }
    return lookup, sizes


def make_sequence_loaders(
    train_df,
    val_df,
    test_df,
    feature_cols,
    sequence_length,
    batch_size=64,
    users=None,
    demographics=False,
    drop_summer_test=False,
):
    if DataLoader is None:
        raise ModuleNotFoundError("make_sequence_loaders requires torch.")

    x_train, y_train, u_train = make_sequences(
        train_df,
        feature_cols,
        sequence_length,
        return_user_ids=True,
    )
    x_val, y_val, u_val = make_sequences(
        val_df,
        feature_cols,
        sequence_length,
        return_user_ids=True,
    )
    x_test, y_test, u_test = make_sequences(
        test_df,
        feature_cols,
        sequence_length,
        drop_summer_labels=drop_summer_test,
        return_user_ids=True,
    )

    if demographics:
        if users is None:
            raise ValueError("users must be provided when demographics=True.")
        demo_lookup, demo_sizes = make_demo_lookup(users)
        train_dataset = SequenceWithDemoDataset(x_train, y_train, u_train, demo_lookup)
        val_dataset = SequenceWithDemoDataset(x_val, y_val, u_val, demo_lookup)
        test_dataset = SequenceWithDemoDataset(x_test, y_test, u_test, demo_lookup)
    else:
        demo_sizes = None
        train_dataset = SequenceDataset(x_train, y_train)
        val_dataset = SequenceDataset(x_val, y_val)
        test_dataset = SequenceDataset(x_test, y_test)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size),
    ), demo_sizes


if nn is not None:

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=64):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size=hidden_size,
                batch_first=True,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.classifier(hidden[-1]).squeeze(-1)


    class LSTMWithDemographics(nn.Module):
        def __init__(self, input_size, n_gender, n_canton, n_class, hidden_size=64):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size=hidden_size,
                batch_first=True,
            )
            self.canton_embedding = nn.Embedding(n_canton, 8)
            self.class_embedding = nn.Embedding(n_class, 12)
            self.demo_net = nn.Sequential(
                nn.Linear(n_gender + 8 + 12, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size + 16, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x, gender, canton, class_level):
            _, (hidden, _) = self.lstm(x)
            sequence_repr = hidden[-1]
            demo_repr = self.demo_net(
                torch.cat(
                    [
                        gender.float(),
                        self.canton_embedding(canton),
                        self.class_embedding(class_level),
                    ],
                    dim=1,
                )
            )
            return self.classifier(
                torch.cat([sequence_repr, demo_repr], dim=1)
            ).squeeze(-1)

else:
    LSTMModel = None
    LSTMWithDemographics = None


def batch_to_device(batch, device, demographics=False):
    if demographics:
        x, gender, canton, class_level, y = batch
        return (
            x.to(device),
            gender.to(device),
            canton.to(device),
            class_level.to(device),
            y.to(device),
        )
    x, y = batch
    return x.to(device), y.to(device)


def forward_batch(model, batch, demographics=False):
    if demographics:
        x, gender, canton, class_level, _ = batch
        return model(x, gender, canton, class_level)
    x, _ = batch
    return model(x)


def labels_from_batch(batch, demographics=False):
    return batch[-1] if demographics else batch[1]


def collect_probabilities(model, loader, device, demographics=False):
    if torch is None:
        raise ModuleNotFoundError("collect_probabilities requires torch.")

    model.eval()
    probabilities = []
    targets = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = batch_to_device(raw_batch, device, demographics=demographics)
            logits = forward_batch(model, batch, demographics=demographics)
            y = labels_from_batch(batch, demographics=demographics)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy())
            targets.extend(y.cpu().numpy())

    return np.asarray(probabilities), np.asarray(targets)


def train_lstm_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=30,
    patience=5,
    learning_rate=1e-3,
    demographics=False,
):
    if torch is None or nn is None:
        raise ModuleNotFoundError("train_lstm_model requires torch.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for raw_batch in train_loader:
            batch = batch_to_device(raw_batch, device, demographics=demographics)
            y = labels_from_batch(batch, demographics=demographics).float()

            optimizer.zero_grad()
            logits = forward_batch(model, batch, demographics=demographics)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = y.shape[0]
            train_loss += loss.item() * batch_size
            n_train += batch_size

        val_prob, val_y = collect_probabilities(
            model,
            val_loader,
            device,
            demographics=demographics,
        )
        val_threshold, val_f1 = best_threshold_by_f1(val_y, val_prob)
        train_loss = train_loss / max(n_train, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_f1": val_f1,
                "val_threshold": val_threshold,
            }
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def evaluate_lstm_model(model, val_loader, test_loader, device, demographics=False):
    val_prob, val_y = collect_probabilities(model, val_loader, device, demographics)
    threshold, val_f1 = best_threshold_by_f1(val_y, val_prob)
    test_prob, test_y = collect_probabilities(model, test_loader, device, demographics)
    metrics = binary_metrics(test_y, test_prob, threshold)
    metrics["val_f1"] = val_f1
    return metrics, test_prob, test_y
