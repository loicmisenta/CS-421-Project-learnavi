import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def create_dataset(df_full, feature_cols, users_clean, demographics=False, SEQUENCE_LENGTH=12, mode="default"):
    users = df_full["user_id"].unique()

    # First split: train+val vs test
    train_val_users, test_users = train_test_split(
        users, test_size=0.15, random_state=42
    )

    # Second split: train vs validation
    train_users, val_users = train_test_split(
        train_val_users, test_size=0.2, random_state=42
    )

    train_df = df_full[df_full["user_id"].isin(train_users)].copy()
    val_df   = df_full[df_full["user_id"].isin(val_users)].copy()
    test_df  = df_full[df_full["user_id"].isin(test_users)].copy()
    
    scaler = StandardScaler()

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    if demographics == False:
        train_loader, val_loader, test_loader = create_dataset_no_demographics(train_df, val_df, test_df, feature_cols, SEQUENCE_LENGTH, mode)
    else:
        train_loader, val_loader, test_loader = create_dataset_demographics(train_df, val_df, test_df, users_clean, feature_cols, SEQUENCE_LENGTH)
        
    return train_loader, val_loader, test_loader

def create_dataset_no_demographics(train_df, val_df, test_df, feature_cols, SEQUENCE_LENGTH = 12, mode="default"):
    if mode == "default":
        X_train, y_train, u_train = create_sequences(train_df, feature_cols, SEQUENCE_LENGTH)
        X_val, y_val, u_val       = create_sequences(val_df, feature_cols, SEQUENCE_LENGTH)
        X_test, y_test, u_test    = create_sequences(test_df, feature_cols, SEQUENCE_LENGTH)
    elif mode == "drop_summer_test":
        X_train, y_train, u_train = create_sequences(train_df, feature_cols, SEQUENCE_LENGTH)
        X_val, y_val, u_val       = create_sequences(val_df, feature_cols, SEQUENCE_LENGTH)
        X_test, y_test, u_test    = create_sequences(test_df, feature_cols, SEQUENCE_LENGTH, drop_summer_labels=True)
    else : 
        print("Warning : invalid mode, set default mode.")
        X_train, y_train, u_train = create_sequences(train_df, feature_cols, SEQUENCE_LENGTH)
        X_val, y_val, u_val       = create_sequences(val_df, feature_cols, SEQUENCE_LENGTH)
        X_test, y_test, u_test    = create_sequences(test_df, feature_cols, SEQUENCE_LENGTH)
    
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)



    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    
    return train_loader, val_loader, test_loader



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


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
        
        

def create_dataset_demographics(train_df, val_df, test_df, users_clean, feature_cols, SEQUENCE_LENGTH=12):
    X_train, y_train, u_train = create_sequences(train_df, feature_cols, SEQUENCE_LENGTH)
    X_val, y_val, u_val       = create_sequences(val_df, feature_cols, SEQUENCE_LENGTH)
    X_test, y_test, u_test    = create_sequences(test_df, feature_cols, SEQUENCE_LENGTH)
    
    # Copy to avoid modifying original
    df_demo = users_clean.copy()

    # Create mappings
    canton2idx = {c: i for i, c in enumerate(df_demo['canton'].unique())}
    class2idx = {c: i for i, c in enumerate(df_demo['class_level'].unique())}

    # Apply mappings
    df_demo['canton_idx'] = df_demo['canton'].map(canton2idx)
    df_demo['class_idx'] = df_demo['class_level'].map(class2idx)
    
    # One-hot encoding for gender
    gender_ohe = pd.get_dummies(df_demo["gender"], prefix="gender")

    # Replace (not concat repeatedly)
    df_demo = pd.concat(
        [df_demo.drop(columns=["gender"]), gender_ohe],
        axis=1
    )

    gender_cols = gender_ohe.columns.tolist()

    # Index by user_id for fast access
    demo_indexed = df_demo.set_index("user_id")

    # Convert everything to tensors once
    demo_tensors = {
        "canton": torch.tensor(demo_indexed["canton_idx"].values, dtype=torch.long),
        "class": torch.tensor(demo_indexed["class_idx"].values, dtype=torch.long),
        "gender": torch.tensor(demo_indexed[gender_cols].values, dtype=torch.float32),
    }

    # Map user_id -> row index
    user_id_to_idx = {uid: i for i, uid in enumerate(demo_indexed.index)}
    
    train_dataset = SequenceWithDemoDataset(X_train, y_train, u_train, demo_tensors, user_id_to_idx)
    val_dataset   = SequenceWithDemoDataset(X_val, y_val, u_val, demo_tensors, user_id_to_idx)
    test_dataset  = SequenceWithDemoDataset(X_test, y_test, u_test, demo_tensors, user_id_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64)
    test_loader  = DataLoader(test_dataset, batch_size=64)
    
    return train_loader, val_loader, test_loader
    
    
class SequenceWithDemoDataset(Dataset):
    def __init__(self, X, y, user_ids, demo_tensors, user_id_to_idx):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.user_ids = user_ids

        self.demo_tensors = demo_tensors
        self.user_id_to_idx = user_id_to_idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_seq = self.X[idx]
        y = self.y[idx]

        uid = self.user_ids[idx]
        demo_idx = self.user_id_to_idx[uid]

        canton = self.demo_tensors["canton"][demo_idx].long()
        class_level = self.demo_tensors["class"][demo_idx].long()
        gender = self.demo_tensors["gender"][demo_idx]

        return x_seq, gender, canton, class_level, y
        

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size=64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        out = self.fc(out)
        return out.squeeze(-1)

def train_model_no_demographics(model, train_loader, val_loader, max_epoch, device, criterion, optimizer, patience=5):
    best_f1 = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    threshold = 0.5

    for epoch in range(max_epoch):

        # ===== TRAINING =====
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            preds = model(X_batch)

            preds = preds.reshape(-1).float()
            y_batch = y_batch.reshape(-1).float()

            loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(X_batch)

                preds = preds.reshape(-1)
                y_batch = y_batch.reshape(-1)

                loss = criterion(preds, y_batch.float())
                val_loss += loss.item()

                probs = torch.sigmoid(preds)
                preds_binary = (probs > threshold).float()

                all_preds.extend(preds_binary.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # ===== EARLY STOPPING ON F1 =====
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print("✅ New best model!")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("⏹ Early stopping triggered")
            break

    # Load best model at the end
    model.load_state_dict(best_model_weights)

    print(f"Best Validation F1: {best_f1:.4f}")
    
    return model
    
    
    

def evaluate_model(model, test_loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)

            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy())
            targets.extend(y_batch.numpy())

    preds = np.array(preds)
    targets = np.array(targets)
    
    threshold = 0.5
    pred_labels = (preds > threshold).astype(int)
    
    roc = roc_auc_score(targets, preds)
    print("ROC-AUC:", roc)
    print("F1:", f1_score(targets, pred_labels))
    print("Precision:", precision_score(targets, pred_labels))
    print("Recall:", recall_score(targets, pred_labels))
    
    best_f1 = 0
    best_thresh = 0

    for t in np.linspace(0.1, 0.9, 50):
        pred_labels = (preds > t).astype(int)
        f1 = f1_score(targets, pred_labels)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print("Best F1:", best_f1)
    print("Best threshold:", best_thresh)
    


def train_model_demographics(model, train_loader, val_loader, max_epoch, device, criterion, optimizer, patience=5):
    best_f1 = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    threshold = 0.5

    for epoch in range(max_epoch):

        # ===== TRAIN =====
        model.train()
        train_loss = 0

        for X_batch, gender, canton, class_level, y_batch in train_loader:
            X_batch = X_batch.to(device)
            gender = gender.to(device)
            canton = canton.to(device)
            class_level = class_level.to(device)
            y_batch = y_batch.to(device).float()

            optimizer.zero_grad()

            logits = model(X_batch, gender, canton, class_level)

            loss = criterion(logits, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, gender, canton, class_level, y_batch in val_loader:
                X_batch = X_batch.to(device)
                gender = gender.to(device)
                canton = canton.to(device)
                class_level = class_level.to(device)
                y_batch = y_batch.to(device).float()

                logits = model(X_batch, gender, canton, class_level)

                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)

                probs = torch.sigmoid(logits)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        # Convert to numpy
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        preds_binary = (all_probs > threshold).astype(int)

        val_f1 = f1_score(all_labels, preds_binary)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # ===== EARLY STOPPING =====
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print("✅ New best model!")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("⏹ Early stopping triggered")
            break

    # Restore best model
    model.load_state_dict(best_model_weights)

    print(f"Best Validation F1: {best_f1:.4f}")
    return model
    
    
    


class LSTMWithDemo(nn.Module):
    def __init__(self, input_size, n_gender, n_canton, n_class):
        super().__init__()

        # LSTM branch
        self.lstm = nn.LSTM(input_size, hidden_size=64, batch_first=True)

        # Embeddings (ONLY for high-cardinality features)
        self.canton_emb = nn.Embedding(n_canton, 8)
        self.class_emb = nn.Embedding(n_class, 12)

        # gender is one-hot → size = n_gender
        demo_input_size = n_gender + 8 + 12

        self.demo_net = nn.Sequential(
            nn.Linear(demo_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq, gender, canton, class_level):
        # ===== LSTM branch =====
        _, (h_n, _) = self.lstm(x_seq)
        h_last = h_n[-1]  # (batch, 64)

        # ===== Demographic branch =====
        g = gender.float()               # (batch, n_gender)
        c = self.canton_emb(canton)      # (batch, 8)
        cl = self.class_emb(class_level) # (batch, 12)

        # Optional debug (remove later)
        # print(g.shape, c.shape, cl.shape)

        demo = torch.cat([g, c, cl], dim=1)

        d = self.demo_net(demo)

        # ===== Fusion =====
        z = torch.cat([h_last, d], dim=1)

        out = self.classifier(z)

        return out.squeeze(-1)
        
        

def evaluate_model_demographics(model, test_loader, device):
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, gender, canton, class_level, y_batch in test_loader:
            X_batch = X_batch.to(device)
            gender = gender.to(device)
            canton = canton.to(device)
            class_level = class_level.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch, gender, canton, class_level)

            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

        # Convert to numpy
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)