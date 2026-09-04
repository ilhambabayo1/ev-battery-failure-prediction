"""
Reproduces the Battery.ipynb pipeline and saves every artifact the API needs.

Pipeline (matches the notebook):
  1. Load CSV, fill missing values (mean for numeric, mode for categorical)
  2. Feature selection: top-35 features correlated with battery_failure
  3. Encode categorical features (OneHotEncoder) + scale (StandardScaler)
  4. Handle class imbalance with RandomUnderSampler
  5. Train BatteryMLP (PyTorch) for 50 epochs

Saved artifacts (models/):
  - Battery_model.pkl  : PyTorch model state_dict
  - preprocessor.pkl   : fitted ColumnTransformer (StandardScaler + OneHotEncoder)
  - model_meta.json    : feature lists, thresholds, metrics
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from app.model import BatteryMLP

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ev_battery_failure_dataset.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TOP_N_FEATURES = 35
EPOCHS = 50
BATCH_SIZE = 256
LR = 0.001
THRESHOLD = 0.5


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def select_features(df: pd.DataFrame) -> list[str]:
    correlations = df.corr(numeric_only=True)["battery_failure"].abs().sort_values(ascending=False)
    top = correlations.head(TOP_N_FEATURES + 1).index.tolist()
    top.remove("battery_failure")
    return top


def build_preprocessor(feature_cols: list[str], df: pd.DataFrame):
    cat_cols = [c for c in feature_cols if df[c].dtype == object]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    from sklearn.compose import ColumnTransformer
    return ColumnTransformer(transformers=transformers)


def main() -> None:
    print("Loading data...")
    df = load_data()
    X = df.drop(columns=["battery_failure"])
    y = df["battery_failure"]

    feature_cols = select_features(df)
    X = X[feature_cols]
    print(f"Selected {len(feature_cols)} features")

    print("Resampling (RandomUnderSampler)...")
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_res, y_res = rus.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=RANDOM_STATE, stratify=y_res
    )

    preprocessor = build_preprocessor(feature_cols, df)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

    model = BatteryMLP(X_train_t.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {epoch_loss / len(train_loader):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t))
        y_pred = (probs > THRESHOLD).float()
    metrics = {
        "accuracy": round(accuracy_score(y_test_t, y_pred), 4),
        "precision": round(precision_score(y_test_t, y_pred), 4),
        "recall": round(recall_score(y_test_t, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test_t, y_pred).tolist(),
    }
    print("Metrics:", metrics)

    # ---- SAVE ARTIFACTS ----
    # 1. PyTorch model weights
    torch.save(model.state_dict(), MODELS_DIR / "Battery_model.pkl")
    # 2. Fitted scaler + encoder (one object, applied in the API)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    # 3. Metadata so the API knows the exact input contract
    meta = {
        "feature_cols": feature_cols,
        "n_input_features": int(X_train_t.shape[1]),
        "threshold": THRESHOLD,
        "metrics": metrics,
    }
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
