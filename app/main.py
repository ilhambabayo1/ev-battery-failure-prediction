"""FastAPI service for EV battery failure prediction."""

from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.model import load_model

app = FastAPI(title="EV Battery Failure Prediction API")

model, preprocessor, meta = load_model()
FEATURE_COLS = meta["feature_cols"]
THRESHOLD = meta["threshold"]


class BatteryFeatures(BaseModel):
    # Accepts any subset of features; missing ones are filled with 0 before scaling/encoding.
    features: dict


@app.get("/")
def index():
    return FileResponse(Path(__file__).resolve().parent.parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_metrics": meta["metrics"]}


@app.get("/features")
def features():
    return {"feature_cols": FEATURE_COLS, "threshold": THRESHOLD}


@app.post("/predict")
def predict(payload: BatteryFeatures):
    row = {c: float(payload.features.get(c, 0)) for c in FEATURE_COLS}
    X = pd.DataFrame([row], columns=FEATURE_COLS)
    X_scaled = preprocessor.transform(X)
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(X_scaled, dtype=torch.float32))).item()
    return {
        "failure_probability": round(prob, 4),
        "failure_prediction": int(prob > THRESHOLD),
        "label": "Failure" if prob > THRESHOLD else "No failure",
    }

