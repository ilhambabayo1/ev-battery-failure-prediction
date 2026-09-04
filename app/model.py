"""Model definition + artifact loading for the EV battery failure API."""

import json
from pathlib import Path

import joblib
import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class BatteryMLP(nn.Module):
    """Same architecture as the notebook: 35 -> 64 -> 32 -> 1."""

    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_model():
    meta = json.loads((MODELS_DIR / "model_meta.json").read_text())
    model = BatteryMLP(meta["n_input_features"])
    model.load_state_dict(torch.load(MODELS_DIR / "Battery_model.pkl", map_location="cpu"))
    model.eval()
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    return model, preprocessor, meta

