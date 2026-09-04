# EV Battery Failure Prediction

PyTorch MLP that predicts EV battery failure from vehicle/battery telemetry.

## Pipeline (matches `Battery.ipynb`)
1. Fill missing values (mean for numeric, mode for categorical)
2. Feature selection: top-35 features correlated with `battery_failure`
3. Encode categoricals (OneHotEncoder) + scale (StandardScaler) — saved together in `models/preprocessor.pkl`
4. Handle class imbalance with `RandomUnderSampler`
5. Train `BatteryMLP` (35 → 64 → 32 → 1) for 50 epochs

## Saved artifacts (`models/`)
| File | Contents |
|---|---|
| `Battery_model.pkl` | PyTorch model `state_dict` |
| `preprocessor.pkl` | Fitted StandardScaler + OneHotEncoder (`ColumnTransformer`) |
| `model_meta.json` | Feature list, decision threshold, test metrics |

## Retrain
```bash
python train.py
```

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000

## Run with Docker
```bash
docker build -t battery-api .
docker run --rm -p 8000:8000 battery-api
```

## Endpoints
- `GET /` — simple prediction form
- `GET /health` — liveness + training metrics
- `GET /features` — feature names the model expects
- `POST /predict` — `{"features": {"state_of_health": 42.1, ...}}` → probability + label

