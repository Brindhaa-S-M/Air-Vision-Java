"""Training and inference helpers for the AirVision Random Forest prototype."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "real" / "airvision_sample_dataset.csv"
MODEL_PATH = ROOT / "models" / "airvision_random_forest.joblib"
METRICS_PATH = ROOT / "models" / "evaluation_metrics.json"

FEATURES = [
    "latitude", "longitude", "no2", "aod",
    "temperature", "humidity", "wind_speed", "month"
]
TARGET = "pm25"


def load_data():
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing columns: {missing}")
    df = df[required].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET])
    if len(df) < 10:
        raise ValueError("At least 10 rows with PM2.5 targets are required for this prototype.")
    return df


def train_model():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    # Keep a held-out test set. Metrics are meaningful only if the data are representative.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=250, max_depth=8, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "model": "RandomForestRegressor",
        "target": TARGET,
        "features": FEATURES,
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4) if len(y_test) > 1 else None,
        "data_notice": "Dataset is synthetic demonstration data; these metrics are not scientific evidence."
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": FEATURES, "target": TARGET}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model():
    if not MODEL_PATH.exists():
        train_model()
    artifact = joblib.load(MODEL_PATH)
    return artifact["pipeline"]


def predict_rows(rows):
    pipeline = load_model()
    frame = pd.DataFrame(rows)
    for feature in FEATURES:
        if feature not in frame.columns:
            frame[feature] = np.nan
    frame = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    preds = pipeline.predict(frame)
    return [max(0.0, float(p)) for p in preds]
