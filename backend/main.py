
from pathlib import Path
import json
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.ml import (
    DATA_PATH,
    METRICS_PATH,
    FEATURES,
    load_data,
    load_model,
    predict_rows,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="AirVision AI/ML API",
    description=(
        "Educational prototype for Random Forest PM2.5 "
        "estimation and spatial visualization."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict before public deployment.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    no2: float = Field(ge=0)
    aod: float = Field(ge=0)
    temperature: float
    humidity: float = Field(ge=0, le=100)
    wind_speed: float = Field(ge=0)
    month: int = Field(ge=1, le=12)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "project": "AirVision",
        "model": "Random Forest Regression",
    }


@app.get("/api/stations")
def stations():
    try:
        df = load_data()
        features = df.to_dict(orient="records")

        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            float(row["longitude"]),
                            float(row["latitude"]),
                        ],
                    },
                    "properties": {
                        **{
                            key: (
                                float(value)
                                if pd.notna(value)
                                else None
                            )
                            for key, value in row.items()
                        },
                        "is_demo": True,
                    },
                }
                for row in features
            ],
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        try:
            train_model()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/api/train")
def retrain():
    try:
        return train_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/predict")
def predict(payload: PredictionInput):
    try:
        value = predict_rows([payload.model_dump()])[0]

        return {
            "predicted_pm25": round(value, 2),
            "unit": "µg/m³",
            "model": "Random Forest Regression",
            "notice": (
                "Prototype estimate from synthetic demo training data; "
                "not for health or regulatory decisions."
            ),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/prediction-grid")
def prediction_grid(
    south: float = 12.7,
    west: float = 80.0,
    north: float = 13.3,
    east: float = 80.4,
    resolution: int = 8,
    no2: float = 20,
    aod: float = 0.3,
    temperature: float = 30,
    humidity: float = 65,
    wind_speed: float = 2,
    month: int = 1,
):
    # Validate the requested geographic bounds.
    if not (
        -90 <= south < north <= 90
        and -180 <= west < east <= 180
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid geographic bounds.",
        )

    if resolution < 3 or resolution > 30:
        raise HTTPException(
            status_code=400,
            detail="resolution must be between 3 and 30.",
        )

    if (
        no2 < 0
        or aod < 0
        or not 0 <= humidity <= 100
        or wind_speed < 0
        or not 1 <= month <= 12
    ):
        raise HTTPException(
            status_code=400,
            detail="One or more feature values are outside valid ranges.",
        )

    try:
        # Confirm that the configured training dataset is available.
        load_data()

        latitudes = [
            south + (north - south) * i / (resolution - 1)
            for i in range(resolution)
        ]
        longitudes = [
            west + (east - west) * j / (resolution - 1)
            for j in range(resolution)
        ]

        rows = []

        for lat in latitudes:
            for lon in longitudes:
                rows.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "no2": no2,
                        "aod": aod,
                        "temperature": temperature,
                        "humidity": humidity,
                        "wind_speed": wind_speed,
                        "month": month,
                    }
                )

        predictions = predict_rows(rows)

        points = [
            {
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "pm25": round(prediction, 2),
            }
            for row, prediction in zip(rows, predictions)
        ]

        return {
            "model": "Random Forest Regression",
            "unit": "µg/m³",
            "resolution": resolution,
            "points": points,
            "notice": (
                "Spatial model output from synthetic demo training data. "
                "It is not validated real-world downscaling."
            ),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Serve the dashboard after registering the API routes.
app.mount(
    "/",
    StaticFiles(directory=ROOT / "frontend", html=True),
    name="frontend",
)

