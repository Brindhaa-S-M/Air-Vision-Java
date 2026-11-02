# AirVision — AI/ML Air Quality Mapping Prototype

A student-oriented Python project that trains a **Random Forest Regression** model to estimate PM2.5 from tabular atmospheric, weather and geographic features, then visualizes predictions on a map.

## Important scientific limitation

The included `data/stations.csv` values are **synthetic demonstration data** created to exercise the application. The model metrics and predicted values are therefore software-demo outputs, not evidence of real-world accuracy. Do not present them as measured pollution, validated satellite downscaling, or information for health/regulatory decisions. Replace the dataset with documented real observations before drawing scientific conclusions.

The map's Esri imagery layer is a geographic basemap, not the model's pollution input. The current prototype accepts NO2 and AOD as input features but does not automatically download satellite products. Genuine satellite-based downscaling requires suitable, traceable satellite-derived features and reference station data aligned in space and time.

## Requirements
- Python 3.10 or newer
- Internet access for the Leaflet map tiles/CDN in the dashboard

## Run on Windows (PowerShell)

Open the extracted project folder in VS Code's terminal:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe train_model.py
.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Then open:
- Dashboard: http://127.0.0.1:8000
- API health: http://127.0.0.1:8000/api/health
- API documentation: http://127.0.0.1:8000/docs
- Metrics: http://127.0.0.1:8000/api/metrics

The first training run creates `models/airvision_random_forest.joblib` and `models/evaluation_metrics.json`. These generated files can be recreated by running `train_model.py`.

## Model design
- **Algorithm:** `sklearn.ensemble.RandomForestRegressor`
- **Features:** latitude, longitude, NO2, AOD, temperature, humidity, wind speed, month
- **Target:** PM2.5
- **Evaluation:** held-out train/test split; MAE, RMSE, R²
- **Prediction:** one location via `POST /api/predict`, or regular grid via `GET /api/prediction-grid`
- **Retraining:** `POST /api/train`

## Replace the demo data
Keep the same CSV column names and use documented real measurements with units, timestamps and data sources. Before model training, check missing values, units, measurement time alignment, duplicates and outliers. Use a spatial or time-based test split when possible: a random split can overstate performance for spatial prediction. Compare against a simple baseline and report uncertainty.

Suggested places to investigate official data:
- OpenAQ: https://openaq.org/
- Copernicus Data Space Ecosystem: https://dataspace.copernicus.eu/
- NASA Earthdata: https://earthdata.nasa.gov/
- Sentinel-5P/TROPOMI documentation: https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-5p

Confirm access, licenses, temporal coverage, spatial resolution and variables before selecting data. Satellite aerosol optical depth and NO2 are not direct ground-level PM2.5 measurements.

## Folder layout

```text
AirVision_AI_ML/
├── backend/
│   ├── main.py
│   └── ml.py
├── data/stations.csv
├── docs/
├── frontend/index.html
├── models/                 # model artifacts generated at training time
├── train_model.py
├── requirements.txt
└── README.md
```
