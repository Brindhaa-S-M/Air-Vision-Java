# AirVision viva guide

## 30-second explanation
“AirVision is an educational prototype for spatial air-quality estimation. I use a Random Forest Regression pipeline in scikit-learn to learn a relationship between geographic, atmospheric and weather features and PM2.5. I evaluate it with MAE and RMSE on a held-out test subset, then predict a grid and display the outputs on a Leaflet map. The included dataset is synthetic, so the current results demonstrate the software pipeline only. Scientific conclusions require real, documented data and spatial validation.”

## Questions to understand before presenting

**Why Random Forest?** It combines many decision trees and can model nonlinear relationships without requiring a linear relationship between every feature and the target. It is a useful baseline, not automatically the best model.

**What is regression?** A supervised learning task that predicts a continuous numeric value; here the target is PM2.5 concentration.

**What are the features and target?** Features are latitude, longitude, NO2, AOD, temperature, humidity, wind speed and month. Target is PM2.5.

**What is MAE?** Mean Absolute Error: average absolute difference between prediction and observed target, in the target's units.

**What is RMSE?** Root Mean Squared Error: square root of mean squared prediction errors. It penalizes larger errors more heavily than MAE.

**What is R²?** A measure comparing model residual variation with a constant-mean baseline. It can be negative on a test set and is unstable with very small test sets.

**Why split data?** A held-out subset gives an estimate of performance on examples not used to fit the model. With spatial data, a random split can still leak geographic similarity; spatial or time-based validation is preferable.

**Does a satellite basemap measure pollution?** No. The imagery layer provides visual geographic context. Actual satellite-derived features need to be downloaded/processed and aligned with ground observations.

**Is this validated downscaling?** Not yet. It is a prototype grid prediction. Valid downscaling requires real satellite products, station observations, appropriate spatial/temporal alignment, a defensible validation design and uncertainty reporting.

**Can the demo results be used for health decisions?** No. The input dataset is synthetic and the model is not validated.

## Run and explain
1. Install dependencies from `requirements.txt`.
2. Run `python train_model.py` and inspect the printed metrics.
3. Run `python -m uvicorn backend.main:app --reload`.
4. Open `/docs` to inspect API endpoints.
5. Use the dashboard to produce a prediction grid and a single-location estimate.
6. Explain which parts are prototype features and what real data work remains.
