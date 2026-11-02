# Plan for replacing synthetic data with real observations

1. Choose a target such as hourly or daily PM2.5 from a documented ground monitoring source.
2. Preserve station ID, coordinates, timestamps, units, QA flags and source/license metadata.
3. Obtain satellite-derived features with appropriate QA filtering and spatial/temporal aggregation. Aerosol optical depth and satellite NO2 are proxies/features, not direct ground-level PM2.5.
4. Match satellite and weather features to station observations using a defensible time window and spatial extraction method.
5. Check missingness, duplicates, units, outliers, station representativeness and possible leakage.
6. Establish a simple baseline before ML; compare Random Forest with the baseline.
7. Use spatial-block or time-based validation if the goal is to generalize to new locations or dates.
8. Report MAE, RMSE, R², sample sizes, uncertainty, geographic coverage and known limitations.
9. Keep a reproducible data-processing script and record source URLs, retrieval date, product version, filters and licenses.
10. Do not claim downscaling to finer resolution than the source/observations justify without validation.

Until these steps are completed, keep the dataset and results labeled as synthetic prototype output.
