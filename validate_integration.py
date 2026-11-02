
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "integrated" / "Royapuram_matched.csv"
REPORT_FILE = ROOT / "data" / "integrated" / "validation_report.json"

REQUIRED_COLUMNS = [
    "data_status",
    "date",
    "ground_pm25_ug_m3_daily_mean",
    "ground_readings_in_daily_mean",
    "aod_055_demo_only",
    "satellite_quality",
    "source_product",
    "pixel_distance_km",
    "exact_overpass_time_matched",
]

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Matched dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ground_pm25_ug_m3_daily_mean"] = pd.to_numeric(
        df["ground_pm25_ug_m3_daily_mean"], errors="coerce"
    )
    df["ground_readings_in_daily_mean"] = pd.to_numeric(
        df["ground_readings_in_daily_mean"], errors="coerce"
    )
    df["pixel_distance_km"] = pd.to_numeric(
        df["pixel_distance_km"], errors="coerce"
    )

    demo_only = (
        df["source_product"].eq("DEMO_ONLY")
        & df["satellite_quality"].eq("SAMPLE_NOT_REAL")
        & df["data_status"].eq("SYNTHETIC_DEMO_NOT_REAL_SATELLITE_DATA")
    )

    report = {
        "project": "AirVision AI/ML",
        "dataset": str(DATA_FILE.relative_to(ROOT)),
        "validation_mode": "DEMO DATA QUALITY CHECK ONLY",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_required_columns": missing,
        "duplicate_dates": int(df["date"].duplicated().sum()),
        "invalid_dates": int(df["date"].isna().sum()),
        "invalid_pm25_values": int(
            df["ground_pm25_ug_m3_daily_mean"].isna().sum()
        ),
        "non_positive_reading_counts": int(
            (df["ground_readings_in_daily_mean"].fillna(0) <= 0).sum()
        ),
        "invalid_pixel_distances": int(df["pixel_distance_km"].isna().sum()),
        "dates_with_demo_only_labels": int(demo_only.sum()),
        "all_rows_marked_demo_only": bool(demo_only.all()),
        "exact_overpass_time_matched_rows": int(
            df["exact_overpass_time_matched"].astype(str).str.lower().eq("true").sum()
        ),
        "pm25_daily_mean_ug_m3": {
            "mean": round(float(df["ground_pm25_ug_m3_daily_mean"].mean()), 3),
            "minimum": round(float(df["ground_pm25_ug_m3_daily_mean"].min()), 3),
            "maximum": round(float(df["ground_pm25_ug_m3_daily_mean"].max()), 3),
        },
        "scientific_validation": False,
        "note": (
            "This checks file structure and basic data quality only. "
            "Satellite values are invented demo examples, and this report "
            "does not establish satellite-ground validity or model performance."
        ),
    }

    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("\nValidation report saved to:", REPORT_FILE)

if __name__ == "__main__":
    main()
