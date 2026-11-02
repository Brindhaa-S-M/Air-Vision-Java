
import json
from pathlib import Path

import pandas as pd

from backend.ml import FEATURES, TARGET

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "integrated" / "Royapuram_matched.csv"
REPORT_FILE = ROOT / "data" / "integrated" / "model_compatibility.json"

COLUMN_MAPPING = {
    "latitude": "station_latitude_approx",
    "longitude": "station_longitude_approx",
    "pm25": "ground_pm25_ug_m3_daily_mean",
}

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    # Identify which model inputs are available in the matched file.
    available = []
    missing = []

    for feature in FEATURES:
        source_column = COLUMN_MAPPING.get(feature, feature)

        if feature == "month":
            available.append({
                "model_feature": feature,
                "source_column": "date",
                "status": "DERIVABLE",
            })
        elif source_column in df.columns:
            status = (
                "DEMO ONLY - NOT VALID FOR SCIENTIFIC TRAINING"
                if feature == "aod"
                else "AVAILABLE - CHECK PROVENANCE"
            )
            available.append({
                "model_feature": feature,
                "source_column": source_column,
                "status": status,
            })
        else:
            missing.append(feature)

    demo_only = (
        df["source_product"].astype(str).eq("DEMO_ONLY").all()
        and df["satellite_quality"].astype(str).eq("SAMPLE_NOT_REAL").all()
    )

    report = {
        "project": "AirVision AI/ML",
        "dataset": str(DATA_FILE.relative_to(ROOT)),
        "model_features_required": FEATURES,
        "target_required": TARGET,
        "available_or_derivable_features": available,
        "missing_features": missing,
        "row_count": int(len(df)),
        "all_satellite_values_demo_only": bool(demo_only),
        "direct_training_ready": False,
        "recommendation": (
            "Do not train the existing model on this matched file. "
            "Obtain genuine, appropriately time-matched satellite data "
            "and the missing meteorological and NO2 features first."
        ),
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("\nReport saved to:", REPORT_FILE)

if __name__ == "__main__":
    main()
