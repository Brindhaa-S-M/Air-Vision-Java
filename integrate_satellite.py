
from pathlib import Path
import json
import math

import pandas as pd

ROOT = Path(__file__).resolve().parent

GROUND_FILE = ROOT / "data" / "integrated" / "royapuram_pm25.csv"
SATELLITE_FILE = ROOT / "data" / "satellite" / "Royapuram_MAIAC_AOD.csv"

OUTPUT_FILE = ROOT / "data" / "integrated" / "Royapuram_matched.csv"
REPORT_FILE = ROOT / "data" / "integrated" / "matching_report.json"

# Approximate demo coordinates; verify before scientific use.
STATION_LAT = 13.115
STATION_LON = 80.295
MAX_DISTANCE_KM = 2.0


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(min(1.0, a)))


def load_ground_data():
    if not GROUND_FILE.exists():
        raise FileNotFoundError(f"Ground CSV not found: {GROUND_FILE}")

    # Your file has header: date,time,pm25,ug/m3
    # Some rows contain only date,time,pm25. Handle both.
    df = pd.read_csv(GROUND_FILE, dtype=str)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" not in df.columns or "time" not in df.columns:
        raise ValueError("Ground CSV must contain date and time columns.")

    # The fourth header 'ug/m3' is interpreted as the unit column.
    if "pm25" in df.columns:
        value_col = "pm25"
    elif "value" in df.columns:
        value_col = "value"
    else:
        raise ValueError(
            f"Cannot find PM2.5 column. CSV columns: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    ).dt.date
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.dropna(subset=["date", value_col])
    df = df[df[value_col] >= 0]

    if df.empty:
        raise ValueError("No usable PM2.5 readings found in ground CSV.")

    daily = df.groupby("date", as_index=False).agg(
        ground_pm25_ug_m3=(value_col, "mean"),
        ground_readings=(value_col, "count"),
    )

    return daily, len(df)


def load_satellite_demo():
    if not SATELLITE_FILE.exists():
        raise FileNotFoundError(
            f"Satellite CSV not found: {SATELLITE_FILE}"
        )

    df = pd.read_csv(SATELLITE_FILE, dtype=str)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {
        "date", "pixel_latitude", "pixel_longitude",
        "aod_055", "aod_qa", "product"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Satellite CSV missing columns: {sorted(missing)}"
        )

    # Explicitly restrict this script to the labelled demo file.
    if not df["product"].astype(str).str.strip().eq("DEMO_ONLY").all():
        raise ValueError("Expected product=DEMO_ONLY for demo mode.")

    if not df["aod_qa"].astype(str).str.contains(
        "SAMPLE_NOT_REAL", case=False, na=False
    ).all():
        raise ValueError("Expected SAMPLE_NOT_REAL markers in demo file.")

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    ).dt.date

    for col in ["pixel_latitude", "pixel_longitude", "aod_055"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["date", "pixel_latitude", "pixel_longitude", "aod_055"]
    )

    if df.empty:
        raise ValueError("No usable illustrative satellite rows found.")

    return df


def main():
    ground, ground_rows = load_ground_data()
    satellite = load_satellite_demo()

    matches = []

    for _, g in ground.iterrows():
        candidates = satellite[satellite["date"] == g["date"]].copy()
        if candidates.empty:
            continue

        candidates["distance_km"] = candidates.apply(
            lambda s: haversine_km(
                STATION_LAT, STATION_LON,
                s["pixel_latitude"], s["pixel_longitude"]
            ),
            axis=1,
        )

        nearest = candidates.sort_values("distance_km").iloc[0]
        if nearest["distance_km"] > MAX_DISTANCE_KM:
            continue

        matches.append({
            "data_status": "SYNTHETIC_DEMO_NOT_REAL_SATELLITE_DATA",
            "date": str(g["date"]),
            "station": "Royapuram, Chennai",
            "station_latitude_approx": STATION_LAT,
            "station_longitude_approx": STATION_LON,
            "ground_pm25_ug_m3_daily_mean": round(
                float(g["ground_pm25_ug_m3"]), 3
            ),
            "ground_readings_in_daily_mean": int(g["ground_readings"]),
            "pixel_latitude_demo": float(nearest["pixel_latitude"]),
            "pixel_longitude_demo": float(nearest["pixel_longitude"]),
            "aod_055_demo_only": float(nearest["aod_055"]),
            "satellite_quality": "SAMPLE_NOT_REAL",
            "source_product": "DEMO_ONLY",
            "pixel_distance_km": round(float(nearest["distance_km"]), 3),
            "spatial_limit_km": MAX_DISTANCE_KM,
            "temporal_method": "Same calendar date; daily ground mean",
            "exact_overpass_time_matched": False,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matches).to_csv(OUTPUT_FILE, index=False)

    report = {
        "project": "AirVision AI/ML",
        "mode": "DEMO ONLY",
        "real_satellite_data_used": False,
        "ground_file": str(GROUND_FILE.relative_to(ROOT)),
        "satellite_file": str(SATELLITE_FILE.relative_to(ROOT)),
        "ground_rows_used": ground_rows,
        "ground_days": int(len(ground)),
        "illustrative_satellite_rows": int(len(satellite)),
        "matched_days": len(matches),
        "unmatched_ground_days": int(len(ground) - len(matches)),
        "spatial_rule": f"Nearest demo pixel within {MAX_DISTANCE_KM} km",
        "temporal_rule": "Same calendar date; hourly ground readings averaged by day",
        "limitations": [
            "Satellite values and pixel coordinates are invented demo examples.",
            "This is not a real satellite-ground match.",
            "Station coordinates are approximate.",
            "AOD is not equivalent to ground-level PM2.5.",
            "This script does not train or alter the Random Forest model."
        ],
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, indent=2))
    print("\nDemo integration finished.")
    print(f"Matched CSV: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
