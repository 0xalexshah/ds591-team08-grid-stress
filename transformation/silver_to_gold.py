"""
Silver → Gold transformation for ISO-NE streaming demand data.

Reads the Silver processed_events.csv from ADLS, engineers a model-ready
feature matrix, and writes demand_features.parquet to the Gold layer.

Feature engineering includes:
  - Time features: hour_of_day, day_of_week, month, is_weekend
  - Lag features: load at t-1, t-2, t-3, t-6, t-12 intervals (5-min each)
  - Rolling features: already in Silver (rolling_3_avg_mw, rolling_12_avg_mw)
  - Derived: load_delta_mw, hour_sin/cos cyclical encoding
  - Target: total_load_mw (what the model predicts)

Usage:
    python transformation/silver_to_gold.py
    python transformation/silver_to_gold.py --dry-run
"""

import argparse
import io
import os
import sys
from datetime import timezone

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from azure.storage.filedatalake import DataLakeServiceClient

load_dotenv()

# ---------- Config ----------
SILVER_CONTAINER = "silver"
SILVER_PATH = "iso-ne-stream/5-minute-system-demand/processed_events.csv"
GOLD_CONTAINER = "gold"
GOLD_PATH = "iso-ne-stream/demand_features.parquet"


# ---------- ADLS helpers ----------
def get_client() -> DataLakeServiceClient:
    conn_str = os.environ.get("ADLS_CONNECTION_STRING")
    if not conn_str:
        sys.exit("ERROR: ADLS_CONNECTION_STRING not set in .env")
    return DataLakeServiceClient.from_connection_string(conn_str)


def read_silver(client: DataLakeServiceClient) -> pd.DataFrame:
    print("Reading Silver from ADLS...")
    fs = client.get_file_system_client(SILVER_CONTAINER)
    data = fs.get_file_client(SILVER_PATH).download_file().readall()
    df = pd.read_csv(io.BytesIO(data))
    print(f"  Loaded {len(df)} rows from Silver")
    return df


def write_gold(client: DataLakeServiceClient, df: pd.DataFrame) -> None:
    print(f"Writing Gold to ADLS at {GOLD_CONTAINER}/{GOLD_PATH}...")
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    fs = client.get_file_system_client(GOLD_CONTAINER)
    file_client = fs.get_file_client(GOLD_PATH)
    file_client.upload_data(buf.read(), overwrite=True)
    print(f"  Written {len(df)} rows to Gold")


# ---------- Feature engineering ----------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features...")

    # --- Clean and sort ---
    # Dedupe on event_time keeping latest ingestion (same as before)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
    df["ingestion_time"] = pd.to_datetime(df["ingestion_time"], utc=True, errors="coerce")
    df = (
        df.sort_values("ingestion_time")
        .drop_duplicates(subset=["event_time"], keep="last")
        .sort_values("event_time")
        .reset_index(drop=True)
    )
    print(f"  After dedup: {len(df)} unique event times")

    # Cast numeric columns
    numeric_cols = [
        "total_load_mw", "native_load_mw", "asset_related_load_mw",
        "total_load_w_solar_mw", "native_load_w_solar_mw",
        "rolling_3_avg_mw", "rolling_12_avg_mw",
        "ingestion_lag_seconds", "demand_change_pct"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Time features ---
    df["hour_of_day"] = df["event_time"].dt.hour
    df["minute_of_hour"] = df["event_time"].dt.minute
    df["day_of_week"] = df["event_time"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"] = df["event_time"].dt.month
    df["day_of_year"] = df["event_time"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_weekday"] = (~df["day_of_week"].isin([5, 6])).astype(int)

    # Cyclical encoding — prevents model treating hour 23 and hour 0 as far apart
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # --- Lag features (5-min intervals) ---
    # Each lag = one 5-minute step back in event_time
    # Only valid if the previous row is exactly 5 min earlier (no gaps)
    df["prev_event_time"] = df["event_time"].shift(1)
    df["time_gap_minutes"] = (
        df["event_time"] - df["prev_event_time"]
    ).dt.total_seconds() / 60

    # Flag rows where previous row is exactly 5 min back (no gap)
    is_consecutive = df["time_gap_minutes"] == 5.0

    for lag in [1, 2, 3, 6, 12]:
        col = f"load_lag_{lag}"
        df[col] = df["total_load_mw"].shift(lag)
        # Null out lags that cross a gap (different day or missing interval)
        # Conservative: only null lag_1; for longer lags use rolling check
        if lag == 1:
            df.loc[~is_consecutive, col] = np.nan

    # Load delta from previous interval
    df["load_delta_mw"] = df["total_load_mw"] - df["load_lag_1"]

    # --- Season feature ---
    # Simple meteorological seasons for New England
    def get_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    df["season"] = df["month"].apply(get_season)
    # Also encode as integer for models that need numeric
    season_map = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}
    df["season_int"] = df["season"].map(season_map)

    # --- Peak period flag ---
    # Morning peak: 7-10 AM, Evening peak: 5-8 PM (typical NE grid peaks)
    df["is_morning_peak"] = df["hour_of_day"].between(7, 10).astype(int)
    df["is_evening_peak"] = df["hour_of_day"].between(17, 20).astype(int)
    df["is_peak_hour"] = (df["is_morning_peak"] | df["is_evening_peak"]).astype(int)

    # --- Target variable ---
    # Next interval's load — what the model predicts
    df["target_load_mw"] = df["total_load_mw"].shift(-1)
    # Null out targets that cross day boundaries
    next_gap = (df["event_time"].shift(-1) - df["event_time"]).dt.total_seconds() / 60
    df.loc[next_gap != 5.0, "target_load_mw"] = np.nan

    # --- Clean up helper columns ---
    df = df.drop(columns=["prev_event_time", "time_gap_minutes"])

    # --- Final column selection for Gold ---
    gold_columns = [
        # Identifiers
        "event_time",
        # Target
        "target_load_mw",
        # Raw load signals
        "total_load_mw",
        "native_load_mw",
        "asset_related_load_mw",
        "total_load_w_solar_mw",
        "native_load_w_solar_mw",
        # Pre-computed Silver features
        "rolling_3_avg_mw",
        "rolling_12_avg_mw",
        "demand_change_pct",
        "stress_flag",
        "ingestion_lag_seconds",
        # Time features
        "hour_of_day",
        "minute_of_hour",
        "day_of_week",
        "month",
        "day_of_year",
        "is_weekend",
        "is_weekday",
        "is_morning_peak",
        "is_evening_peak",
        "is_peak_hour",
        "season",
        "season_int",
        # Cyclical encodings
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        # Lag features
        "load_lag_1",
        "load_lag_2",
        "load_lag_3",
        "load_lag_6",
        "load_lag_12",
        "load_delta_mw",
    ]

    # Only keep columns that exist
    gold_columns = [c for c in gold_columns if c in df.columns]
    df = df[gold_columns]

    print(f"  Feature matrix shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Null counts in key columns:")
    for col in ["target_load_mw", "load_lag_1", "load_lag_12", "rolling_12_avg_mw"]:
        if col in df.columns:
            print(f"    {col}: {df[col].isna().sum()} nulls")

    return df


# ---------- Summary stats ----------
def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Gold Layer Summary")
    print("=" * 60)
    print(f"Total rows:          {len(df)}")
    print(f"Date range:          {df['event_time'].min()} → {df['event_time'].max()}")
    print(f"Load range:          {df['total_load_mw'].min():.1f} → {df['total_load_mw'].max():.1f} MW")
    print(f"Avg load:            {df['total_load_mw'].mean():.1f} MW")
    print(f"Seasons covered:     {df['season'].value_counts().to_dict()}")
    print(f"Weekend rows:        {df['is_weekend'].sum()} ({df['is_weekend'].mean()*100:.1f}%)")
    print(f"Peak hour rows:      {df['is_peak_hour'].sum()} ({df['is_peak_hour'].mean()*100:.1f}%)")
    print(f"Stress events:       {(df['stress_flag'] == True).sum()}")
    print(f"Rows with target:    {df['target_load_mw'].notna().sum()}")
    print(f"Rows with lag_1:     {df['load_lag_1'].notna().sum()}")
    print("=" * 60)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Process and print stats but don't write to ADLS")
    args = parser.parse_args()

    client = get_client()

    # Read Silver
    silver_df = read_silver(client)

    # Engineer features
    gold_df = engineer_features(silver_df)

    # Print summary
    print_summary(gold_df)

    # Write Gold
    if args.dry_run:
        print("\nDRY RUN — skipping ADLS write")
        print("First 3 rows preview:")
        print(gold_df.head(3).to_string())
    else:
        write_gold(client, gold_df)
        print(f"\nGold layer written to {GOLD_CONTAINER}/{GOLD_PATH}")
        print("Ready for modeling.")


if __name__ == "__main__":
    main()