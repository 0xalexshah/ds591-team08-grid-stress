"""
ISO-NE 5-minute demand streaming simulator.

Reads pre-downloaded ISO-NE 5-minute system demand CSVs and replays them
at real 5-minute cadence, writing one JSON file per row to ADLS Bronze.
Each emitted row carries both event_time (original reading timestamp) and
ingestion_time (when the simulator emitted it), demonstrating the classic
streaming distinction between event-time and processing-time.

Usage:
    export ADLS_CONNECTION_STRING="..."
    python stream_simulator.py --input-dir ./iso_ne_csvs --max-rows 24
"""


import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from azure.storage.filedatalake import DataLakeServiceClient

from dotenv import load_dotenv
load_dotenv()


# ---------- Config ----------
CONTAINER = "bronze"
BRONZE_PATH = "iso-ne-stream/5-minute-system-demand"
REPLAY_INTERVAL_SECONDS = 300  # 5 minutes


# ---------- ISO-NE CSV parsing ----------
def load_iso_ne_csv(csv_path: Path) -> pd.DataFrame:
    """
    ISO-NE Real-Time Five-Minute System Load CSVs have variable-width rows:
      C = comment/metadata (2 cols)
      H = header (5-7 cols)
      D = data (7 cols): record_type, datetime, total_load, native_load,
                        asset_related_load, total_load_w_solar, native_load_w_solar
      T = trailer

    We use names= to force 7 columns and engine='python' to tolerate
    the variable-width metadata rows, then filter to D rows only.
    """
    col_names = [
        "record_type",
        "datetime_raw",
        "total_load_mw",
        "native_load_mw",
        "asset_related_load_mw",
        "total_load_w_solar_mw",
        "native_load_w_solar_mw",
    ]

    df = pd.read_csv(
        csv_path,
        header=None,
        names=col_names,
        engine="python",
        on_bad_lines="skip",
        dtype=str,
        skip_blank_lines=True,
    )

    # Keep only data rows
    df = df[df["record_type"] == "D"].reset_index(drop=True)

    # Parse datetime (format: "04/17/2026 00:00:00")
    df["event_time"] = pd.to_datetime(
        df["datetime_raw"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
    )
    df = df.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)

    # Cast numeric columns
    numeric_cols = [
        "total_load_mw",
        "native_load_mw",
        "asset_related_load_mw",
        "total_load_w_solar_mw",
        "native_load_w_solar_mw",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_all_csvs(input_dir: Path) -> pd.DataFrame:
    """Load every .csv in input_dir, concat, dedupe on event_time, sort."""
    csvs = sorted(input_dir.glob("*.csv"))
    if not csvs:
        sys.exit(f"No CSVs found in {input_dir}")

    print(f"Found {len(csvs)} CSV file(s) in {input_dir}")
    frames = [load_iso_ne_csv(p) for p in csvs]
    combined = pd.concat(frames, ignore_index=True)

    # Dedupe in case of overlap between daily files
    combined = (
        combined.drop_duplicates(subset=["event_time"])
        .sort_values("event_time")
        .reset_index(drop=True)
    )
    print(f"Loaded {len(combined)} total rows "
          f"from {combined['event_time'].min()} to {combined['event_time'].max()}")
    return combined


# ---------- ADLS writer ----------
def get_adls_client() -> DataLakeServiceClient:
    conn_str = os.environ.get("ADLS_CONNECTION_STRING")
    if not conn_str:
        sys.exit("ERROR: Set ADLS_CONNECTION_STRING environment variable before running.")
    return DataLakeServiceClient.from_connection_string(conn_str)


def emit_row(client: DataLakeServiceClient, row: dict) -> str:
    """Write a single-row JSON file to Bronze. Returns the filename."""
    event_time_str = row["event_time"].strftime("%Y%m%dT%H%M%S")
    ingestion_time_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"event_{event_time_str}_ingested_{ingestion_time_str}.json"

    # Partition Bronze by event date so the lake stays tidy
    event_date = row["event_time"].strftime("year=%Y/month=%m/day=%d")
    full_path = f"{BRONZE_PATH}/{event_date}/{filename}"

    # Serialize — convert Timestamps to ISO strings, NaN -> None
    payload = {
        "event_time": row["event_time"].isoformat(),
        "ingestion_time": datetime.now(timezone.utc).isoformat(),
        "total_load_mw": None if pd.isna(row["total_load_mw"]) else float(row["total_load_mw"]),
        "native_load_mw": None if pd.isna(row["native_load_mw"]) else float(row["native_load_mw"]),
        "asset_related_load_mw": None if pd.isna(row["asset_related_load_mw"]) else float(row["asset_related_load_mw"]),
        "total_load_w_solar_mw": None if pd.isna(row["total_load_w_solar_mw"]) else float(row["total_load_w_solar_mw"]),
        "native_load_w_solar_mw": None if pd.isna(row["native_load_w_solar_mw"]) else float(row["native_load_w_solar_mw"]),
        "source": "iso_ne_rt_fiveminsysload",
        "replay_simulated": True,
    }
    body = json.dumps(payload, indent=2).encode("utf-8")

    fs_client = client.get_file_system_client(CONTAINER)
    file_client = fs_client.get_file_client(full_path)
    file_client.upload_data(body, overwrite=True)

    return full_path


# ---------- Main replay loop ----------
def replay(df: pd.DataFrame, max_rows: int, dry_run: bool, fast: bool = False):
    client = None if dry_run else get_adls_client()

    rows_to_emit = df.head(max_rows).to_dict("records")
    print(f"\nReplaying {len(rows_to_emit)} rows at {REPLAY_INTERVAL_SECONDS}s cadence")
    print(f"Estimated total runtime: {len(rows_to_emit) * REPLAY_INTERVAL_SECONDS / 60:.1f} minutes")
    print(f"Destination: {'DRY RUN (no writes)' if dry_run else f'{CONTAINER}/{BRONZE_PATH}/...'}")
    print("-" * 60)

    for i, row in enumerate(rows_to_emit, start=1):
        emit_started = time.monotonic()

        if dry_run:
            path = f"[dry-run] event_{row['event_time']}.json"
        else:
            path = emit_row(client, row)

        print(f"[{i}/{len(rows_to_emit)}] event={row['event_time']}  "
              f"load={row['total_load_mw']} MW  →  {path}")

        # Sleep the remainder of the 5-min interval (except after the last row)
        if i < len(rows_to_emit) and not fast:
            elapsed = time.monotonic() - emit_started
            sleep_for = max(0, REPLAY_INTERVAL_SECONDS - elapsed)
            time.sleep(sleep_for)

    print("-" * 60)
    print("Replay complete.")


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("./iso_ne_csvs"),
                        help="Directory of ISO-NE 5-min demand CSVs (default: ./iso_ne_csvs)")
    parser.add_argument("--max-rows", type=int, default=12,
                        help="Number of rows to replay (default: 12 = 1 hour)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write to ADLS; just print what would be emitted")
    parser.add_argument("--fast", action="store_true",
                        help="Skip 5-minute sleep between rows (bulk load mode)")
    args = parser.parse_args()

    df = load_all_csvs(args.input_dir)
    replay(df, max_rows=args.max_rows, dry_run=args.dry_run, fast=args.fast)


if __name__ == "__main__":
    main()