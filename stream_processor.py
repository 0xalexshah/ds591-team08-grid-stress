"""
ISO-NE 5-minute demand stream processor (Bronze -> Silver).

Polls the Bronze layer for new event JSON files produced by stream_simulator.py,
transforms each into a Silver record with derived streaming metrics, and appends
to a single consolidated Silver CSV.

Key derived fields:
  - ingestion_lag_seconds : ingestion_time - event_time
                            (demonstrates event-time vs. processing-time semantics)
  - rolling_3_avg_mw      : 3-point rolling average of total_load_mw (~15 min)
  - rolling_12_avg_mw     : 12-point rolling average of total_load_mw (~1 hour)
  - demand_change_pct     : % change in total_load_mw vs. previous event
  - stress_flag           : True when demand deviates sharply from rolling 1-hr avg
                            (proxy for Prediction 4: forecast-accuracy-linked stress)

Usage:
    python stream_processor.py                  # runs forever, polls every 60s
    python stream_processor.py --once           # single pass, useful for debugging
    python stream_processor.py --poll 30        # poll every 30s instead of 60
"""

import argparse
import csv
import io
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from azure.storage.filedatalake import DataLakeServiceClient

load_dotenv()

# ---------- Config ----------
CONTAINER = "bronze"
BRONZE_PATH = "iso-ne-stream/5-minute-system-demand"
SILVER_CONTAINER = "silver"
SILVER_PATH = "iso-ne-stream/5-minute-system-demand"
SILVER_FILENAME = "processed_events.csv"

# Stress detection threshold: flag any event where demand deviates from the
# rolling 1-hr average by more than this percentage.
STRESS_PCT_THRESHOLD = 5.0

SILVER_COLUMNS = [
    "event_time",
    "ingestion_time",
    "ingestion_lag_seconds",
    "total_load_mw",
    "native_load_mw",
    "asset_related_load_mw",
    "total_load_w_solar_mw",
    "native_load_w_solar_mw",
    "rolling_3_avg_mw",
    "rolling_12_avg_mw",
    "demand_change_pct",
    "stress_flag",
    "source",
    "bronze_file",
    "processed_at",
]


# ---------- ADLS client ----------
def get_adls_client() -> DataLakeServiceClient:
    conn_str = os.environ.get("ADLS_CONNECTION_STRING")
    if not conn_str:
        sys.exit("ERROR: Set ADLS_CONNECTION_STRING in .env or environment.")
    return DataLakeServiceClient.from_connection_string(conn_str)


# ---------- Bronze discovery ----------
def list_bronze_files(client: DataLakeServiceClient) -> list[str]:
    """Return sorted list of all JSON paths currently in Bronze."""
    fs = client.get_file_system_client(CONTAINER)
    paths = fs.get_paths(path=BRONZE_PATH, recursive=True)
    return sorted(p.name for p in paths if p.name.endswith(".json"))


def read_bronze_json(client: DataLakeServiceClient, path: str) -> dict:
    """Download and parse a single Bronze JSON file."""
    fs = client.get_file_system_client(CONTAINER)
    data = fs.get_file_client(path).download_file().readall()
    return json.loads(data)


# ---------- Silver I/O ----------
def silver_full_path() -> str:
    return f"{SILVER_PATH}/{SILVER_FILENAME}"


def read_silver_if_exists(client: DataLakeServiceClient) -> tuple[list[dict], set[str]]:
    """
    Load existing Silver CSV (if any) so we can append + dedup.
    Returns (existing_rows, set_of_bronze_files_already_processed).
    """
    fs = client.get_file_system_client(SILVER_CONTAINER)
    file_client = fs.get_file_client(silver_full_path())
    try:
        data = file_client.download_file().readall().decode("utf-8")
    except Exception:
        return [], set()
    reader = csv.DictReader(io.StringIO(data))
    rows = list(reader)
    processed = {r["bronze_file"] for r in rows}
    return rows, processed


def write_silver(client: DataLakeServiceClient, rows: list[dict]) -> None:
    """Overwrite Silver CSV with the full row set (simple and correct)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=SILVER_COLUMNS)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    body = buf.getvalue().encode("utf-8")

    fs = client.get_file_system_client(SILVER_CONTAINER)
    file_client = fs.get_file_client(silver_full_path())
    file_client.upload_data(body, overwrite=True)


# ---------- Transformation ----------
def compute_derived_fields(
    event: dict,
    bronze_file: str,
    recent_loads: deque,
    prev_load: float | None,
) -> dict:
    """Turn one Bronze event into one Silver row with derived metrics."""
    event_dt = datetime.fromisoformat(event["event_time"])
    ingest_dt = datetime.fromisoformat(event["ingestion_time"])

    # Event time from the simulator is naive (no tz); the original data is ET.
    # For lag math we just need both in the same tz; assume event is UTC-naive,
    # make it UTC-aware so subtraction with ingestion_time works. This is fine
    # for a relative-lag demonstration — production would handle ET -> UTC properly.
    if event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)

    lag_seconds = (ingest_dt - event_dt).total_seconds()

    total_load = event["total_load_mw"]

    # Update rolling window (keep last 12 points for 1-hr window)
    if total_load is not None:
        recent_loads.append(total_load)

    rolling_3 = (
        sum(list(recent_loads)[-3:]) / min(3, len(recent_loads))
        if recent_loads else None
    )
    rolling_12 = sum(recent_loads) / len(recent_loads) if recent_loads else None

    # % change from previous event
    if prev_load is not None and prev_load != 0 and total_load is not None:
        change_pct = ((total_load - prev_load) / prev_load) * 100.0
    else:
        change_pct = None

    # Stress flag: demand deviates >THRESHOLD% from rolling 1-hr avg
    # (only meaningful once we have a few points in the window)
    if rolling_12 is not None and total_load is not None and len(recent_loads) >= 3:
        deviation_pct = abs((total_load - rolling_12) / rolling_12) * 100.0
        stress_flag = deviation_pct > STRESS_PCT_THRESHOLD
    else:
        stress_flag = False

    return {
        "event_time": event["event_time"],
        "ingestion_time": event["ingestion_time"],
        "ingestion_lag_seconds": f"{lag_seconds:.1f}",
        "total_load_mw": total_load,
        "native_load_mw": event.get("native_load_mw"),
        "asset_related_load_mw": event.get("asset_related_load_mw"),
        "total_load_w_solar_mw": event.get("total_load_w_solar_mw"),
        "native_load_w_solar_mw": event.get("native_load_w_solar_mw"),
        "rolling_3_avg_mw": f"{rolling_3:.3f}" if rolling_3 is not None else "",
        "rolling_12_avg_mw": f"{rolling_12:.3f}" if rolling_12 is not None else "",
        "demand_change_pct": f"{change_pct:.3f}" if change_pct is not None else "",
        "stress_flag": stress_flag,
        "source": event.get("source", ""),
        "bronze_file": bronze_file,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------- Main loop ----------
def process_pass(client: DataLakeServiceClient) -> int:
    """
    Single pass: find new Bronze files, process them, update Silver.
    Returns number of new rows added.
    """
    existing_rows, processed_files = read_silver_if_exists(client)

    all_bronze = list_bronze_files(client)
    new_bronze = [p for p in all_bronze if p not in processed_files]

    if not new_bronze:
        return 0

    # Rebuild rolling state from existing rows so derived fields stay consistent
    # across restarts (critical for rolling averages to be meaningful).
    recent_loads: deque = deque(maxlen=12)
    prev_load: float | None = None
    for r in existing_rows[-12:]:
        try:
            recent_loads.append(float(r["total_load_mw"]))
        except (ValueError, TypeError):
            pass
    if existing_rows:
        try:
            prev_load = float(existing_rows[-1]["total_load_mw"])
        except (ValueError, TypeError):
            prev_load = None

    new_rows = []
    # Process in event-time order so rolling metrics build up correctly
    bronze_events = []
    for path in new_bronze:
        try:
            event = read_bronze_json(client, path)
            bronze_events.append((event["event_time"], path, event))
        except Exception as e:
            print(f"  [warn] could not read {path}: {e}")

    bronze_events.sort(key=lambda x: x[0])

    for _, path, event in bronze_events:
        row = compute_derived_fields(event, path, recent_loads, prev_load)
        new_rows.append(row)
        if event.get("total_load_mw") is not None:
            prev_load = event["total_load_mw"]

    combined = existing_rows + new_rows
    write_silver(client, combined)

    for row in new_rows:
        flag = "⚠ STRESS" if row["stress_flag"] else "  normal"
        print(f"  -> event={row['event_time']}  load={row['total_load_mw']} MW  "
              f"lag={row['ingestion_lag_seconds']}s  {flag}")

    return len(new_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true",
                        help="Single pass and exit (default: poll forever)")
    parser.add_argument("--poll", type=int, default=60,
                        help="Seconds between polls in loop mode (default: 60)")
    args = parser.parse_args()

    client = get_adls_client()

    if args.once:
        print("Single-pass mode.")
        added = process_pass(client)
        print(f"Done. Added {added} new Silver row(s).")
        return

    print(f"Polling {CONTAINER}/{BRONZE_PATH}/ every {args.poll}s. Ctrl+C to stop.")
    print("-" * 60)
    try:
        while True:
            ts = datetime.now().strftime("%H:%M:%S")
            added = process_pass(client)
            if added:
                print(f"[{ts}] Added {added} new row(s) to Silver.")
            else:
                print(f"[{ts}] No new Bronze files.")
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()