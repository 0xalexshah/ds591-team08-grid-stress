"""
Batch Silver → Gold transformation for EIA electricity consumption data.

Reads the EIA Silver Parquet from ADLS, cleans and engineers features,
and writes two Gold tables:
  1. consumption_monthly.parquet  — cleaned monthly time series per state/sector
  2. consumption_summary.parquet  — aggregated features for Predictions 1-3

Supports:
  Prediction 1: Seasonal demand patterns (summer vs winter by state)
  Prediction 2: Industrial growth trends 2022-2026 (AI/data center proxy)
  Prediction 3: Monthly demand patterns as proxy for weekly operational cycles

Usage:
    python transformation/batch_silver_to_gold.py
    python transformation/batch_silver_to_gold.py --dry-run
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from azure.storage.filedatalake import DataLakeServiceClient

load_dotenv()

# ---------- Config ----------
SILVER_CONTAINER = "silver"
SILVER_PATH = "eia-batch/part-00000-07ffde1f-959a-49ef-bc13-37b6a9986dfe-c000.snappy.parquet"
GOLD_CONTAINER = "gold"
GOLD_PATH_MONTHLY = "eia-batch/consumption_monthly.parquet"
GOLD_PATH_SUMMARY = "eia-batch/consumption_summary.parquet"

NE_STATES = ["CT", "MA", "ME", "NH", "RI", "VT"]
SECTORS = ["commercial", "industrial", "residential"]


# ---------- ADLS ----------
def get_client() -> DataLakeServiceClient:
    conn_str = os.environ.get("ADLS_CONNECTION_STRING")
    if not conn_str:
        sys.exit("ERROR: ADLS_CONNECTION_STRING not set in .env")
    return DataLakeServiceClient.from_connection_string(conn_str)


def read_silver(client: DataLakeServiceClient) -> pd.DataFrame:
    print("Reading EIA Silver from ADLS...")
    fs = client.get_file_system_client(SILVER_CONTAINER)
    data = fs.get_file_client(SILVER_PATH).download_file().readall()
    df = pd.read_parquet(io.BytesIO(data))
    print(f"  Loaded {len(df)} rows")
    return df


def write_gold(client: DataLakeServiceClient,
               df: pd.DataFrame, path: str) -> None:
    print(f"  Writing to gold/{path}...")
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    fs = client.get_file_system_client(GOLD_CONTAINER)
    fs.get_file_client(path).upload_data(buf.read(), overwrite=True)
    print(f"  Written {len(df)} rows")


# ---------- Cleaning ----------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning Silver data...")
    df = df.copy()

    # Cast numeric columns
    df["sales_million_kwh"] = pd.to_numeric(df["sales"], errors="coerce")
    df["customers_count"] = pd.to_numeric(df["customers"], errors="coerce")

    # Parse period to datetime
    df["period_dt"] = pd.to_datetime(df["period"] + "-01", format="%Y-%m-%d")
    df["year"] = df["period_dt"].dt.year
    df["month"] = df["period_dt"].dt.month
    df["quarter"] = df["period_dt"].dt.quarter

    # Season assignment (meteorological)
    def get_season(m):
        if m in [12, 1, 2]: return "winter"
        elif m in [3, 4, 5]: return "spring"
        elif m in [6, 7, 8]: return "summer"
        else: return "fall"

    df["season"] = df["month"].apply(get_season)

    # Normalize sales per customer (removes population growth effect)
    df["sales_per_customer_kwh"] = (
        (df["sales_million_kwh"] * 1_000_000) / df["customers_count"]
    ).replace([np.inf, -np.inf], np.nan)

    # Drop rows with null sales
    before = len(df)
    df = df.dropna(subset=["sales_million_kwh"]).reset_index(drop=True)
    print(f"  Dropped {before - len(df)} rows with null sales")
    print(f"  Clean rows: {len(df)}")
    print(f"  Period: {df['period_dt'].min().date()} → {df['period_dt'].max().date()}")

    return df


# ---------- Feature engineering ----------
def build_monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gold Table 1: cleaned monthly time series.
    One row per state × sector × month.
    """
    print("\nBuilding monthly Gold table...")

    keep_cols = [
        "period", "period_dt", "year", "month", "quarter", "season",
        "stateid", "stateDescription", "sectorid", "sectorName",
        "sales_million_kwh", "customers_count", "sales_per_customer_kwh",
    ]
    monthly = df[keep_cols].copy()

    # Year-over-year growth per state/sector
    monthly = monthly.sort_values(["stateid", "sectorName", "period_dt"])
    monthly["sales_yoy_growth_pct"] = (
        monthly.groupby(["stateid", "sectorName"])["sales_million_kwh"]
        .pct_change(periods=12) * 100
    )

    # Rolling 12-month average (smooths seasonality for trend analysis)
    monthly["sales_rolling_12m_avg"] = (
        monthly.groupby(["stateid", "sectorName"])["sales_million_kwh"]
        .transform(lambda x: x.rolling(12, min_periods=6).mean())
    )

    print(f"  Monthly table shape: {monthly.shape}")
    return monthly


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gold Table 2: aggregated summary features for Predictions 1-3.
    """
    print("\nBuilding summary Gold table...")

    # --- Prediction 1: Seasonal patterns ---
    # Summer peak vs winter baseline by state
    seasonal = df.groupby(["stateid", "season", "sectorName"]).agg(
        avg_sales_million_kwh=("sales_million_kwh", "mean"),
        avg_customers=("customers_count", "mean"),
        avg_sales_per_customer=("sales_per_customer_kwh", "mean"),
        months_count=("period", "count"),
    ).reset_index()

    # Summer/winter ratio per state (key metric for Prediction 1)
    summer = seasonal[seasonal["season"] == "summer"].set_index(
        ["stateid", "sectorName"])["avg_sales_million_kwh"]
    winter = seasonal[seasonal["season"] == "winter"].set_index(
        ["stateid", "sectorName"])["avg_sales_million_kwh"]
    summer_winter_ratio = (summer / winter).reset_index()
    summer_winter_ratio.columns = ["stateid", "sectorName", "summer_winter_ratio"]

    # --- Prediction 2: Industrial growth 2022-2026 ---
    industrial = df[
        (df["sectorName"] == "industrial") &
        (df["year"] >= 2022)
    ].groupby(["stateid", "year"]).agg(
        total_industrial_sales=("sales_million_kwh", "sum"),
        avg_industrial_customers=("customers_count", "mean"),
    ).reset_index()

    # CAGR 2022 → 2025 per state
    ind_2022 = industrial[industrial["year"] == 2022].set_index(
        "stateid")["total_industrial_sales"]
    ind_2025 = industrial[industrial["year"] == 2025].set_index(
        "stateid")["total_industrial_sales"]
    cagr = ((ind_2025 / ind_2022) ** (1/3) - 1) * 100
    cagr = cagr.reset_index()
    cagr.columns = ["stateid", "industrial_cagr_2022_2025_pct"]

    # --- Prediction 3: Monthly operational patterns ---
    # All-sector NE total by month number (proxy for weekly patterns)
    monthly_pattern = df.groupby(["month", "sectorName"]).agg(
        avg_sales=("sales_million_kwh", "mean"),
    ).reset_index()
    monthly_pattern["month_name"] = pd.to_datetime(
        monthly_pattern["month"], format="%m").dt.strftime("%b")

    print(f"  Seasonal summary: {len(seasonal)} rows")
    print(f"  Summer/winter ratios: {len(summer_winter_ratio)} state/sector pairs")
    print(f"  Industrial CAGR: {len(cagr)} states")

    return seasonal, summer_winter_ratio, cagr, monthly_pattern


# ---------- EDA Plots ----------
def plot_eda(df: pd.DataFrame, monthly: pd.DataFrame,
             summer_winter: pd.DataFrame, cagr: pd.DataFrame,
             monthly_pattern: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EIA New England Electricity Consumption — EDA\n"
                 "Team 08 Grid Stress Monitoring (Predictions 1-3)",
                 fontsize=13)

    # --- Plot 1: Total NE sales over time by sector (Prediction 1 + 2) ---
    ax = axes[0, 0]
    ne_total = df.groupby(["period_dt", "sectorName"])["sales_million_kwh"].sum().reset_index()
    for sector in SECTORS:
        s = ne_total[ne_total["sectorName"] == sector]
        ax.plot(s["period_dt"], s["sales_million_kwh"],
                label=sector.capitalize(), linewidth=1.5)
    ax.set_title("New England Total Sales by Sector\n(Jan 2016 – Dec 2025)")
    ax.set_ylabel("Sales (million kWh)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    # --- Plot 2: Summer/winter ratio by state (Prediction 1) ---
    ax = axes[0, 1]
    residential_sw = summer_winter[
        summer_winter["sectorName"] == "residential"
    ].sort_values("summer_winter_ratio", ascending=False)
    commercial_sw = summer_winter[
        summer_winter["sectorName"] == "commercial"
    ].sort_values("summer_winter_ratio", ascending=False)
    x = np.arange(len(NE_STATES))
    w = 0.35
    res_vals = [residential_sw[residential_sw["stateid"] == s]["summer_winter_ratio"].values[0]
                if s in residential_sw["stateid"].values else 0 for s in NE_STATES]
    com_vals = [commercial_sw[commercial_sw["stateid"] == s]["summer_winter_ratio"].values[0]
                if s in commercial_sw["stateid"].values else 0 for s in NE_STATES]
    ax.bar(x - w/2, res_vals, w, label="Residential", color="steelblue")
    ax.bar(x + w/2, com_vals, w, label="Commercial", color="orange", alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle=":", label="No seasonal diff")
    ax.set_xticks(x)
    ax.set_xticklabels(NE_STATES)
    ax.set_title("Summer/Winter Sales Ratio by State\n(>1 = cooling-driven summer peak)")
    ax.set_ylabel("Ratio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # --- Plot 3: Industrial sales trend 2016-2025 (Prediction 2) ---
    ax = axes[1, 0]
    ind = df[df["sectorName"] == "industrial"].groupby(
        ["period_dt", "stateid"])["sales_million_kwh"].sum().reset_index()
    for state in ["MA", "CT"]:
        s = ind[ind["stateid"] == state]
        ax.plot(s["period_dt"], s["sales_million_kwh"],
                label=state, linewidth=1.5)
    ne_ind = df[df["sectorName"] == "industrial"].groupby(
        "period_dt")["sales_million_kwh"].sum().reset_index()
    ax2 = ax.twinx()
    ax2.plot(ne_ind["period_dt"], ne_ind["sales_million_kwh"],
             label="NE Total", color="black", linewidth=2, linestyle="--")
    ax2.set_ylabel("NE Total (million kWh)")
    ax.set_title("Industrial Electricity Sales\n(Prediction 2: AI/data center growth proxy)")
    ax.set_ylabel("Sales (million kWh)")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    # --- Plot 4: Monthly pattern (Prediction 3) ---
    ax = axes[1, 1]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    for sector in SECTORS:
        s = monthly_pattern[monthly_pattern["sectorName"] == sector].sort_values("month")
        ax.plot(s["month"], s["avg_sales"],
                label=sector.capitalize(), marker="o", linewidth=1.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, fontsize=8)
    ax.set_title("Average Monthly Sales Pattern\n(Prediction 3: operational cycles)")
    ax.set_ylabel("Avg Sales (million kWh)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("notebooks/eia_eda.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("  Chart saved to notebooks/eia_eda.png")


# ---------- Print findings ----------
def print_findings(df, summer_winter, cagr, monthly_pattern):
    print("\n" + "=" * 60)
    print("EIA Gold Layer — Key Findings")
    print("=" * 60)

    # Prediction 1
    print("\nPrediction 1 — Seasonal Patterns:")
    print("Summer/Winter sales ratio (residential, all NE states):")
    res = summer_winter[summer_winter["sectorName"] == "residential"]
    for _, row in res.sort_values("summer_winter_ratio", ascending=False).iterrows():
        print(f"  {row['stateid']}: {row['summer_winter_ratio']:.3f}x")

    # Prediction 2
    print("\nPrediction 2 — Industrial CAGR 2022-2025:")
    for _, row in cagr.sort_values("industrial_cagr_2022_2025_pct",
                                    ascending=False).iterrows():
        direction = "↑" if row["industrial_cagr_2022_2025_pct"] > 0 else "↓"
        print(f"  {row['stateid']}: {row['industrial_cagr_2022_2025_pct']:+.2f}% {direction}")

    # Prediction 3
    print("\nPrediction 3 — Peak vs Off-peak Monthly Pattern:")
    res_monthly = monthly_pattern[monthly_pattern["sectorName"] == "residential"]
    peak_month = res_monthly.loc[res_monthly["avg_sales"].idxmax(), "month_name"]
    trough_month = res_monthly.loc[res_monthly["avg_sales"].idxmin(), "month_name"]
    peak_val = res_monthly["avg_sales"].max()
    trough_val = res_monthly["avg_sales"].min()
    print(f"  Residential peak:   {peak_month} ({peak_val:.1f}M kWh avg)")
    print(f"  Residential trough: {trough_month} ({trough_val:.1f}M kWh avg)")
    print(f"  Peak/trough ratio:  {peak_val/trough_val:.2f}x")
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
    raw = read_silver(client)

    # Clean
    df = clean(raw)

    # Build Gold tables
    monthly = build_monthly_table(df)
    seasonal, summer_winter, cagr, monthly_pattern = build_summary_table(df)

    # Print findings
    print_findings(df, summer_winter, cagr, monthly_pattern)

    # Plot EDA
    print("\nGenerating EDA charts...")
    plot_eda(df, monthly, summer_winter, cagr, monthly_pattern)

    # Write to ADLS
    if args.dry_run:
        print("\nDRY RUN — skipping ADLS write")
    else:
        print("\nWriting Gold tables to ADLS...")
        write_gold(client, monthly, GOLD_PATH_MONTHLY)
        write_gold(client, seasonal, GOLD_PATH_SUMMARY)
        print(f"\nGold tables written:")
        print(f"  gold/{GOLD_PATH_MONTHLY}")
        print(f"  gold/{GOLD_PATH_SUMMARY}")

    print("\nDone.")


if __name__ == "__main__":
    main()