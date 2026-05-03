"""
ISO-NE Load Forecasting Model — Milestone 4

Reads the Gold feature matrix from ADLS and trains four models:
  1. Persistence baseline (next load = current load)
  2. Linear Regression
  3. Random Forest
  4. XGBoost

Evaluates each on:
  - MAPE and RMSE (overall + season-stratified)
  - Multi-horizon forecasting (t+1, t+3, t+6, t+12 intervals = 5/15/30/60 min)
  - Stress detection F1 score (Prediction 4)
  - Stress event deep-dive analysis

Usage:
    python modeling/load_forecast.py
    python modeling/load_forecast.py --dry-run
    python modeling/load_forecast.py --skip-xgb   # if XGBoost not installed
"""

import argparse
import io
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv
from azure.storage.filedatalake import DataLakeServiceClient
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_percentage_error,
    root_mean_squared_error,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

load_dotenv()

# ---------- Config ----------
GOLD_CONTAINER = "gold"
GOLD_PATH = "iso-ne-stream/demand_features.parquet"

FEATURE_COLS = [
    "load_lag_1", "load_lag_2", "load_lag_3",
    "load_lag_6", "load_lag_12",
    "rolling_3_avg_mw", "rolling_12_avg_mw",
    "load_delta_mw",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "is_weekend", "is_peak_hour",
    "season_int",
    "is_overnight",   # new feature
]

TARGET_COL = "target_load_mw"
STRESS_COL = "stress_flag"
TRAIN_RATIO = 0.80

# Forecast horizons in intervals (1 interval = 5 minutes)
HORIZONS = {
    "t+5min":  1,
    "t+15min": 3,
    "t+30min": 6,
    "t+60min": 12,
}

STRESS_THRESHOLD_PCT = 5.0


# ---------- ADLS ----------
def get_client() -> DataLakeServiceClient:
    conn_str = os.environ.get("ADLS_CONNECTION_STRING")
    if not conn_str:
        sys.exit("ERROR: ADLS_CONNECTION_STRING not set in .env")
    return DataLakeServiceClient.from_connection_string(conn_str)


def read_gold(client: DataLakeServiceClient) -> pd.DataFrame:
    print("Reading Gold from ADLS...")
    fs = client.get_file_system_client(GOLD_CONTAINER)
    data = fs.get_file_client(GOLD_PATH).download_file().readall()
    df = pd.read_parquet(io.BytesIO(data))
    print(f"  Loaded {len(df)} rows")
    return df


# ---------- Preprocessing ----------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features not in Gold layer yet."""
    df = df.copy()
    # Overnight flag — model struggles at demand troughs
    df["is_overnight"] = df["hour_of_day"].between(0, 5).astype(int)
    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """Clean, split, return train/test arrays."""
    df = add_features(df)

    available_features = [f for f in FEATURE_COLS if f in df.columns]
    required = available_features + [TARGET_COL, "event_time", STRESS_COL,
                                      "season", "total_load_mw",
                                      "rolling_12_avg_mw", "asset_related_load_mw",
                                      "load_lag_1", "load_delta_mw"]
    required = list(set(required))
    df = df[required].dropna(subset=available_features + [TARGET_COL]).reset_index(drop=True)
    print(f"  After dropping nulls: {len(df)} rows")

    split_idx = int(len(df) * TRAIN_RATIO)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train)} rows "
          f"({train['event_time'].min().date()} → {train['event_time'].max().date()})")
    print(f"  Test:  {len(test)} rows "
          f"({test['event_time'].min().date()} → {test['event_time'].max().date()})")

    X_train = train[available_features].values
    y_train = train[TARGET_COL].values
    X_test = test[available_features].values
    y_test = test[TARGET_COL].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train, y_train,
        X_test, y_test,
        X_train_scaled, X_test_scaled,
        train, test, scaler,
        available_features
    )


# ---------- Metrics ----------
def mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100


def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def season_mape(y_true, y_pred, seasons):
    """MAPE broken down by season."""
    results = {}
    for season in ["summer", "fall", "winter", "spring"]:
        mask = seasons == season
        if mask.sum() > 10:
            results[season] = mape(y_true[mask], y_pred[mask])
    return results


# ---------- Stress detection ----------
def evaluate_stress(test_df, y_pred, model_name, threshold=STRESS_THRESHOLD_PCT):
    rolling_avg = test_df["rolling_12_avg_mw"].values
    pred_deviation = np.abs((y_pred - rolling_avg) / rolling_avg) * 100
    pred_stress = pred_deviation > threshold
    actual_stress = test_df[STRESS_COL].astype(bool).values

    if actual_stress.sum() == 0:
        return {"f1": 0, "precision": 0, "recall": 0,
                "actual": 0, "predicted": 0}

    f1 = f1_score(actual_stress, pred_stress, zero_division=0)
    prec = precision_score(actual_stress, pred_stress, zero_division=0)
    rec = recall_score(actual_stress, pred_stress, zero_division=0)

    print(f"  [{model_name}] Stress: actual={actual_stress.sum()} "
          f"predicted={pred_stress.sum()} "
          f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    return {"f1": f1, "precision": prec, "recall": rec,
            "actual": actual_stress.sum(), "predicted": pred_stress.sum()}


# ---------- Multi-horizon forecasting ----------
def multi_horizon_eval(train_df, test_df, feature_cols, model_class, model_kwargs,
                       scaler=None, model_name="Model"):
    """
    Train separate models for each forecast horizon.
    For horizon h, target = load at t+h intervals.
    """
    print(f"  Multi-horizon eval for {model_name}...")
    results = {}

    for horizon_name, h in HORIZONS.items():
        # Build target for this horizon
        train_target = train_df["total_load_mw"].shift(-h)
        test_target = test_df["total_load_mw"].shift(-h)

        train_mask = train_target.notna()
        test_mask = test_target.notna()

        X_tr = train_df.loc[train_mask, feature_cols].values
        y_tr = train_target[train_mask].values
        X_te = test_df.loc[test_mask, feature_cols].values
        y_te = test_target[test_mask].values

        if scaler:
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        model = model_class(**model_kwargs)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        results[horizon_name] = {
            "mape": mape(y_te, y_pred),
            "rmse": rmse(y_te, y_pred),
        }

    return results


# ---------- Stress event deep dive ----------
def stress_deep_dive(test_df, y_pred_rf, feature_cols):
    """
    Identify and characterize actual stress events in the test set.
    Returns a DataFrame of stress event clusters with context.
    """
    test_df = test_df.copy()
    test_df["y_pred_rf"] = y_pred_rf

    actual_stress = test_df[STRESS_COL].astype(bool)
    stress_events = test_df[actual_stress].copy()

    if len(stress_events) == 0:
        print("  No stress events in test set")
        return pd.DataFrame()

    # Cluster consecutive stress events into episodes
    stress_events = stress_events.sort_values("event_time").reset_index(drop=True)
    stress_events["gap"] = (
        stress_events["event_time"].diff().dt.total_seconds() / 60
    ).fillna(0)
    stress_events["episode"] = (stress_events["gap"] > 15).cumsum()

    episodes = stress_events.groupby("episode").agg(
        start_time=("event_time", "min"),
        end_time=("event_time", "max"),
        duration_min=("event_time", lambda x: (x.max() - x.min()).total_seconds() / 60),
        peak_load=("total_load_mw", "max"),
        min_load=("total_load_mw", "min"),
        avg_asset_load=("asset_related_load_mw", "mean"),
        max_delta=("load_delta_mw", "max"),
        count=("event_time", "count"),
    ).reset_index(drop=True)

    print(f"\n  Stress Event Episodes in Test Set: {len(episodes)}")
    print(f"  Total stress intervals: {len(stress_events)}")
    print(f"  Avg episode duration: {episodes['duration_min'].mean():.1f} min")
    print(f"  Max asset-related load during stress: {episodes['avg_asset_load'].max():.1f} MW")

    return episodes, stress_events


# ---------- Models ----------
def persistence_baseline(y_test, test_df):
    return test_df["load_lag_1"].values


def train_lr(X_train_scaled, y_train, X_test_scaled):
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled), model


def train_rf(X_train, y_train, X_test):
    model = RandomForestRegressor(
        n_estimators=100, max_depth=15,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model.predict(X_test), model


def train_xgb(X_train, y_train, X_test):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  XGBoost not installed — run: pip install xgboost")
        return None, None
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)
    return model.predict(X_test), model


# ---------- Plotting ----------
def plot_all(test_df, predictions, stress_results,
             season_mapes, horizon_results, episodes, stress_events):

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("ISO-NE Load Forecasting — Team 08 Grid Stress Monitoring\n"
                 "Milestone 4 Model Results", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    colors = {
        "Persistence": "crimson",
        "Linear Regression": "steelblue",
        "Random Forest": "green",
        "XGBoost": "darkorange",
    }

    # --- Plot 1: Actual vs Predicted (1 day) ---
    ax1 = fig.add_subplot(gs[0, :])
    sample = test_df.iloc[:288].copy()
    ax1.plot(sample["event_time"], sample[TARGET_COL],
             label="Actual", linewidth=2, color="black")
    for name, preds in predictions.items():
        ax1.plot(sample["event_time"], preds[:288],
                 label=name, linewidth=1.2, linestyle="--",
                 color=colors.get(name, "gray"), alpha=0.85)
    ax1.set_title("Actual vs Predicted Load — First Test Day (5-min resolution)")
    ax1.set_ylabel("Load (MW)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis="x", rotation=30)

    # --- Plot 2: MAPE bar chart ---
    ax2 = fig.add_subplot(gs[1, 0])
    model_names = list(predictions.keys())
    mapes = [mape(test_df[TARGET_COL].values, p) for p in predictions.values()]
    bars = ax2.bar(model_names, mapes,
                   color=[colors.get(n, "gray") for n in model_names])
    ax2.axhline(2.0, color="orange", linestyle=":",
                label="ISO-NE day-ahead target (~2%)")
    ax2.set_title("MAPE by Model")
    ax2.set_ylabel("MAPE (%)")
    ax2.legend(fontsize=8)
    for bar, val in zip(bars, mapes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}%", ha="center", fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    # --- Plot 3: Season-stratified MAPE ---
    ax3 = fig.add_subplot(gs[1, 1])
    seasons = ["summer", "fall", "winter", "spring"]
    x = np.arange(len(seasons))
    width = 0.2
    for i, (name, smapes) in enumerate(season_mapes.items()):
        vals = [smapes.get(s, 0) for s in seasons]
        ax3.bar(x + i * width, vals, width,
                label=name, color=colors.get(name, "gray"), alpha=0.85)
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels([s.capitalize() for s in seasons])
    ax3.set_title("MAPE by Season")
    ax3.set_ylabel("MAPE (%)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3, axis="y")

    # --- Plot 4: Multi-horizon MAPE ---
    ax4 = fig.add_subplot(gs[2, 0])
    for name, h_results in horizon_results.items():
        horizons = list(h_results.keys())
        h_mapes = [h_results[h]["mape"] for h in horizons]
        ax4.plot(horizons, h_mapes, marker="o",
                 label=name, color=colors.get(name, "gray"), linewidth=1.5)
    ax4.set_title("MAPE vs Forecast Horizon\n(how accuracy degrades further ahead)")
    ax4.set_ylabel("MAPE (%)")
    ax4.set_xlabel("Forecast Horizon")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    ax4.tick_params(axis="x", rotation=15)

    # --- Plot 5: Stress detection comparison ---
    ax5 = fig.add_subplot(gs[2, 1])
    stress_models = list(stress_results.keys())
    f1s = [stress_results[m]["f1"] for m in stress_models]
    precs = [stress_results[m]["precision"] for m in stress_models]
    recs = [stress_results[m]["recall"] for m in stress_models]
    x = np.arange(len(stress_models))
    w = 0.25
    ax5.bar(x - w, f1s, w, label="F1", color="steelblue")
    ax5.bar(x, precs, w, label="Precision", color="green", alpha=0.8)
    ax5.bar(x + w, recs, w, label="Recall", color="crimson", alpha=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(stress_models, fontsize=8)
    ax5.set_title("Stress Detection — Prediction 4\n(F1 / Precision / Recall)")
    ax5.set_ylabel("Score")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3, axis="y")
    ax5.set_ylim(0, 1.1)

    # --- Plot 6: Stress event timeline ---
    ax6 = fig.add_subplot(gs[3, :])
    ax6.plot(test_df["event_time"], test_df["total_load_mw"],
             color="steelblue", linewidth=0.8, label="Total Load (MW)", alpha=0.7)
    if len(stress_events) > 0:
        ax6.scatter(stress_events["event_time"],
                    stress_events["total_load_mw"],
                    color="crimson", s=8, zorder=5,
                    label=f"Stress events ({len(stress_events)} intervals)")
    if len(episodes) > 0:
        for _, ep in episodes.iterrows():
            ax6.axvspan(ep["start_time"], ep["end_time"],
                        alpha=0.15, color="red")
    ax6.set_title("Stress Event Timeline — Full Test Period\n"
                  "(red shading = stress episodes, red dots = individual stress intervals)")
    ax6.set_ylabel("Load (MW)")
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)
    ax6.tick_params(axis="x", rotation=30)

    plt.savefig("notebooks/model_results_v2.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("  Chart saved to notebooks/model_results_v2.png")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and show split info but skip training")
    parser.add_argument("--skip-xgb", action="store_true",
                        help="Skip XGBoost (if not installed)")
    args = parser.parse_args()

    client = get_client()
    df = read_gold(client)

    print("\nPreparing data...")
    (X_train, y_train,
     X_test, y_test,
     X_train_scaled, X_test_scaled,
     train_df, test_df, scaler,
     feature_cols) = prepare_data(df)

    if args.dry_run:
        print("\nDRY RUN — skipping model training")
        return

    predictions = {}
    stress_results = {}
    season_mapes_all = {}
    horizon_results = {}
    test_seasons = test_df["season"].values

    # ---- 1: Persistence ----
    print("\n[1/4] Persistence Baseline...")
    y_persist = persistence_baseline(y_test, test_df)
    print(f"  MAPE: {mape(y_test, y_persist):.2f}%  "
          f"RMSE: {rmse(y_test, y_persist):.1f} MW")
    stress_results["Persistence"] = evaluate_stress(test_df, y_persist, "Persistence")
    season_mapes_all["Persistence"] = season_mape(y_test, y_persist, test_seasons)
    predictions["Persistence"] = y_persist
    # Persistence multi-horizon: MAPE just increases as lag grows
    horizon_results["Persistence"] = {}
    for h_name, h in HORIZONS.items():
        y_h = test_df["total_load_mw"].shift(-h).dropna().values
        y_lag = test_df["load_lag_1"].iloc[:len(y_h)].values
        horizon_results["Persistence"][h_name] = {
            "mape": mape(y_h, y_lag),
            "rmse": rmse(y_h, y_lag)
        }

    # ---- 2: Linear Regression ----
    print("\n[2/4] Linear Regression...")
    y_lr, lr_model = train_lr(X_train_scaled, y_train, X_test_scaled)
    print(f"  MAPE: {mape(y_test, y_lr):.2f}%  "
          f"RMSE: {rmse(y_test, y_lr):.1f} MW")
    stress_results["Linear Regression"] = evaluate_stress(test_df, y_lr, "Linear Regression")
    season_mapes_all["Linear Regression"] = season_mape(y_test, y_lr, test_seasons)
    predictions["Linear Regression"] = y_lr
    horizon_results["Linear Regression"] = multi_horizon_eval(
        train_df, test_df, feature_cols,
        LinearRegression, {},
        scaler=StandardScaler(), model_name="Linear Regression"
    )

    # ---- 3: Random Forest ----
    print("\n[3/4] Random Forest (may take 1-2 min)...")
    y_rf, rf_model = train_rf(X_train, y_train, X_test)
    print(f"  MAPE: {mape(y_test, y_rf):.2f}%  "
          f"RMSE: {rmse(y_test, y_rf):.1f} MW")
    stress_results["Random Forest"] = evaluate_stress(test_df, y_rf, "Random Forest")
    season_mapes_all["Random Forest"] = season_mape(y_test, y_rf, test_seasons)
    predictions["Random Forest"] = y_rf
    horizon_results["Random Forest"] = multi_horizon_eval(
        train_df, test_df, feature_cols,
        RandomForestRegressor,
        {"n_estimators": 50, "max_depth": 10, "random_state": 42, "n_jobs": -1},
        model_name="Random Forest"
    )

    # ---- 4: XGBoost ----
    if not args.skip_xgb:
        print("\n[4/4] XGBoost (may take 1-2 min)...")
        y_xgb, xgb_model = train_xgb(X_train, y_train, X_test)
        if y_xgb is not None:
            print(f"  MAPE: {mape(y_test, y_xgb):.2f}%  "
                  f"RMSE: {rmse(y_test, y_xgb):.1f} MW")
            stress_results["XGBoost"] = evaluate_stress(test_df, y_xgb, "XGBoost")
            season_mapes_all["XGBoost"] = season_mape(y_test, y_xgb, test_seasons)
            predictions["XGBoost"] = y_xgb
            horizon_results["XGBoost"] = multi_horizon_eval(
                train_df, test_df, feature_cols,
                __import__("xgboost").XGBRegressor,
                {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
                 "random_state": 42, "n_jobs": -1, "verbosity": 0},
                model_name="XGBoost"
            )
    else:
        print("\n[4/4] XGBoost skipped")

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("Model Comparison Summary")
    print("=" * 70)
    print(f"{'Model':<20} {'MAPE':>7} {'RMSE':>10}  {'Stress F1':>10}  {'vs Baseline'}")
    print("-" * 70)
    baseline_mape = mape(y_test, y_persist)
    for name, preds in predictions.items():
        m = mape(y_test, preds)
        r = rmse(y_test, preds)
        f1 = stress_results[name]["f1"]
        vs = f"({((m - baseline_mape)/baseline_mape*100):+.1f}%)" if name != "Persistence" else "(baseline)"
        print(f"{name:<20} {m:>6.2f}% {r:>10.1f} MW {f1:>10.3f}  {vs}")
    print(f"\nISO-NE day-ahead forecast target: ~2.0% MAPE")
    print("=" * 70)

    # ---- Season breakdown ----
    print("\nMAPE by Season:")
    print(f"{'Model':<20} {'Summer':>8} {'Fall':>8} {'Winter':>8} {'Spring':>8}")
    print("-" * 56)
    for name, smapes in season_mapes_all.items():
        row = f"{name:<20}"
        for s in ["summer", "fall", "winter", "spring"]:
            row += f" {smapes.get(s, 0):>7.2f}%"
        print(row)

    # ---- Multi-horizon breakdown ----
    print("\nMAPE by Forecast Horizon:")
    print(f"{'Model':<20} {'t+5min':>8} {'t+15min':>9} {'t+30min':>9} {'t+60min':>9}")
    print("-" * 58)
    for name, h_res in horizon_results.items():
        row = f"{name:<20}"
        for h in ["t+5min", "t+15min", "t+30min", "t+60min"]:
            row += f" {h_res.get(h, {}).get('mape', 0):>8.2f}%"
        print(row)

    # ---- Stress event deep dive ----
    print("\nStress Event Deep Dive...")
    best_pred = predictions.get("XGBoost", predictions.get("Random Forest"))
    episodes, stress_events = stress_deep_dive(test_df, best_pred, feature_cols)
    if len(episodes) > 0:
        print("\nTop 5 stress episodes by duration:")
        print(episodes.nlargest(5, "duration_min")[
            ["start_time", "duration_min", "peak_load",
             "avg_asset_load", "count"]
        ].to_string(index=False))

    # ---- Plot ----
    print("\nGenerating charts...")
    plot_all(test_df, predictions, stress_results,
             season_mapes_all, horizon_results, episodes, stress_events)

    print("\nDone. Results saved to notebooks/model_results_v2.png")


if __name__ == "__main__":
    main()