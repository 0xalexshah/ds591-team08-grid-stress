# Team 08 — Grid Stress Monitoring (ISO New England)
**DS591/562 Engineering for Big Data Workloads — Boston University, Spring 2026**

Team: Alexander Shah, Aijia Zhang, Samantha Gibson, Jiaxi Li

## Project Overview
Data engineering pipeline monitoring real-time grid stress in ISO New England,
combining 5-minute streaming demand data with multi-year historical consumption
data to predict load patterns and detect grid stress events.

## Architecture
Bronze (raw ingestion) → Silver (cleaned + transformed) → Gold (feature matrix for modeling)
- **Batch:** Azure Data Factory pipelines ingesting ISO-NE historical + EIA data
- **Streaming:** Python simulator replaying ISO-NE 5-min demand CSVs at real cadence

## Repository Structure
- `ingestion/` — `stream_simulator.py`, `stream_processor.py`
- `transformation/` — `silver_to_gold.py` (Milestone 4)
- `modeling/` — load forecasting model (Milestone 4)
- `notebooks/` — EDA and analysis notebooks
- `dashboard/` — Power BI files (Milestone 4)

## Setup
1. Clone this repo
2. Copy `.env.example` to `.env` and fill in your ADLS connection string
3. Install dependencies: `pip install -r requirements.txt`

### Running the streaming pipeline
```bash
# Replay ISO-NE 5-min demand CSVs to ADLS Bronze (dry run — no writes)
python ingestion/stream_simulator.py --input-dir ./iso_ne_csvs --max-rows 24 --dry-run

# Process Bronze → Silver (single pass)
python ingestion/stream_processor.py --once

# Process Bronze → Silver (poll continuously every 60s)
python ingestion/stream_processor.py
```

## Milestones
- ✅ Milestone 1: Architecture & proposal
- ✅ Milestone 2: ADF batch ingestion pipelines
- ✅ Milestone 3: Streaming simulation + stream processing + EDA
- 🔄 Milestone 4: Gold layer, modeling, dashboard, final report
