# Team 08: Grid Stress Monitoring, Demand, & Growth Modeling
**DS591 Engineering for Big Data Workloads — Boston University, Spring 2026**

Team: Alexander Shah, Aijia Zhang, Samantha Gibson, Jiaxi Li

## Project Overview
Data pipeline monitoring real-time grid stress in New England, using ISO-NE & EIA data, 
combining 5-minute streaming demand data with multi-year historical consumption
data to predict load patterns and detect grid stress events.

## Architecture
<img width="1503" height="649" alt="team08-591-architecture" src="https://github.com/user-attachments/assets/37453b8b-8905-4e7a-89ea-4351a70c362c" />

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
