# MineGuard — AI-Powered Predictive Maintenance for Mining Equipment

> An end-to-end industrial ML system that monitors heavy mining equipment in real time,
> predicts failures before they happen, and estimates remaining useful life —
> built on real sensor datasets from hydraulic systems, turbofan engines, and bearings.

---

## Why this project exists

Unplanned equipment failure in underground mining costs upwards of **$50,000/hour** in lost
production. This project simulates and extends a production-grade predictive maintenance
platform for a fleet of haul trucks, hydraulic drill rigs, and LHD loaders — the exact
equipment I worked with as a Radiation Technician / Mining Engineer at Wireline Africa.

The system combines three real industrial datasets with a physics-based sensor simulator
to produce a full MLOps pipeline: from raw sensor ingestion to a live dashboard.

---

## Architecture

```
Data sources
  ├── UCI Hydraulic System (real)      → Failure Classifier   (Random Forest / XGBoost)
  ├── NASA CMAPSS (real)               → RUL Predictor        (LSTM + Attention)
  ├── CWRU Bearing Fault (real)        → Anomaly Detector     (Isolation Forest / AE)
  └── MineGuard Simulator (synthetic)  → Live dashboard stream

                         ↓
                   FastAPI Backend
                  /predict  /rul  /anomaly
                         ↓
            ┌────────────┼────────────┐
       SQLite log    Streamlit     MLflow
       (history)    Dashboard    (experiments)
                         ↓
               Docker + GitHub Actions
```

---

## Datasets

| Dataset | Source | Size | Task | Mining relevance |
|---|---|---|---|---|
| UCI Hydraulic | [archive.uci.edu/dataset/447](https://archive.uci.edu/dataset/447) | ~7 MB | Multi-output classification | Hydraulic systems power every drill, LHD, and roof bolter |
| NASA CMAPSS | [data.nasa.gov](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) | ~4 MB | RUL regression | Engine degradation maps to haul truck powertrains |
| CWRU Bearing | [engineering.case.edu/bearingdatacenter](https://engineering.case.edu/bearingdatacenter) | ~70 MB | Fault classification | Bearing failures = #1 cause of mining downtime |
| MineGuard Sim | `simulator/sensor_generator.py` | Configurable | All three | Physics-based, themed to actual mine equipment |

---

## Project structure

```
mineguard/
├── data/
│   ├── raw/
│   │   ├── hydraulic/     ← UCI dataset (download with make download-hydraulic)
│   │   ├── cmapss/        ← NASA CMAPSS (make download-cmapss)
│   │   └── bearing/       ← CWRU bearing (make download-bearing)
│   ├── processed/         ← cleaned, feature-engineered DataFrames
│   └── simulated/         ← output of sensor_generator.py
│
├── simulator/
│   └── sensor_generator.py   ← physics-based sensor stream generator
│
├── models/
│   ├── failure_classifier/   ← hydraulic component health classifier
│   ├── rul_predictor/        ← LSTM remaining useful life model
│   └── anomaly_detector/     ← bearing fault + anomaly detection
│
├── api/
│   └── main.py               ← FastAPI: /predict /rul /anomaly /stream
│
├── dashboard/
│   └── app.py                ← Streamlit digital twin dashboard
│
├── mlops/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── notebooks/
│   ├── 01_hydraulic_eda.ipynb
│   ├── 02_cmapss_rul.ipynb
│   └── 03_bearing_fault.ipynb
│
├── tests/
├── scripts/
│   └── download_datasets.py  ← automated dataset downloader
│
├── Makefile
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/takumusimwa-tr/mineguard.git
cd mineguard
make setup
```

### 2. Download datasets

```bash
make download
```

This runs `scripts/download_datasets.py` which fetches:
- UCI Hydraulic via `ucimlrepo` Python client (no login required)
- NASA CMAPSS via direct download
- CWRU Bearing via direct download (or Kaggle mirror — see note below)

> **Bearing dataset note:** If the CWRU server is slow, use the Kaggle mirror:
> ```bash
> pip install kaggle
> # place your kaggle.json API key in ~/.kaggle/
> kaggle datasets download brjapon/gearbox-fault-diagnosis
> unzip gearbox-fault-diagnosis.zip -d data/raw/bearing/
> ```

### 3. Generate simulated data

```bash
make simulate
```

Generates a full run-to-failure dataset for 200 synthetic equipment units
(~650K rows) with three equipment types and five failure modes.

### 4. Run the full stack

```bash
# Terminal 1: API
make api

# Terminal 2: Dashboard
make dashboard

# Terminal 3: MLflow (optional)
make mlflow
```

Or spin everything up with Docker:

```bash
make docker-up
```

---

## Models

### Failure Classifier (hydraulic component health)
- **Dataset:** UCI Hydraulic System
- **Input:** 17 sensor streams (pressure, flow, temperature) over 60-second cycles
- **Output:** Health status of 4 components — cooler, valve, pump, accumulator
- **Architecture:** Random Forest + XGBoost ensemble, multi-output classification

### RUL Predictor (remaining useful life)
- **Dataset:** NASA CMAPSS FD001–FD004
- **Input:** Sliding window of 21 sensor readings over last 30 cycles
- **Output:** Estimated cycles until failure (regression)
- **Architecture:** LSTM with attention mechanism

### Anomaly Detector (bearing fault diagnosis)
- **Dataset:** CWRU Bearing Fault
- **Input:** Vibration signal windows (FFT features)
- **Output:** Fault type (normal / inner race / outer race / ball) + severity
- **Architecture:** Isolation Forest (unsupervised) + CNN classifier (supervised)

---

## Tech stack

| Layer | Technology |
|---|---|
| Data | pandas, numpy, scipy, ucimlrepo |
| Models | scikit-learn, XGBoost, TensorFlow/Keras |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| MLOps | MLflow, Docker, GitHub Actions |
| Testing | pytest, httpx |

---

## Background

Built by Takudzwa Musimwa — Mining Engineering (BSc, Midlands State University) +
Data Science (MSc candidate, Pace University). Field experience includes radiation
monitoring across 50+ well logging operations and coordination of multi-department
safety teams across three mining sites at Wireline Africa (Mozambique, 2023–2025).

---

## License

MIT
