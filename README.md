# MineGuard — AI-Powered Predictive Maintenance for Mining Equipment

> An end-to-end industrial ML system that monitors heavy mining equipment in real time,
> predicts failures before they happen, and estimates remaining useful life —
> built on real sensor datasets from hydraulic systems, turbofan engines, and bearings.

---

## Why this project exists

Unplanned equipment failure in underground mining costs upwards of **$50,000/hour** in lost
production. This project builds a production-grade predictive maintenance platform for a
fleet of haul trucks, hydraulic drill rigs, and LHD loaders — the exact equipment I worked
with as a Mining Engineer and Radiation Technician at Wireline Africa (Mozambique, 2023–2025).

The system combines three real industrial datasets with a physics-based sensor simulator
to deliver a full MLOps pipeline: from raw sensor ingestion through trained models to a
live dashboard.

---

## Model performance

| Component | Dataset | Model | Result | Classes |
|---|---|---|---|---|
| Cooler condition | UCI Hydraulic | Gradient Boosting | F1 = **0.9935** | 3 |
| Accumulator health | UCI Hydraulic | Gradient Boosting | F1 = **0.9262** | 4 |
| Pump leakage | UCI Hydraulic | Random Forest + SMOTE | F1 = **1.000** | 3 |
| RUL prediction | NASA CMAPSS FD001 | LSTM (2-layer, 128 hidden) | RMSE = **14.74 cycles** | — |
| Bearing fault | SEU Gearbox vibration | Random Forest | F1 = **1.000** | 2 |

---

## Architecture

```
Data sources
  ├── UCI Hydraulic System (real)        → Failure Classifier  (Gradient Boosting)
  ├── NASA CMAPSS FD001–FD004 (real)     → RUL Predictor       (LSTM 2-layer)
  ├── SEU Gearbox Vibration (real)       → Fault Classifier    (Random Forest)
  └── MineGuard Simulator (synthetic)    → Live dashboard stream

                          ↓
                    FastAPI Backend
                   /predict  /rul  /fault  /stream
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
| UCI Hydraulic | [archive.uci.edu/dataset/447](https://archive.uci.edu/dataset/447) | 556 MB | Multi-output classification | Hydraulic systems power every drill, LHD, and roof bolter |
| NASA CMAPSS | [data.nasa.gov](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) | 45 MB | RUL regression | Engine degradation maps directly to haul truck powertrains |
| SEU Gearbox vibration | [Kaggle](https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis) | 70 MB | Fault classification | Bearing/gear failures are the #1 cause of mining downtime |
| MineGuard Simulator | `simulator/sensor_generator.py` | Configurable | All three | Physics-based, themed to real mine equipment |

---

## Project structure

```
mineguard/
├── data/
│   ├── raw/
│   │   ├── hydraulic/          ← UCI hydraulic dataset (556 MB, 17 sensors)
│   │   ├── cmapss/             ← NASA CMAPSS FD001–FD004 (45 MB)
│   │   └── bearing/            ← SEU gearbox vibration CSVs (70 MB)
│   ├── processed/              ← Feature-engineered DataFrames + figures
│   └── simulated/              ← Output of sensor_generator.py
│
├── simulator/
│   └── sensor_generator.py     ← Physics-based 3-phase degradation simulator
│
├── models/
│   ├── failure_classifier/     ← cooler/accumulator/pump .pkl + imputer + metadata
│   ├── rul_predictor/          ← lstm_rul.pt + scaler.pkl + metadata
│   └── anomaly_detector/       ← bearing_classifier.pkl + metadata
│
├── api/
│   └── main.py                 ← FastAPI: /predict /rul /fault /stream /health
│
├── dashboard/
│   └── app.py                  ← Streamlit digital twin dashboard
│
├── mlops/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── notebooks/
│   ├── 01_hydraulic_eda.ipynb          ← EDA: 17 sensors, 136 features
│   ├── 02_hydraulic_classifier.ipynb   ← Failure classifier: F1 0.926–1.000
│   ├── 03_cmapss_rul.ipynb             ← LSTM RUL: RMSE=14.74 cycles
│   └── 04_bearing_fault.ipynb          ← Fault classifier: F1=1.000
│
├── tests/
├── scripts/
│   └── download_datasets.py    ← Automated dataset downloader
│
├── Makefile
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```powershell
git clone https://github.com/takumusimwa-tr/mineguard.git
cd mineguard
pip install -r requirements.txt
```

### 2. Download datasets

```powershell
python scripts/download_datasets.py --dataset all
```

UCI Hydraulic downloads automatically via `ucimlrepo`. NASA CMAPSS and the
SEU gearbox dataset require a free Kaggle account — the script prints
exact instructions if auto-download fails.

### 3. Generate simulated data

```powershell
python simulator/sensor_generator.py --mode batch --units 200 --output data/simulated/
```

Generates ~650K rows of physics-based run-to-failure data across 200 equipment
units (haul trucks, drill rigs, LHDs) with five failure modes.

### 4. Run notebooks in order

```
notebooks/01_hydraulic_eda.ipynb          → EDA and feature extraction
notebooks/02_hydraulic_classifier.ipynb   → Train and evaluate failure classifiers
notebooks/03_cmapss_rul.ipynb             → Train LSTM RUL predictor
notebooks/04_bearing_fault.ipynb          → Train bearing fault classifier
```

### 5. Run the full stack

```powershell
# API
uvicorn api.main:app --reload --port 8000

# Dashboard (separate terminal)
streamlit run dashboard/app.py --server.port 8501

# MLflow experiment tracking (optional)
mlflow ui --port 5000
```

---

## Models

### Failure Classifier — hydraulic component health
- **Dataset:** UCI Hydraulic System Condition Monitoring (2205 cycles, 17 sensors)
- **Features:** 136 statistical features (8 stats × 17 sensors per cycle)
- **Targets:** Cooler condition (3 classes), accumulator pressure (4 classes), pump leakage (3 classes)
- **Architecture:** Gradient Boosting (cooler, accumulator) + Random Forest with SMOTE (pump)
- **Results:** Cooler F1=0.9935 | Accumulator F1=0.9262 | Pump F1=1.000

### RUL Predictor — remaining useful life
- **Dataset:** NASA CMAPSS FD001 (100 engines, run-to-failure)
- **Input:** Sliding window of 14 sensor readings over last 30 cycles
- **Output:** Estimated cycles until failure (regression), RUL capped at 125
- **Architecture:** 2-layer LSTM (128 hidden units) + MLP head, trained with early stopping
- **Results:** FD001 Test RMSE=14.74 cycles | NASA Score=367.65
- **Note:** FD003 generalization RMSE=49.12 — model is fault-mode specific (documented as future work)

### Fault Classifier — gearbox bearing health
- **Dataset:** SEU gearbox vibration, 88,320 samples × 4 channels per file
- **Input:** 64 features per 512-sample window (9 time-domain + 7 frequency-domain per channel)
- **Output:** Healthy vs ball fault classification
- **Architecture:** Random Forest (300 trees) with StandardScaler pipeline
- **Results:** Test F1=1.000 | Accuracy=1.000 | Perfect generalization across unseen load levels (70%, 80%, 90%)

---

## Key engineering insights

**Hydraulic classifier:** CE (cooling efficiency) sensor features dominate — CE_min, CE_rms,
CE_mean, and CE_max are the top 4 features by significant margin. Cooler degradation manifests
first in cooling efficiency before propagating to downstream pressure (PS5, PS6) and temperature
(TS4) sensors. The stable-flag filter removed 1,449 of 2,205 cycles (66%) — training only on
the 756 stable cycles produced cleaner decision boundaries and higher F1 scores.

**RUL predictor:** Validation RMSE dropped from 83.7 → 12.6 across 60 epochs with a clean
convergence curve. The error distribution is centred near zero with slight overestimation bias
at low RUL values — the more dangerous direction for maintenance scheduling. FD003
generalization failure (RMSE=49.12) is documented honestly: a single-fault-mode model cannot
reliably predict degradation from an unseen second fault mode without retraining.

**Bearing fault classifier:** Kurtosis and crest factor features dominate the importance ranking —
the industry-standard bearing fault indicators. The model achieves perfect F1=1.0 on unseen
load conditions (70–90%), confirming it learned fault physics rather than load-level artifacts.

---

## Tech stack

| Layer | Technology |
|---|---|
| Data | pandas, numpy, scipy, ucimlrepo |
| Models | scikit-learn, imbalanced-learn, PyTorch |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| MLOps | MLflow, Docker, GitHub Actions |
| Testing | pytest, httpx |

---

## Background

Built by **Takudzwa Musimwa** — Mining Engineering BEng (Midlands State University, 2023)
+ Data Science MSc candidate (Pace University, expected 2027).

Field experience: radiation monitoring across 50+ well logging operations, coordination
of multi-department safety teams across three mining sites, and subsurface geological
mapping at Wireline Africa, Mozambique (2023–2025).

---

## License

MIT