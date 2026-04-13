# MineGuard — AI-Powered Predictive Maintenance for Mining Equipment

> An end-to-end industrial ML system that monitors heavy mining equipment in real time,
> predicts failures before they happen, and estimates remaining useful life —
> built on real sensor datasets from hydraulic systems, turbofan engines, and bearings.

**Live demo:** https://huggingface.co/spaces/trent1808/mineguard

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
| RUL v1 (FD001) | NASA CMAPSS | LSTM 2-layer | RMSE = **14.74 cycles** | — |
| RUL v2 (FD003) | NASA CMAPSS | LSTM 3-layer (all FDs) | RMSE = **23.51 cycles** | — |
| Bearing fault | SEU Gearbox | Random Forest | F1 = **1.000** | 2 |

---

## Architecture

```
Data sources
  ├── UCI Hydraulic System (real)        → Failure Classifier  (Gradient Boosting)
  ├── NASA CMAPSS FD001–FD004 (real)     → RUL Predictor       (LSTM v1 + v2)
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
                Docker + GitHub Actions CI/CD
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
│   ├── processed/              ← Feature-engineered DataFrames + 19 figures
│   └── simulated/              ← Output of sensor_generator.py (~650K rows)
│
├── simulator/
│   └── sensor_generator.py     ← Physics-based 3-phase degradation simulator
│
├── models/
│   ├── failure_classifier/     ← cooler/accumulator/pump .pkl + imputer + metadata
│   ├── rul_predictor/          ← lstm_rul.pt + lstm_rul_v2.pt + scalers + metadata
│   └── anomaly_detector/       ← bearing_classifier.pkl + metadata
│
├── api/
│   └── main.py                 ← FastAPI: /predict /rul /fault /stream /health /history
│
├── dashboard/
│   └── app.py                  ← Streamlit digital twin dashboard (5 tabs)
│
├── mlops/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── notebooks/
│   ├── 01_hydraulic_eda.ipynb          ← EDA: 17 sensors, 136 features
│   ├── 02_hydraulic_classifier.ipynb   ← Failure classifier: F1 0.926–1.000
│   ├── 03_cmapss_rul.ipynb             ← LSTM v1: RMSE=14.74 (FD001)
│   ├── 04_bearing_fault.ipynb          ← Fault classifier: F1=1.000
│   └── 05_improved_rul.ipynb           ← LSTM v2: all 4 FDs, FD003 RMSE 49→23
│
├── scripts/
│   ├── download_datasets.py    ← Automated dataset downloader
│   └── log_experiments.py      ← MLflow experiment logger
│
├── tests/
│   └── test_mineguard.py       ← pytest suite (5 tests)
│
├── .github/workflows/ci.yml    ← GitHub Actions CI/CD
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
notebooks/02_hydraulic_classifier.ipynb   → Train failure classifiers
notebooks/03_cmapss_rul.ipynb             → Train LSTM v1 RUL predictor
notebooks/04_bearing_fault.ipynb          → Train bearing fault classifier
notebooks/05_improved_rul.ipynb           → Train LSTM v2 (all 4 FDs)
```

### 5. Run the full stack

```powershell
# API
uvicorn api.main:app --reload --port 8000

# Dashboard (separate terminal)
streamlit run dashboard/app.py --server.port 8501

# MLflow experiment tracking (separate terminal)
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts

# Log all experiments to MLflow
python scripts/log_experiments.py
```

Or spin everything up with Docker:

```powershell
docker-compose up --build
```

---

## Models

### Failure Classifier — hydraulic component health
- **Dataset:** UCI Hydraulic System Condition Monitoring (2205 cycles, 17 sensors)
- **Features:** 136 statistical features (8 stats × 17 sensors per cycle)
- **Targets:** Cooler (3 classes), accumulator pressure (4 classes), pump leakage (3 classes)
- **Architecture:** Gradient Boosting (cooler, accumulator) + Random Forest + SMOTE (pump)
- **Results:** Cooler F1=0.9935 | Accumulator F1=0.9262 | Pump F1=1.000

### RUL Predictor v1 — single fault mode
- **Dataset:** NASA CMAPSS FD001 (100 engines, run-to-failure)
- **Input:** 14 sensors × 30-cycle sliding window
- **Architecture:** 2-layer LSTM (128 hidden) + MLP head
- **Results:** FD001 RMSE=14.74 | NASA Score=367.65

### RUL Predictor v2 — all fault modes (improved)
- **Dataset:** NASA CMAPSS FD001+FD002+FD003+FD004 combined
- **Input:** 15 sensors × 30-cycle window, per-cluster MinMaxScaler (KMeans k=6)
- **Architecture:** 3-layer LSTM (256 hidden) + deeper MLP head
- **Results:** FD001 RMSE=21.87 | FD003 RMSE=23.51 (49% improvement over v1) | FD002=30.76 | FD004=29.97

### Fault Classifier — gearbox bearing health
- **Dataset:** SEU gearbox vibration, 88,320 samples × 4 channels per file
- **Input:** 64 features per 512-sample window (time-domain + FFT, 4 channels)
- **Architecture:** Random Forest (300 trees) with StandardScaler pipeline
- **Results:** F1=1.000 | Accuracy=1.000 | Perfect generalization at 70%, 80%, 90% load

---

## Key engineering insights

**Hydraulic classifier:** CE (cooling efficiency) features dominate — CE_min, CE_rms, CE_mean,
and CE_max are the top 4 features. Cooler degradation manifests first in cooling efficiency
before propagating downstream. The stable-flag filter removed 1,449 of 2,205 cycles (66%) —
training only on 756 stable cycles produced significantly cleaner decision boundaries.

**RUL v1 → v2:** Training on all four CMAPSS sub-datasets fixed the FD003 generalization failure
(RMSE 49.12 → 23.51). The multi-task tradeoff is expected — FD001 RMSE rose from 14.74 to 21.87,
but a model that works across fault modes is more valuable for real mining deployment than one
optimised for a single scenario.

**Bearing fault classifier:** Kurtosis and crest factor features dominate — the industry-standard
bearing fault indicators used in real condition monitoring. F1=1.0 on unseen load conditions
(70–90%) confirms the model learned fault physics, not load-level artifacts.

---

## MLflow experiment tracking

All five model runs are logged with full parameter, metric, and artifact tracking:

```powershell
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts
python scripts/log_experiments.py
```

Navigate to **http://localhost:5000** → Model training to compare runs side by side.

---

## Tech stack

| Layer | Technology |
|---|---|
| Data | pandas, numpy, scipy, ucimlrepo |
| Models | scikit-learn, imbalanced-learn, PyTorch 2.6 (CUDA 12.4) |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| MLOps | MLflow 3.11, Docker, GitHub Actions |
| Testing | pytest |
| Deployment | Hugging Face Spaces |

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