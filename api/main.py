"""
MineGuard — FastAPI Backend
============================
Serves three trained models via a REST API:

  POST /predict   — hydraulic component health (cooler, accumulator, pump)
  POST /rul       — remaining useful life prediction (LSTM)
  POST /fault     — bearing fault classification
  GET  /health    — API health check + model status
  GET  /models    — model metadata and performance benchmarks
  GET  /stream    — simulate a live sensor reading from MineGuard generator
  GET  /history   — last N predictions from SQLite log

Run:
    uvicorn api.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import json
import time
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
FC_DIR    = MODEL_DIR / "failure_classifier"
RUL_DIR   = MODEL_DIR / "rul_predictor"
AD_DIR    = MODEL_DIR / "anomaly_detector"
DB_PATH   = ROOT / "logs" / "predictions.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── LSTM architecture (must match training exactly) ───────────────────────────
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ── Model registry (loaded once at startup) ───────────────────────────────────
class ModelRegistry:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Failure classifier ─────────────────────────────────────────────
        with open(FC_DIR / "metadata.json") as f:
            self.fc_meta = json.load(f)

        self.fc_feature_cols = self.fc_meta["feature_columns"]
        self.fc_imputer      = joblib.load(FC_DIR / "imputer.pkl")
        self.fc_cooler       = joblib.load(FC_DIR / "cooler_classifier.pkl")
        self.fc_accumulator  = joblib.load(FC_DIR / "accumulator_classifier.pkl")
        self.fc_pump         = joblib.load(FC_DIR / "pump_classifier.pkl")
        self.fc_targets      = self.fc_meta["targets"]

        # ── RUL predictor ──────────────────────────────────────────────────
        with open(RUL_DIR / "metadata.json") as f:
            self.rul_meta = json.load(f)

        self.rul_feature_cols = self.rul_meta["feature_cols"]
        self.rul_window       = self.rul_meta["window_size"]
        self.rul_cap          = self.rul_meta["rul_cap"]
        self.rul_scaler       = joblib.load(RUL_DIR / "scaler.pkl")

        ckpt = torch.load(
            RUL_DIR / "lstm_rul.pt",
            map_location=self.device,
            weights_only=False
        )
        cfg = ckpt["model_config"]
        self.rul_model = LSTMPredictor(**cfg).to(self.device)
        self.rul_model.load_state_dict(ckpt["model_state_dict"])
        self.rul_model.eval()

        # ── Bearing fault classifier ───────────────────────────────────────
        with open(AD_DIR / "metadata.json") as f:
            self.ad_meta = json.load(f)

        self.ad_window      = self.ad_meta["window_size"]
        self.ad_channels    = self.ad_meta["channels"]
        self.ad_class_names = self.ad_meta["class_names"]
        self.ad_classifier  = joblib.load(AD_DIR / "bearing_classifier.pkl")

        print(f"Models loaded — device: {self.device}")
        print(f"  Failure classifier : {len(self.fc_feature_cols)} features")
        print(f"  RUL predictor      : window={self.rul_window}, "
              f"{len(self.rul_feature_cols)} sensors")
        print(f"  Bearing classifier : window={self.ad_window}, "
              f"{len(self.ad_channels)} channels")


registry: ModelRegistry | None = None


# ── SQLite prediction log ──────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            endpoint    TEXT NOT NULL,
            input_hash  TEXT,
            result      TEXT NOT NULL,
            latency_ms  REAL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(endpoint: str, result: dict, latency_ms: float,
                   input_hash: str = ""):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO predictions "
            "(timestamp, endpoint, input_hash, result, latency_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), endpoint,
             input_hash, json.dumps(result), latency_ms)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # never crash the API over a logging failure


# ── Alert logic ───────────────────────────────────────────────────────────────
def hydraulic_alert(pred: int, n_classes: int) -> str:
    """
    Generic alert for hydraulic components.
    Last class = healthy/optimal = NORMAL.
    First class = worst = CRITICAL.
    Middle classes = WARNING.
    """
    if pred == n_classes - 1:
        return "NORMAL"
    if pred == 0:
        return "CRITICAL"
    return "WARNING"


# ── Bearing feature extraction ────────────────────────────────────────────────
def extract_bearing_features(window: np.ndarray) -> np.ndarray:
    """Extract 64 features from a (512, 4) vibration window."""
    from scipy import stats
    from scipy.fft import fft, fftfreq

    SAMPLE_RATE = 5120
    feats = []
    for ch in range(window.shape[1]):
        s    = window[:, ch]
        rms  = np.sqrt(np.mean(s ** 2))
        peak = np.max(np.abs(s))
        feats.extend([
            s.mean(), s.std(), rms, peak,
            peak / (rms + 1e-10),           # crest factor
            stats.kurtosis(s),
            stats.skew(s),
            np.mean(np.abs(s)),             # mean absolute value
            np.mean(np.abs(s)) / (rms + 1e-10),  # shape factor
        ])
        N    = len(s)
        yf   = np.abs(fft(s))[:N // 2]
        xf   = fftfreq(N, 1 / SAMPLE_RATE)[:N // 2]
        tot  = yf.sum() + 1e-10
        cent = np.sum(xf * yf) / tot
        sstd = np.sqrt(np.sum(((xf - cent) ** 2) * yf) / tot)
        top5 = yf[np.argsort(yf)[-5:]]
        feats.extend([cent, sstd] + list(top5))
    return np.array(feats, dtype=np.float32)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry
    init_db()
    registry = ModelRegistry()
    yield
    registry = None


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MineGuard API",
    description=(
        "Predictive maintenance API for heavy mining equipment. "
        "Hydraulic failure classification, RUL prediction, and bearing fault detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class HydraulicInput(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Dict of feature_name → value for all 136 hydraulic features. "
                    "Use GET /stream to generate a valid example.",
    )


class HydraulicPrediction(BaseModel):
    cooler:      dict[str, Any]
    accumulator: dict[str, Any]
    pump:        dict[str, Any]
    latency_ms:  float


class RULInput(BaseModel):
    sensor_window: list[dict[str, float]] = Field(
        ...,
        description=(
            "30 sensor reading dicts ordered oldest → newest. "
            "Required keys: s_02 s_03 s_04 s_06 s_07 s_08 s_09 "
            "s_11 s_12 s_13 s_14 s_15 s_17 s_20 s_21"
        ),
        min_length=30,
        max_length=30,
    )


class RULPrediction(BaseModel):
    rul_cycles:  float
    rul_capped:  bool
    health_pct:  float
    alert:       str
    latency_ms:  float


class BearingInput(BaseModel):
    vibration_window: list[list[float]] = Field(
        ...,
        description="512 × 4 vibration samples [[a1, a2, a3, a4], ...]",
        min_length=512,
        max_length=512,
    )


class BearingPrediction(BaseModel):
    fault_class:   str
    confidence:    float
    probabilities: dict[str, float]
    alert:         str
    latency_ms:    float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """API health check — confirms all three models are loaded."""
    if registry is None:
        raise HTTPException(503, "Models not loaded")
    return {
        "status"    : "ok",
        "timestamp" : datetime.now(timezone.utc).isoformat(),
        "device"    : str(registry.device),
        "models"    : {
            "failure_classifier": "loaded",
            "rul_predictor"     : "loaded",
            "bearing_classifier": "loaded",
        },
    }


@app.get("/models", tags=["System"])
def model_info():
    """Benchmark performance for all three models."""
    if registry is None:
        raise HTTPException(503, "Models not loaded")
    return {
        "failure_classifier": registry.fc_meta["performance"],
        "rul_predictor"     : registry.rul_meta["performance"],
        "bearing_classifier": registry.ad_meta["performance"],
    }


@app.post("/predict", response_model=HydraulicPrediction, tags=["Hydraulic"])
def predict_hydraulic(body: HydraulicInput):
    """
    Predict hydraulic component health from one 60-second cycle.

    Returns health status for cooler, accumulator, and pump
    with class probabilities and CRITICAL / WARNING / NORMAL alert.

    Alert logic:
      - Cooler:      Near failure=CRITICAL, Reduced=WARNING, Full efficiency=NORMAL
      - Accumulator: Near failure=CRITICAL, Severely reduced=WARNING, others=NORMAL
      - Pump:        No leakage=NORMAL, Weak=WARNING, Severe=CRITICAL
    """
    if registry is None:
        raise HTTPException(503, "Models not loaded")

    t0 = time.perf_counter()

    fc      = registry.fc_feature_cols
    missing = [c for c in fc if c not in body.features]
    if missing:
        raise HTTPException(
            422,
            f"Missing {len(missing)} features: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    X = np.array([[body.features[c] for c in fc]], dtype=np.float32)
    X = registry.fc_imputer.transform(X)

    def _predict(model, target_key: str) -> dict:
        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        names = registry.fc_targets[target_key]["class_names"]
        return {
            "prediction"   : names[pred],
            "class_index"  : pred,
            "probabilities": {n: round(float(p), 4)
                              for n, p in zip(names, proba)},
            "alert"        : hydraulic_alert(pred, len(names)),
        }

    result = {
        "cooler"     : _predict(registry.fc_cooler,      "cooler"),
        "accumulator": _predict(registry.fc_accumulator, "accumulator"),
        "pump"       : _predict(registry.fc_pump,        "pump"),
        "latency_ms" : round((time.perf_counter() - t0) * 1000, 2),
    }
    log_prediction("/predict", result, result["latency_ms"])
    return result


@app.post("/rul", response_model=RULPrediction, tags=["RUL"])
def predict_rul(body: RULInput):
    """
    Predict Remaining Useful Life in cycles from the last 30 sensor readings.

    Trained on NASA CMAPSS FD001 (single fault mode).
    Alert thresholds: CRITICAL < 20 cycles, WARNING < 50 cycles.
    """
    if registry is None:
        raise HTTPException(503, "Models not loaded")

    t0 = time.perf_counter()
    fc = registry.rul_feature_cols

    for i, reading in enumerate(body.sensor_window):
        missing = [s for s in fc if s not in reading]
        if missing:
            raise HTTPException(422, f"Cycle {i}: missing sensors {missing}")

    arr    = np.array([[r[s] for s in fc] for r in body.sensor_window],
                      dtype=np.float32)
    arr    = registry.rul_scaler.transform(arr)
    tensor = torch.tensor(arr).unsqueeze(0).to(registry.device)

    with torch.no_grad():
        rul = float(registry.rul_model(tensor).item())

    rul    = max(0.0, rul)
    capped = rul >= registry.rul_cap
    health = round(min(100.0, (rul / registry.rul_cap) * 100), 1)
    alert  = "CRITICAL" if rul < 20 else ("WARNING" if rul < 50 else "NORMAL")

    result = {
        "rul_cycles" : round(rul, 1),
        "rul_capped" : capped,
        "health_pct" : health,
        "alert"      : alert,
        "latency_ms" : round((time.perf_counter() - t0) * 1000, 2),
    }
    log_prediction("/rul", result, result["latency_ms"])
    return result


@app.post("/fault", response_model=BearingPrediction, tags=["Bearing"])
def predict_fault(body: BearingInput):
    """
    Classify bearing health from a 512-sample × 4-channel vibration window.

    Returns fault class (Healthy / Ball fault), confidence score,
    and CRITICAL alert if any fault is detected.
    """
    if registry is None:
        raise HTTPException(503, "Models not loaded")

    t0     = time.perf_counter()
    window = np.array(body.vibration_window, dtype=np.float32)

    if window.shape != (512, 4):
        raise HTTPException(422, f"Expected (512, 4), got {window.shape}")

    feats = extract_bearing_features(window).reshape(1, -1)
    pred  = int(registry.ad_classifier.predict(feats)[0])
    proba = registry.ad_classifier.predict_proba(feats)[0]
    names = registry.ad_class_names

    result = {
        "fault_class"  : names[pred],
        "confidence"   : round(float(proba[pred]), 4),
        "probabilities": {n: round(float(p), 4) for n, p in zip(names, proba)},
        "alert"        : "NORMAL" if pred == 0 else "CRITICAL",
        "latency_ms"   : round((time.perf_counter() - t0) * 1000, 2),
    }
    log_prediction("/fault", result, result["latency_ms"])
    return result


@app.get("/stream", tags=["Simulator"])
def stream_reading(
    equipment_type: str  = "HT",
    failure_mode:   str  = "overheating",
    operating_hour: float = 3200.0,
    total_life:     float = 4000.0,
):
    """
    Generate a simulated sensor reading from the MineGuard physics simulator.

    Returns a live-style sensor record. Useful for dashboard demos
    and integration testing without needing real equipment data.

    equipment_type : HT (haul truck) | DR (drill rig) | LHD (loader)
    failure_mode   : overheating | bearing_wear | hydraulic_leak |
                     pump_degradation | valve_stiction | healthy
    operating_hour : current hour in the equipment lifecycle
    total_life     : total expected life in hours
    """
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from simulator.sensor_generator import (
            EquipmentUnit, EquipmentType, FailureMode,
        )

        unit = EquipmentUnit(
            unit_id      = f"{equipment_type}-API",
            equip_type   = EquipmentType(equipment_type),
            failure_mode = FailureMode(failure_mode),
            total_life_h = total_life,
        )
        unit.current_hour = operating_hour

        sensors = unit.read_sensors()
        alert   = unit.alert_status(sensors)

        return {
            "timestamp"      : datetime.now(timezone.utc).isoformat(),
            "unit_id"        : unit.unit_id,
            "equipment_type" : equipment_type,
            "failure_mode"   : failure_mode,
            "operating_hour" : operating_hour,
            "rul_h"          : round(unit.rul, 1),
            "health_score"   : unit.health_score,
            "phase"          : unit.phase,
            "alert"          : alert,
            "sensors"        : {k: round(v, 3) for k, v in sensors.items()},
        }
    except Exception as e:
        raise HTTPException(500, f"Simulator error: {e}")


@app.get("/history", tags=["System"])
def prediction_history(limit: int = 50):
    """Returns the last N predictions logged to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, endpoint, result, latency_ms "
        "FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [
        {
            "timestamp" : r[0],
            "endpoint"  : r[1],
            "result"    : json.loads(r[2]),
            "latency_ms": r[3],
        }
        for r in rows
    ]