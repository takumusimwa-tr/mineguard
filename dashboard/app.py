"""
MineGuard — Streamlit Dashboard
=================================
Digital twin dashboard for mining equipment predictive maintenance.

Tabs:
  1. Fleet Overview   — live health scores for all simulated units
  2. Hydraulic Monitor — real-time cooler/accumulator/pump prediction
  3. RUL Predictor    — engine remaining useful life from CMAPSS data
  4. Bearing Health   — vibration fault classification
  5. Prediction Log   — SQLite history from the API

Run:
    streamlit run dashboard/app.py --server.port 8501

Requires the API to be running:
    uvicorn api.main:app --reload --port 8000
"""

import time
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE  = "http://localhost:8000"
ROOT      = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title  = "MineGuard",
    page_icon   = "⛏",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
C_CRITICAL = "#E24B4A"
C_WARNING  = "#BA7517"
C_NORMAL   = "#1D9E75"
C_BLUE     = "#185FA5"
C_PURPLE   = "#534AB7"
C_BG       = "#0E1117"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1a1d24;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #185FA5;
        margin-bottom: 8px;
    }
    .metric-card.critical { border-left-color: #E24B4A; }
    .metric-card.warning  { border-left-color: #BA7517; }
    .metric-card.normal   { border-left-color: #1D9E75; }
    .metric-label { font-size: 12px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 24px; font-weight: 600; color: #fff; }
    .metric-sub   { font-size: 12px; color: #aaa; margin-top: 2px; }
    .alert-badge  {
        display: inline-block; padding: 3px 10px;
        border-radius: 12px; font-size: 11px; font-weight: 600;
    }
    .badge-critical { background: #3d1515; color: #E24B4A; }
    .badge-warning  { background: #2d2010; color: #BA7517; }
    .badge-normal   { background: #0d2d1f; color: #1D9E75; }
    div[data-testid="stTabs"] button { font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({path}): {e}")
        return None


def api_post(path: str, body: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({path}): {e}")
        return None


def alert_badge(alert: str) -> str:
    cls = {"CRITICAL": "badge-critical", "WARNING": "badge-warning"}.get(
        alert, "badge-normal"
    )
    return f'<span class="alert-badge {cls}">{alert}</span>'


def alert_color(alert: str) -> str:
    return {"CRITICAL": C_CRITICAL, "WARNING": C_WARNING}.get(alert, C_NORMAL)


def health_gauge(value: float, title: str, alert: str = "NORMAL") -> go.Figure:
    color = alert_color(alert)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = value,
        title = {"text": title, "font": {"size": 13}},
        number = {"suffix": "%", "font": {"size": 22}},
        gauge = {
            "axis"      : {"range": [0, 100], "tickfont": {"size": 10}},
            "bar"       : {"color": color, "thickness": 0.25},
            "bgcolor"   : "#1a1d24",
            "bordercolor": "#333",
            "steps"     : [
                {"range": [0,  20], "color": "#2d1515"},
                {"range": [20, 50], "color": "#2d2010"},
                {"range": [50, 100],"color": "#0d2d1f"},
            ],
            "threshold" : {
                "line" : {"color": color, "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        }
    ))
    fig.update_layout(
        height=200, margin=dict(t=40, b=0, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#ccc",
    )
    return fig


def sparkline(values: list, color: str = C_BLUE) -> go.Figure:
    fig = go.Figure(go.Scatter(
        y=values, mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba")
        if color.startswith("rgb") else color + "22",
    ))
    fig.update_layout(
        height=60, margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ── Session state initialisation ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = {
        "health_scores"   : [],
        "rul_values"      : [],
        "alerts"          : [],
        "hydraulic_preds" : [],
        "bearing_preds"   : [],
        "timestamps"      : [],
    }

if "sim_hour" not in st.session_state:
    st.session_state.sim_hour = 2000.0

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛏ MineGuard")
    st.caption("AI Predictive Maintenance")
    st.divider()

    health = api_get("/health")
    if health and health.get("status") == "ok":
        st.success("API connected")
        st.caption(f"Device: {health.get('device','—')}")
    else:
        st.error("API offline — start with:\nuvicorn api.main:app --port 8000")

    st.divider()
    st.markdown("### Simulator controls")

    equip_type = st.selectbox(
        "Equipment type",
        ["HT", "DR", "LHD"],
        format_func=lambda x: {
            "HT": "Haul Truck", "DR": "Drill Rig", "LHD": "LHD Loader"
        }[x],
    )

    failure_mode = st.selectbox(
        "Failure mode",
        ["healthy", "overheating", "bearing_wear",
         "hydraulic_leak", "pump_degradation", "valve_stiction"],
    )

    total_life = st.slider("Total life (hours)", 2000, 6000, 4000, 100)

    st.session_state.sim_hour = st.slider(
        "Operating hour",
        0.0, float(total_life),
        st.session_state.sim_hour,
        step=50.0,
    )

    phase_pct = st.session_state.sim_hour / total_life * 100
    phase_label = (
        "Phase 1 — Stable" if phase_pct < 55 else
        "Phase 2 — Drift"  if phase_pct < 85 else
        "Phase 3 — Critical"
    )
    st.caption(f"Lifecycle: {phase_pct:.0f}% — {phase_label}")

    st.divider()
    auto = st.toggle("Auto-refresh (3s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto

    if st.button("Fetch live reading", use_container_width=True, type="primary"):
        st.session_state.fetch_now = True
    else:
        st.session_state.fetch_now = False


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if st.session_state.auto_refresh:
    time.sleep(3)
    st.rerun()


# ── Fetch current simulator reading ───────────────────────────────────────────
reading = api_get("/stream", params={
    "equipment_type": equip_type,
    "failure_mode"  : failure_mode,
    "operating_hour": st.session_state.sim_hour,
    "total_life"    : total_life,
})

if reading:
    h = st.session_state.history
    h["health_scores"].append(reading["health_score"])
    h["alerts"].append(reading["alert"])
    h["timestamps"].append(datetime.now().strftime("%H:%M:%S"))
    if len(h["health_scores"]) > 60:
        for k in h:
            h[k] = h[k][-60:]


# ── Main header ───────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(f"# ⛏ MineGuard — Equipment Health Dashboard")
    if reading:
        unit_label = {
            "HT": "Haul Truck", "DR": "Drill Rig", "LHD": "LHD Loader"
        }.get(equip_type, equip_type)
        st.caption(
            f"Unit: **{reading['unit_id']}** | "
            f"Type: **{unit_label}** | "
            f"Failure mode: **{failure_mode}** | "
            f"Hour: **{reading['operating_hour']:.0f} / {total_life}**"
        )

with col_status:
    if reading:
        alert = reading["alert"]
        color = alert_color(alert)
        st.markdown(
            f"<div style='text-align:right;padding-top:12px'>"
            f"<span style='font-size:28px;font-weight:700;color:{color}'>"
            f"{alert}</span><br>"
            f"<span style='font-size:12px;color:#888'>System status</span>"
            f"</div>",
            unsafe_allow_html=True
        )


# ── KPI row ───────────────────────────────────────────────────────────────────
if reading:
    k1, k2, k3, k4 = st.columns(4)
    metrics = [
        (k1, "Health Score",    f"{reading['health_score']:.1f}%",
         f"Phase {reading['phase']}",     reading["alert"]),
        (k2, "RUL (hours)",     f"{reading['rul_h']:.0f} h",
         f"of {total_life:.0f} h total",  reading["alert"]),
        (k3, "Operating Hour",  f"{reading['operating_hour']:.0f}",
         f"{reading['operating_hour']/total_life*100:.1f}% of life used", "NORMAL"),
        (k4, "Lifecycle Phase", f"Phase {reading['phase']}",
         {1:"Stable operation",2:"Gradual drift",3:"Accelerated failure"}.get(
             reading["phase"], "—"), reading["alert"]),
    ]
    for col, label, value, sub, alert in metrics:
        with col:
            cls = alert.lower()
            st.markdown(
                f"<div class='metric-card {cls}'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value'>{value}</div>"
                f"<div class='metric-sub'>{sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Fleet Overview",
    "Hydraulic Monitor",
    "RUL Predictor",
    "Bearing Health",
    "Prediction Log",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Fleet Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Live fleet health snapshot")
    st.caption("Simulates 9 equipment units across the fleet at current operating hour.")

    fleet_configs = [
        ("HT-001", "HT",  "overheating",      3800, 4000),
        ("HT-002", "HT",  "bearing_wear",      2100, 4000),
        ("HT-003", "HT",  "healthy",           1500, 4000),
        ("DR-001", "DR",  "pump_degradation",  2800, 3500),
        ("DR-002", "DR",  "valve_stiction",    1200, 3500),
        ("DR-003", "DR",  "healthy",            800, 3500),
        ("LHD-001","LHD", "hydraulic_leak",    3200, 4500),
        ("LHD-002","LHD", "bearing_wear",      1900, 4500),
        ("LHD-003","LHD", "healthy",            600, 4500),
    ]

    cols = st.columns(3)
    for idx, (uid, etype, fmode, op_hr, tot_life) in enumerate(fleet_configs):
        r = api_get("/stream", params={
            "equipment_type": etype,
            "failure_mode"  : fmode,
            "operating_hour": op_hr,
            "total_life"    : tot_life,
        })
        if not r:
            continue

        with cols[idx % 3]:
            health_pct = r["health_score"]
            alert      = r["alert"]
            color      = alert_color(alert)
            type_label = {"HT":"Haul Truck","DR":"Drill Rig","LHD":"LHD Loader"}[etype]

            st.markdown(
                f"<div class='metric-card {alert.lower()}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div>"
                f"<div class='metric-label'>{uid} — {type_label}</div>"
                f"<div class='metric-value' style='color:{color}'>{health_pct:.0f}%</div>"
                f"<div class='metric-sub'>{fmode.replace('_',' ').title()} | "
                f"RUL: {r['rul_h']:.0f}h | Phase {r['phase']}</div>"
                f"</div>"
                f"{alert_badge(alert)}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Health score history chart
    if len(st.session_state.history["health_scores"]) > 2:
        st.markdown("### Selected unit — health score over time")
        hist_df = pd.DataFrame({
            "time"  : st.session_state.history["timestamps"],
            "health": st.session_state.history["health_scores"],
            "alert" : st.session_state.history["alerts"],
        })
        fig = px.line(
            hist_df, x="time", y="health",
            color_discrete_sequence=[C_BLUE],
            labels={"time": "Time", "health": "Health Score (%)"},
        )
        fig.add_hline(y=50, line_dash="dash", line_color=C_WARNING,
                      annotation_text="Warning threshold")
        fig.add_hline(y=20, line_dash="dash", line_color=C_CRITICAL,
                      annotation_text="Critical threshold")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            height=280,
            margin=dict(t=20, b=40, l=40, r=20),
            xaxis=dict(gridcolor="#222"),
            yaxis=dict(gridcolor="#222", range=[0, 105]),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Hydraulic Monitor
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Hydraulic component health classifier")
    st.caption(
        "Predicts cooler, accumulator, and pump condition from 136 statistical sensor features. "
        "Model: Gradient Boosting | Cooler F1=0.994 | Accumulator F1=0.926 | Pump F1=1.000"
    )

    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### Input features")
        st.caption("Adjust key sensor values to simulate different conditions.")

        ps1_mean = st.slider("PS1 mean (bar)", 100.0, 200.0, 152.3, 1.0)
        ce_mean  = st.slider("CE mean (µS/cm)", 10.0, 80.0, 45.2, 0.5)
        ts1_mean = st.slider("TS1 mean (°C)", 30.0, 60.0, 35.7, 0.5)
        fs1_mean = st.slider("FS1 mean (L/min)", 1.5, 3.0, 2.07, 0.01)
        eps1_mean = st.slider("EPS1 mean (W)", 1500.0, 3000.0, 2365.0, 10.0)

        if st.button("Run prediction", type="primary", use_container_width=True):
            st.session_state.run_hydraulic = True

    with col_res:
        if st.session_state.get("run_hydraulic"):
            # Build full 136-feature dict with defaults + user-adjusted values
            feature_defaults = {
                "PS1_mean":152.3,"PS1_std":2.1,"PS1_min":147.0,"PS1_max":157.0,
                "PS1_range":10.0,"PS1_skew":0.1,"PS1_kurt":-0.3,"PS1_rms":152.3,
                "PS2_mean":104.9,"PS2_std":1.8,"PS2_min":101.0,"PS2_max":108.0,
                "PS2_range":7.0,"PS2_skew":0.0,"PS2_kurt":-0.2,"PS2_rms":104.9,
                "PS3_mean":14.8,"PS3_std":0.4,"PS3_min":14.0,"PS3_max":15.5,
                "PS3_range":1.5,"PS3_skew":0.1,"PS3_kurt":-0.1,"PS3_rms":14.8,
                "PS4_mean":7.1,"PS4_std":0.1,"PS4_min":6.9,"PS4_max":7.3,
                "PS4_range":0.4,"PS4_skew":0.0,"PS4_kurt":-0.5,"PS4_rms":7.1,
                "PS5_mean":8.4,"PS5_std":0.2,"PS5_min":8.0,"PS5_max":8.8,
                "PS5_range":0.8,"PS5_skew":0.1,"PS5_kurt":-0.3,"PS5_rms":8.4,
                "PS6_mean":3.8,"PS6_std":0.1,"PS6_min":3.6,"PS6_max":4.0,
                "PS6_range":0.4,"PS6_skew":0.0,"PS6_kurt":-0.4,"PS6_rms":3.8,
                "EPS1_mean":eps1_mean,"EPS1_std":45.0,"EPS1_min":2280.0,"EPS1_max":2450.0,
                "EPS1_range":170.0,"EPS1_skew":0.1,"EPS1_kurt":-0.2,"EPS1_rms":eps1_mean,
                "FS1_mean":fs1_mean,"FS1_std":0.04,"FS1_min":1.99,"FS1_max":2.15,
                "FS1_range":0.16,"FS1_skew":0.0,"FS1_kurt":-0.3,"FS1_rms":fs1_mean,
                "FS2_mean":8.9,"FS2_std":0.1,"FS2_min":8.7,"FS2_max":9.1,
                "FS2_range":0.4,"FS2_skew":0.0,"FS2_kurt":-0.4,"FS2_rms":8.9,
                "TS1_mean":ts1_mean,"TS1_std":0.2,"TS1_min":35.3,"TS1_max":36.1,
                "TS1_range":0.8,"TS1_skew":0.0,"TS1_kurt":-0.5,"TS1_rms":ts1_mean,
                "TS2_mean":40.8,"TS2_std":0.2,"TS2_min":40.4,"TS2_max":41.2,
                "TS2_range":0.8,"TS2_skew":0.0,"TS2_kurt":-0.4,"TS2_rms":40.8,
                "TS3_mean":38.1,"TS3_std":0.2,"TS3_min":37.7,"TS3_max":38.5,
                "TS3_range":0.8,"TS3_skew":0.0,"TS3_kurt":-0.3,"TS3_rms":38.1,
                "TS4_mean":33.4,"TS4_std":0.2,"TS4_min":33.0,"TS4_max":33.8,
                "TS4_range":0.8,"TS4_skew":0.0,"TS4_kurt":-0.4,"TS4_rms":33.4,
                "VS1_mean":0.52,"VS1_std":0.02,"VS1_min":0.48,"VS1_max":0.56,
                "VS1_range":0.08,"VS1_skew":0.1,"VS1_kurt":-0.2,"VS1_rms":0.52,
                "CE_mean":ce_mean,"CE_std":0.3,"CE_min":44.6,"CE_max":45.8,
                "CE_range":1.2,"CE_skew":0.0,"CE_kurt":-0.3,"CE_rms":ce_mean,
                "CP_mean":1.31,"CP_std":0.02,"CP_min":1.27,"CP_max":1.35,
                "CP_range":0.08,"CP_skew":0.0,"CP_kurt":-0.4,"CP_rms":1.31,
                "SE_mean":0.52,"SE_std":0.02,"SE_min":0.48,"SE_max":0.56,
                "SE_range":0.08,"SE_skew":0.1,"SE_kurt":-0.2,"SE_rms":0.52,
            }
            feature_defaults["PS1_mean"] = ps1_mean
            feature_defaults["PS1_rms"]  = ps1_mean

            result = api_post("/predict", {"features": feature_defaults})

            if result:
                st.markdown("#### Prediction results")
                for component in ["cooler", "accumulator", "pump"]:
                    pred = result[component]
                    alert = pred["alert"]
                    color = alert_color(alert)

                    st.markdown(
                        f"<div class='metric-card {alert.lower()}'>"
                        f"<div style='display:flex;justify-content:space-between'>"
                        f"<div>"
                        f"<div class='metric-label'>{component.upper()}</div>"
                        f"<div class='metric-value' style='color:{color}'>"
                        f"{pred['prediction']}</div>"
                        f"</div>"
                        f"{alert_badge(alert)}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # Probability bar chart
                    probs = pred["probabilities"]
                    fig = go.Figure(go.Bar(
                        x=list(probs.values()),
                        y=list(probs.keys()),
                        orientation="h",
                        marker_color=[
                            C_CRITICAL if k == pred["prediction"] else "#333"
                            for k in probs
                        ],
                        text=[f"{v:.1%}" for v in probs.values()],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        height=120, margin=dict(t=0, b=0, l=0, r=60),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#ccc", showlegend=False,
                        xaxis=dict(range=[0, 1.15], visible=False),
                        yaxis=dict(automargin=True),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.caption(f"Latency: {result['latency_ms']:.1f} ms")
        else:
            st.info("Adjust sensor values on the left and click Run prediction.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RUL Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Remaining Useful Life predictor — LSTM")
    st.caption(
        "Predicts cycles until engine failure from last 30 sensor readings. "
        "Model: 2-layer LSTM (128 hidden) | FD001 Test RMSE = 14.74 cycles"
    )

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("#### Simulate engine degradation")
        engine_cycle  = st.slider("Current cycle", 1, 362, 200)
        engine_health = st.slider("Simulated health %", 5, 100, 70)

        st.caption(
            "This generates 30 synthetic CMAPSS-style sensor readings "
            "based on the selected health level and feeds them to the LSTM."
        )

        if st.button("Predict RUL", type="primary", use_container_width=True):
            st.session_state.run_rul = True

    with col_r:
        if st.session_state.get("run_rul"):
            # Generate synthetic CMAPSS window based on health level
            rng    = np.random.default_rng(engine_cycle)
            # Sensor baselines from FD001 training data
            bases  = {
                "s_02":642.0,"s_03":1590.0,"s_04":1408.0,"s_06":21.6,
                "s_07":554.0,"s_08":2388.0,"s_09":9045.0,"s_11":47.5,
                "s_12":522.0,"s_13":2388.0,"s_14":8140.0,"s_15":8.42,
                "s_17":392.0,"s_20":39.0,"s_21":23.4,
            }
            # Degradation: lower health = sensors drift from baseline
            drift = (100 - engine_health) / 100.0
            window = []
            for i in range(30):
                reading = {}
                for k, base in bases.items():
                    noise = rng.normal(0, base * 0.005)
                    deg   = base * drift * 0.08 * (i / 30)
                    reading[k] = round(float(base + deg + noise), 4)
                window.append(reading)

            result = api_post("/rul", {"sensor_window": window})

            if result:
                rul    = result["rul_cycles"]
                health = result["health_pct"]
                alert  = result["alert"]
                color  = alert_color(alert)

                c1, c2, c3 = st.columns(3)
                c1.metric("RUL (cycles)", f"{rul:.0f}")
                c2.metric("Health", f"{health:.1f}%")
                c3.metric("Alert", alert)

                # Gauge
                fig = health_gauge(health, "Engine health", alert)
                st.plotly_chart(fig, use_container_width=True)

                # RUL over simulated lifecycle
                st.markdown("#### Simulated RUL trajectory")
                cycles    = list(range(10, 363, 10))
                rul_curve = [max(0, 125 - max(0, c - 50) * 0.55) for c in cycles]

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=cycles, y=rul_curve,
                    mode="lines", name="Estimated RUL",
                    line=dict(color=C_BLUE, width=2),
                ))
                fig2.add_vline(
                    x=engine_cycle, line_dash="dash",
                    line_color=color, annotation_text="Current cycle",
                )
                fig2.add_hline(y=20, line_dash="dot",
                               line_color=C_CRITICAL,
                               annotation_text="CRITICAL threshold")
                fig2.add_hline(y=50, line_dash="dot",
                               line_color=C_WARNING,
                               annotation_text="WARNING threshold")
                fig2.update_layout(
                    height=280,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc",
                    margin=dict(t=20, b=40, l=40, r=20),
                    xaxis=dict(gridcolor="#222", title="Cycle"),
                    yaxis=dict(gridcolor="#222", title="RUL (cycles)"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption(f"Latency: {result['latency_ms']:.1f} ms")
        else:
            st.info("Set engine parameters and click Predict RUL.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Bearing Health
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Bearing fault classifier")
    st.caption(
        "Classifies bearing health from vibration signals. "
        "Model: Random Forest (300 trees) | Test F1 = 1.000 | "
        "Perfect generalization across unseen load conditions."
    )

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("#### Signal parameters")
        fault_sim  = st.selectbox("Simulate fault type",
                                   ["Healthy", "Ball fault"])
        load_level = st.slider("Load level (%)", 0, 90, 30, 10)
        noise_level = st.slider("Noise level", 0.0, 1.0, 0.1, 0.05)

        if st.button("Classify bearing", type="primary",
                     use_container_width=True):
            st.session_state.run_bearing = True

    with col_r:
        if st.session_state.get("run_bearing"):
            # Generate synthetic vibration window
            rng  = np.random.default_rng(load_level)
            t    = np.linspace(0, 512 / 5120, 512)
            load_factor = 1 + load_level / 100

            channels = []
            for ch in range(4):
                # Base vibration
                sig = (rng.normal(0, 0.3 * load_factor, 512) +
                       0.5 * np.sin(2 * np.pi * 30 * t + ch * 0.5))
                if fault_sim == "Ball fault":
                    # Add ball pass frequency impulses
                    bpf = 97.0
                    for k in [1, 2, 3]:
                        sig += (0.8 / k) * np.sin(2 * np.pi * bpf * k * t)
                    # Add periodic impulses
                    impulse_locs = np.arange(0, 512,
                                             int(5120 / bpf)).astype(int)
                    for loc in impulse_locs:
                        if loc < 512:
                            sig[loc] += rng.normal(3.5, 0.5)
                sig += rng.normal(0, noise_level, 512)
                channels.append(sig)

            window = np.column_stack(channels).tolist()

            result = api_post("/fault", {"vibration_window": window})

            if result:
                fault_class = result["fault_class"]
                confidence  = result["confidence"]
                alert       = result["alert"]
                color       = alert_color(alert)

                c1, c2, c3 = st.columns(3)
                c1.metric("Classification", fault_class)
                c2.metric("Confidence", f"{confidence:.1%}")
                c3.metric("Alert", alert)

                # Probability bars
                probs = result["probabilities"]
                fig = go.Figure(go.Bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    marker_color=[
                        color if k == fault_class else "#333"
                        for k in probs
                    ],
                    text=[f"{v:.1%}" for v in probs.values()],
                    textposition="outside",
                ))
                fig.update_layout(
                    height=200,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc",
                    margin=dict(t=20, b=20, l=20, r=20),
                    yaxis=dict(range=[0, 1.2], gridcolor="#222"),
                    xaxis=dict(gridcolor="#222"),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Raw signal plot
                st.markdown("#### Raw vibration signal — channel a1")
                sig_df = pd.DataFrame({
                    "sample": range(512),
                    "signal": [row[0] for row in window],
                })
                fig2 = px.line(
                    sig_df, x="sample", y="signal",
                    color_discrete_sequence=[color],
                )
                fig2.update_layout(
                    height=200,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc",
                    margin=dict(t=10, b=30, l=40, r=10),
                    xaxis=dict(gridcolor="#222", title="Sample"),
                    yaxis=dict(gridcolor="#222", title="Acceleration (g)"),
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption(f"Latency: {result['latency_ms']:.1f} ms")
        else:
            st.info("Configure signal parameters and click Classify bearing.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Prediction Log
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Prediction history")
    st.caption("Last 50 predictions logged to SQLite by the API.")

    col_refresh, col_limit = st.columns([1, 1])
    with col_limit:
        limit = st.selectbox("Show last", [10, 25, 50, 100], index=1)

    history = api_get("/history", params={"limit": limit})

    if history:
        rows = []
        for entry in history:
            res = entry["result"]
            row = {
                "timestamp"  : entry["timestamp"][:19].replace("T", " "),
                "endpoint"   : entry["endpoint"],
                "latency_ms" : f"{entry['latency_ms']:.1f}",
            }
            if entry["endpoint"] == "/predict":
                row["result"] = (
                    f"Cooler: {res['cooler']['prediction']} | "
                    f"Pump: {res['pump']['prediction']}"
                )
                row["alert"] = (
                    "CRITICAL" if any(
                        v.get("alert") == "CRITICAL"
                        for v in [res["cooler"], res["accumulator"], res["pump"]]
                    ) else "WARNING" if any(
                        v.get("alert") == "WARNING"
                        for v in [res["cooler"], res["accumulator"], res["pump"]]
                    ) else "NORMAL"
                )
            elif entry["endpoint"] == "/rul":
                row["result"] = (
                    f"RUL: {res['rul_cycles']} cycles | "
                    f"Health: {res['health_pct']}%"
                )
                row["alert"] = res.get("alert", "—")
            elif entry["endpoint"] == "/fault":
                row["result"] = (
                    f"{res['fault_class']} "
                    f"({res['confidence']:.0%} confidence)"
                )
                row["alert"] = res.get("alert", "—")
            else:
                row["result"] = "—"
                row["alert"]  = "—"
            rows.append(row)

        df = pd.DataFrame(rows)

        # Colour-code alert column
        def style_alert(val):
            colors = {
                "CRITICAL": "color: #E24B4A; font-weight: 600",
                "WARNING" : "color: #BA7517; font-weight: 600",
                "NORMAL"  : "color: #1D9E75",
            }
            return colors.get(val, "")

        st.dataframe(
            df.style.map(style_alert, subset=["alert"]),
            use_container_width=True,
            hide_index=True,
        )

        # Endpoint breakdown chart
        st.markdown("#### Prediction volume by endpoint")
        ep_counts = df["endpoint"].value_counts().reset_index()
        ep_counts.columns = ["endpoint", "count"]
        fig = px.bar(
            ep_counts, x="endpoint", y="count",
            color_discrete_sequence=[C_BLUE],
            labels={"endpoint": "Endpoint", "count": "Predictions"},
        )
        fig.update_layout(
            height=220,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            margin=dict(t=10, b=30, l=40, r=10),
            xaxis=dict(gridcolor="#222"),
            yaxis=dict(gridcolor="#222"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No predictions logged yet. "
            "Use the Hydraulic Monitor, RUL Predictor, or Bearing Health tabs "
            "to generate predictions."
        )
