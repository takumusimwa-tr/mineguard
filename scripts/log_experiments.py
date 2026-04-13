"""
MineGuard — MLflow Experiment Logger
======================================
Logs all three trained models to MLflow with full experiment tracking:
  - Parameters (model config, training settings)
  - Metrics (F1, RMSE, accuracy, NASA score)
  - Artifacts (confusion matrix figures, feature importance plots)
  - Model registry (registers each model with version tags)

Run:
    python scripts/log_experiments.py

Requires MLflow server running:
    mlflow server --host 0.0.0.0 --port 5000
    --backend-store-uri sqlite:///mlflow/mlflow.db
    --default-artifact-root ./mlflow/artifacts
"""

import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend

from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

ROOT      = Path(__file__).resolve().parents[1]
PROC_DIR  = ROOT / "data" / "processed"
FC_DIR    = ROOT / "models" / "failure_classifier"
RUL_DIR   = ROOT / "models" / "rul_predictor"
AD_DIR    = ROOT / "models" / "anomaly_detector"

MLFLOW_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_URI)

print(f"MLflow tracking URI: {MLFLOW_URI}")
print(f"MLflow version     : {mlflow.__version__}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, class_names, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_feature_importance(model, feature_names, title, path, top_n=20):
    importances = model.feature_importances_
    imp = pd.Series(importances, index=feature_names).nlargest(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.sort_values().plot.barh(ax=ax, color="#185FA5", alpha=0.85)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Hydraulic Failure Classifier
# ══════════════════════════════════════════════════════════════════════════════

def log_hydraulic_classifier():
    print("\n[1/3] Logging hydraulic failure classifier...")

    mlflow.set_experiment("hydraulic-failure-classifier")

    with open(FC_DIR / "metadata.json") as f:
        meta = json.load(f)

    # Load processed data
    df = pd.read_csv(PROC_DIR / "hydraulic_features.csv")
    with open(FC_DIR / "feature_columns.json") as f:
        feat_cols = json.load(f)

    imputer = joblib.load(FC_DIR / "imputer.pkl")
    X = imputer.transform(df[feat_cols].values)

    targets = {
        "cooler"     : {"model_file": "cooler_classifier.pkl",
                        "label_map" : {3:0, 20:1, 100:2},
                        "names"     : ["Near failure","Reduced efficiency","Full efficiency"]},
        "accumulator": {"model_file": "accumulator_classifier.pkl",
                        "label_map" : {90:0, 100:1, 115:2, 130:3},
                        "names"     : ["Near failure","Severely reduced",
                                       "Slightly reduced","Optimal"]},
        "pump"       : {"model_file": "pump_classifier.pkl",
                        "label_map" : {0:0, 1:1, 2:2},
                        "names"     : ["No leakage","Weak leakage","Severe leakage"]},
    }

    for target, cfg in targets.items():
        model  = joblib.load(FC_DIR / cfg["model_file"])
        y_true = df[target].map(cfg["label_map"]).values
        y_pred = model.predict(X)
        f1     = f1_score(y_true, y_pred, average="macro")
        acc    = accuracy_score(y_true, y_pred)

        model_type = type(model).__name__

        with mlflow.start_run(run_name=f"hydraulic-{target}"):

            # Parameters
            mlflow.log_params({
                "target"          : target,
                "model_type"      : model_type,
                "n_features"      : len(feat_cols),
                "n_classes"       : len(cfg["names"]),
                "training_samples": len(X),
                "dataset"         : "UCI Hydraulic System (ID 447)",
                "stable_only"     : True,
            })

            # Add model-specific params
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, "max_depth"):
                mlflow.log_param("max_depth", model.max_depth)
            if hasattr(model, "learning_rate"):
                mlflow.log_param("learning_rate", model.learning_rate)

            # Metrics
            mlflow.log_metrics({
                "f1_macro"  : round(float(f1), 4),
                "accuracy"  : round(float(acc), 4),
            })

            # Per-class F1
            per_class = f1_score(y_true, y_pred, average=None)
            for name, score in zip(cfg["names"], per_class):
                safe_name = name.lower().replace(" ", "_")
                mlflow.log_metric(f"f1_{safe_name}", round(float(score), 4))

            # Confusion matrix artifact
            cm_path = PROC_DIR / f"mlflow_cm_{target}.png"
            save_confusion_matrix(
                y_true, y_pred, cfg["names"],
                f"{target.capitalize()} — confusion matrix",
                cm_path
            )
            mlflow.log_artifact(str(cm_path), "confusion_matrices")

            # Feature importance (if RF)
            if hasattr(model, "feature_importances_"):
                fi_path = PROC_DIR / f"mlflow_fi_{target}.png"
                save_feature_importance(
                    model, feat_cols,
                    f"{target.capitalize()} — feature importance",
                    fi_path
                )
                mlflow.log_artifact(str(fi_path), "feature_importance")

            # Tags
            mlflow.set_tags({
                "project"    : "MineGuard",
                "component"  : target,
                "dataset"    : "UCI-Hydraulic",
                "engineer"   : "Takudzwa Musimwa",
            })

            print(f"  {target:15s}  F1={f1:.4f}  Acc={acc:.4f}  [{model_type}]")

    print("  Hydraulic classifier logged.")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — RUL LSTM Predictor
# ══════════════════════════════════════════════════════════════════════════════

def log_rul_predictor():
    print("\n[2/3] Logging RUL LSTM predictor...")

    mlflow.set_experiment("rul-lstm-predictor")

    with open(RUL_DIR / "metadata.json") as f:
        meta = json.load(f)

    perf = meta["performance"]

    with mlflow.start_run(run_name="lstm-rul-fd001"):

        # Parameters
        mlflow.log_params({
            "model_type"      : "LSTM",
            "architecture"    : "2-layer LSTM + MLP head",
            "hidden_size"     : 128,
            "num_layers"      : 2,
            "dropout"         : 0.2,
            "window_size"     : meta["window_size"],
            "rul_cap"         : meta["rul_cap"],
            "n_features"      : len(meta["feature_cols"]),
            "batch_size"      : 256,
            "max_epochs"      : 60,
            "learning_rate"   : 1e-3,
            "early_stop_patience": 10,
            "optimizer"       : "Adam",
            "loss"            : "MSE",
            "trained_on"      : meta["trained_on"],
            "dataset"         : "NASA CMAPSS FD001",
        })

        # Metrics — FD001
        mlflow.log_metrics({
            "fd001_test_rmse"   : perf["FD001"]["rmse"],
            "fd001_nasa_score"  : perf["FD001"]["nasa_score"],
            "fd003_test_rmse"   : perf["FD003"]["rmse"],
            "fd003_nasa_score"  : perf["FD003"]["nasa_score"],
        })

        # Log training curve figure as artifact
        training_fig = PROC_DIR / "fig11_training_curves.png"
        if training_fig.exists():
            mlflow.log_artifact(str(training_fig), "training_plots")

        rul_eval_fig = PROC_DIR / "fig12_rul_evaluation.png"
        if rul_eval_fig.exists():
            mlflow.log_artifact(str(rul_eval_fig), "evaluation_plots")

        engine_fig = PROC_DIR / "fig14_engine_rul_timeline.png"
        if engine_fig.exists():
            mlflow.log_artifact(str(engine_fig), "evaluation_plots")

        fd3_fig = PROC_DIR / "fig13_fd003_generalization.png"
        if fd3_fig.exists():
            mlflow.log_artifact(str(fd3_fig), "generalization_plots")

        # Tags
        mlflow.set_tags({
            "project"         : "MineGuard",
            "component"       : "RUL predictor",
            "dataset"         : "NASA-CMAPSS",
            "framework"       : "PyTorch",
            "generalization"  : "FD001-only — see FD003 note",
            "engineer"        : "Takudzwa Musimwa",
        })

        print(f"  FD001 RMSE={perf['FD001']['rmse']}  "
              f"NASA Score={perf['FD001']['nasa_score']}")
        print(f"  FD003 RMSE={perf['FD003']['rmse']}  "
              f"(generalization — single fault mode limitation)")

    print("  RUL predictor logged.")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Bearing Fault Classifier
# ══════════════════════════════════════════════════════════════════════════════

def log_bearing_classifier():
    print("\n[3/3] Logging bearing fault classifier...")

    mlflow.set_experiment("bearing-fault-classifier")

    with open(AD_DIR / "metadata.json") as f:
        meta = json.load(f)

    perf = meta["performance"]

    with mlflow.start_run(run_name="rf-bearing-fault"):

        # Parameters
        mlflow.log_params({
            "model_type"      : "RandomForest",
            "n_estimators"    : 300,
            "window_size"     : meta["window_size"],
            "sample_rate_hz"  : meta["sample_rate"],
            "n_channels"      : len(meta["channels"]),
            "n_features"      : meta["n_features"],
            "n_classes"       : len(meta["class_names"]),
            "train_loads"     : str(meta["train_loads"]),
            "test_loads"      : str(meta["test_loads"]),
            "feature_types"   : "time-domain + FFT",
            "scaler"          : "StandardScaler",
            "dataset"         : "SEU Gearbox Vibration",
        })

        # Metrics
        mlflow.log_metrics({
            "test_f1_macro" : perf["test_f1_macro"],
            "test_accuracy" : perf["test_accuracy"],
            "cv_f1_mean"    : perf["cv_f1_mean"],
        })

        # Per-load generalization metrics (from notebook figures)
        for load, f1 in [(70, 1.0), (80, 1.0), (90, 1.0)]:
            mlflow.log_metric(f"f1_load_{load}pct", f1)

        # Artifacts
        for fig_name in ["fig17_bearing_confusion.png",
                         "fig18_load_generalization.png",
                         "fig19_bearing_importance.png"]:
            fig_path = PROC_DIR / fig_name
            if fig_path.exists():
                mlflow.log_artifact(str(fig_path), "evaluation_plots")

        # Tags
        mlflow.set_tags({
            "project"         : "MineGuard",
            "component"       : "bearing fault classifier",
            "dataset"         : "SEU-Gearbox",
            "generalization"  : "perfect across unseen loads 70-90%",
            "key_features"    : "kurtosis + crest factor dominate",
            "engineer"        : "Takudzwa Musimwa",
        })

        print(f"  F1={perf['test_f1_macro']}  "
              f"Acc={perf['test_accuracy']}  "
              f"CV={perf['cv_f1_mean']}")

    print("  Bearing classifier logged.")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "=" * 60)
    print("MLflow experiments logged")
    print("=" * 60)
    print(f"  UI: {MLFLOW_URI}")
    print()
    print("  Experiments:")
    print("    hydraulic-failure-classifier  (3 runs: cooler, acc, pump)")
    print("    rul-lstm-predictor            (1 run: FD001)")
    print("    bearing-fault-classifier      (1 run)")
    print()
    print("  View at: http://localhost:5000/#/experiments")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        log_hydraulic_classifier()
        log_rul_predictor()
        log_bearing_classifier()
        print_summary()
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure MLflow server is running:")
        print("  mlflow server --host 0.0.0.0 --port 5000 "
              "--backend-store-uri sqlite:///mlflow/mlflow.db "
              "--default-artifact-root ./mlflow/artifacts")
        sys.exit(1)
