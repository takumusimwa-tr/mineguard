"""
MineGuard — API tests
Run: pytest tests/ -v
"""
import pytest
import numpy as np


def test_imports():
    """Core dependencies import cleanly."""
    import numpy, pandas, sklearn, joblib, fastapi, pydantic
    assert True


def test_sensor_generator():
    """Simulator generates valid records."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))

    from simulator.sensor_generator import (
        EquipmentUnit, EquipmentType, FailureMode
    )

    unit = EquipmentUnit(
        unit_id      = "TEST-001",
        equip_type   = EquipmentType.HAUL_TRUCK,
        failure_mode = FailureMode.OVERHEATING,
        total_life_h = 1000.0,
    )

    # Phase 1
    unit.current_hour = 100.0
    r = unit.read_sensors()
    assert "engine_temp_c" in r
    assert "hydraulic_pressure_bar" in r
    assert unit.phase == 1
    assert unit.health_score == pytest.approx(90.0, abs=1.0)

    # Phase 3
    unit.current_hour = 950.0
    assert unit.phase == 3
    assert unit.rul == pytest.approx(50.0, abs=1.0)


def test_generator_batch():
    """Batch dataset generates expected shape."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))

    from simulator.sensor_generator import generate_lifecycle_dataset

    df = generate_lifecycle_dataset(n_units=3, seed=42)
    assert len(df) > 0
    assert "unit_id" in df.columns
    assert "rul_h" in df.columns
    assert "health_score" in df.columns
    assert "phase" in df.columns
    assert df["phase"].between(1, 3).all()
    assert df["health_score"].between(0, 100).all()


def test_feature_extraction():
    """Bearing feature extractor returns correct shape."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))

    from api.main import extract_bearing_features

    window = np.random.randn(512, 4).astype(np.float32)
    feats  = extract_bearing_features(window)
    assert feats.shape == (64,), f"Expected (64,), got {feats.shape}"
    assert not np.isnan(feats).any(), "Features contain NaN"


def test_model_files_exist():
    """All required model files are present."""
    from pathlib import Path

    root = Path(__file__).parents[1]
    required = [
        root / "models" / "failure_classifier" / "cooler_classifier.pkl",
        root / "models" / "failure_classifier" / "accumulator_classifier.pkl",
        root / "models" / "failure_classifier" / "pump_classifier.pkl",
        root / "models" / "failure_classifier" / "imputer.pkl",
        root / "models" / "failure_classifier" / "feature_columns.json",
        root / "models" / "rul_predictor"      / "lstm_rul.pt",
        root / "models" / "rul_predictor"      / "scaler.pkl",
        root / "models" / "anomaly_detector"   / "bearing_classifier.pkl",
    ]
    for path in required:
        assert path.exists(), f"Missing: {path.relative_to(root)}"
