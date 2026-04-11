"""
MineGuard — Industrial Sensor Data Generator
=============================================
Simulates realistic sensor streams for heavy mining equipment
(haul trucks, hydraulic drill rigs, LHDs) with three-phase
degradation physics built in.

Equipment modelled
------------------
  HT  - Haul Truck (CAT 793 style)
  DR  - Hydraulic Drill Rig
  LHD - Load-Haul-Dump loader

Degradation model
-----------------
  Phase 1 (0   – p1_end)  : Stable operation + Gaussian noise
  Phase 2 (p1  – p2_end)  : Slow linear drift toward fault region
  Phase 3 (p2  – EOL)     : Accelerated nonlinear approach to failure

Output
------
  Streaming mode  : yields one dict per tick (for live dashboard)
  Batch mode      : returns a pandas DataFrame (for model training)

CLI usage
---------
  python simulator/sensor_generator.py --mode batch --units 200 --output data/simulated/
  python simulator/sensor_generator.py --mode stream --unit-id HT-001 --type HT
"""

import argparse
import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Equipment & sensor configuration
# ---------------------------------------------------------------------------

class EquipmentType(str, Enum):
    HAUL_TRUCK  = "HT"
    DRILL_RIG   = "DR"
    LHD_LOADER  = "LHD"


class FailureMode(str, Enum):
    HYDRAULIC_LEAK       = "hydraulic_leak"
    BEARING_WEAR         = "bearing_wear"
    OVERHEATING          = "overheating"
    PUMP_DEGRADATION     = "pump_degradation"
    VALVE_STICTION       = "valve_stiction"
    HEALTHY              = "healthy"


@dataclass
class SensorSpec:
    """Normal operating range + failure direction for one sensor."""
    name:        str
    unit:        str
    baseline:    float       # Healthy mean
    noise_std:   float       # Gaussian noise in phase 1
    drift_rate:  float       # Units per 100h in phase 2 (+ = rise, - = fall)
    accel_exp:   float       # Exponent for phase-3 nonlinear blow-up
    warn_thresh: float       # Warning alert trigger
    fail_thresh: float       # Critical alert trigger
    direction:   int = 1     # +1 = rises toward failure, -1 = falls


# Sensor specs per equipment type — tuned to real mining equipment ranges
EQUIPMENT_SENSORS: dict[EquipmentType, list[SensorSpec]] = {

    EquipmentType.HAUL_TRUCK: [
        SensorSpec("engine_temp_c",          "°C",    88,   1.2,  +4.0, 2.1,  102, 110, +1),
        SensorSpec("hydraulic_pressure_bar", "bar",  210,   2.5,  -8.0, 2.3,  175, 155, -1),
        SensorSpec("engine_rpm",             "rpm", 1850,  18.0, -30.0, 1.8, 1700, 1600,-1),
        SensorSpec("oil_temp_c",             "°C",    72,   1.0,  +3.5, 2.0,   90,  98, +1),
        SensorSpec("vibration_rms_g",        "g",    0.8,  0.08,  +0.3, 2.5,  2.0,  3.5,+1),
        SensorSpec("coolant_level_pct",      "%",     95,   0.5,  -2.5, 1.9,   78,  65, -1),
        SensorSpec("fuel_flow_lph",          "L/h",  145,   3.0,  +5.0, 1.7,  170, 185, +1),
        SensorSpec("transmission_temp_c",    "°C",    80,   1.1,  +3.0, 2.2,   98, 108, +1),
    ],

    EquipmentType.DRILL_RIG: [
        SensorSpec("drill_pressure_bar",   "bar",  180,  3.0, -12.0, 2.4,  140, 120, -1),
        SensorSpec("rotation_torque_nm",   "Nm",   420,  8.0,  +20,  2.0,  520, 580, +1),
        SensorSpec("hydraulic_temp_c",     "°C",    65,  1.0,  +5.0, 2.3,   85,  95, +1),
        SensorSpec("vibration_rms_g",      "g",    1.2,  0.12, +0.5, 2.6,  3.0,  5.0,+1),
        SensorSpec("pump_flow_lpm",        "L/min", 95,  2.0,  -4.0, 2.1,   75,  60, -1),
        SensorSpec("bit_wear_index",       "idx",  0.1,  0.005,+0.04,2.8,  0.6,  0.85,+1),
        SensorSpec("air_pressure_bar",     "bar",  6.5,  0.1,  -0.3, 1.9,  5.0,  4.2,-1),
    ],

    EquipmentType.LHD_LOADER: [
        SensorSpec("hydraulic_pressure_bar","bar", 195,  2.0, -10.0, 2.2,  158, 138, -1),
        SensorSpec("engine_temp_c",        "°C",    90,  1.3,  +4.5, 2.0,  105, 113, +1),
        SensorSpec("bucket_force_kn",      "kN",   185,  4.0,  -8.0, 1.8,  155, 135, -1),
        SensorSpec("vibration_rms_g",      "g",    0.9,  0.09, +0.35,2.4,  2.2,  3.8,+1),
        SensorSpec("tyre_temp_c",          "°C",    55,  1.5,  +3.0, 2.1,   72,  82, +1),
        SensorSpec("axle_load_t",          "t",     18,  0.3,  +1.2, 1.9,   22,  25, +1),
        SensorSpec("steering_pressure_bar","bar",  160,  2.5,  -7.0, 2.3,  128, 110, -1),
    ],
}

# Which sensors are primarily affected by each failure mode
FAILURE_SENSOR_MAP: dict[FailureMode, list[str]] = {
    FailureMode.HYDRAULIC_LEAK:   ["hydraulic_pressure_bar","pump_flow_lpm","steering_pressure_bar"],
    FailureMode.BEARING_WEAR:     ["vibration_rms_g","engine_temp_c","transmission_temp_c"],
    FailureMode.OVERHEATING:      ["engine_temp_c","oil_temp_c","coolant_level_pct","hydraulic_temp_c"],
    FailureMode.PUMP_DEGRADATION: ["hydraulic_pressure_bar","pump_flow_lpm","drill_pressure_bar"],
    FailureMode.VALVE_STICTION:   ["hydraulic_pressure_bar","drill_pressure_bar","rotation_torque_nm"],
    FailureMode.HEALTHY:          [],
}


# ---------------------------------------------------------------------------
# Degradation engine
# ---------------------------------------------------------------------------

@dataclass
class EquipmentUnit:
    """A single piece of mine equipment with its own lifecycle state."""
    unit_id:       str
    equip_type:    EquipmentType
    failure_mode:  FailureMode
    total_life_h:  float          # Hours from new to failure
    p1_fraction:   float = 0.55   # Fraction of life in stable phase
    p2_fraction:   float = 0.30   # Fraction of life in drift phase
    current_hour:  float = 0.0
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng()
    )

    @property
    def p1_end(self) -> float:
        return self.total_life_h * self.p1_fraction

    @property
    def p2_end(self) -> float:
        return self.total_life_h * (self.p1_fraction + self.p2_fraction)

    @property
    def rul(self) -> float:
        return max(0.0, self.total_life_h - self.current_hour)

    @property
    def health_score(self) -> float:
        return round(100.0 * (self.rul / self.total_life_h), 2)

    @property
    def phase(self) -> int:
        if self.current_hour < self.p1_end:   return 1
        if self.current_hour < self.p2_end:   return 2
        return 3

    def _degradation_factor(self, sensor: SensorSpec) -> float:
        """
        Signed offset on top of baseline.
        Phase 1: ~0   (pure noise)
        Phase 2: linear ramp
        Phase 3: nonlinear blow-up toward failure threshold
        """
        h = self.current_hour
        affected = (
            self.failure_mode != FailureMode.HEALTHY
            and sensor.name in FAILURE_SENSOR_MAP.get(self.failure_mode, [])
        )
        strength = 1.0 if affected else 0.2

        if self.phase == 1:
            return 0.0

        if self.phase == 2:
            progress = (h - self.p1_end) / (self.p2_end - self.p1_end)
            drift = sensor.drift_rate * progress * (self.p2_end - self.p1_end) / 100.0
            return drift * sensor.direction * strength

        # Phase 3 — nonlinear
        p2_drift  = sensor.drift_rate * (self.p2_end - self.p1_end) / 100.0
        progress  = (h - self.p2_end) / max(1, self.total_life_h - self.p2_end)
        progress  = min(progress, 0.999)
        distance  = abs(sensor.fail_thresh - sensor.baseline)
        p3_contrib = distance * (progress ** sensor.accel_exp)
        return (p2_drift + p3_contrib) * sensor.direction * strength

    def read_sensors(self) -> dict:
        readings = {}
        for spec in EQUIPMENT_SENSORS[self.equip_type]:
            degrad = self._degradation_factor(spec)
            noise  = self.rng.normal(0, spec.noise_std)
            if self.phase == 3:
                noise *= 1.0 + 2.0 * (
                    (self.current_hour - self.p2_end)
                    / max(1, self.total_life_h - self.p2_end)
                )
            readings[spec.name] = round(spec.baseline + degrad + noise, 3)
        return readings

    def alert_status(self, readings: dict) -> str:
        specs = {s.name: s for s in EQUIPMENT_SENSORS[self.equip_type]}
        for name, val in readings.items():
            if name not in specs:
                continue
            s = specs[name]
            if s.direction == 1:
                if val >= s.fail_thresh:  return "CRITICAL"
                if val >= s.warn_thresh:  return "WARNING"
            else:
                if val <= s.fail_thresh:  return "CRITICAL"
                if val <= s.warn_thresh:  return "WARNING"
        return "NORMAL"


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------

def stream_sensor_data(
    unit: EquipmentUnit,
    tick_interval_s: float = 1.0,
    hours_per_tick: float = 1.0,
    max_ticks: Optional[int] = None,
) -> Generator[dict, None, None]:
    """
    Yields one sensor record per tick.

    Parameters
    ----------
    unit            : EquipmentUnit to simulate
    tick_interval_s : real-world seconds between ticks (0 = no sleep)
    hours_per_tick  : simulated hours advanced per tick
    max_ticks       : stop after N ticks (None = run to failure)
    """
    tick = 0
    while True:
        if max_ticks is not None and tick >= max_ticks:
            break
        if unit.current_hour >= unit.total_life_h:
            break

        readings = unit.read_sensors()
        alert    = unit.alert_status(readings)

        yield {
            "timestamp":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "unit_id":        unit.unit_id,
            "equipment_type": unit.equip_type.value,
            "failure_mode":   unit.failure_mode.value,
            "operating_hour": round(unit.current_hour, 1),
            "rul_h":          round(unit.rul, 1),
            "health_score":   unit.health_score,
            "phase":          unit.phase,
            "alert":          alert,
            **readings,
        }

        unit.current_hour += hours_per_tick
        tick += 1
        if tick_interval_s > 0:
            time.sleep(tick_interval_s)


# ---------------------------------------------------------------------------
# Batch generator — full lifecycle DataFrame
# ---------------------------------------------------------------------------

def generate_lifecycle_dataset(
    n_units:        int = 50,
    equip_mix:      Optional[dict] = None,
    seed:           int = 42,
    hours_per_tick: float = 1.0,
) -> pd.DataFrame:
    """
    Generate a full run-to-failure dataset for N equipment units.

    Returns a pandas DataFrame with one row per tick per unit.
    Scale: 50 units ≈ 160K rows; 200 units ≈ 650K rows.
    """
    rng = np.random.default_rng(seed)

    if equip_mix is None:
        equip_mix = {
            EquipmentType.HAUL_TRUCK:  0.40,
            EquipmentType.DRILL_RIG:   0.35,
            EquipmentType.LHD_LOADER:  0.25,
        }

    life_ranges = {
        EquipmentType.HAUL_TRUCK:  (3000, 6000),
        EquipmentType.DRILL_RIG:   (1500, 3500),
        EquipmentType.LHD_LOADER:  (2000, 4500),
    }

    failure_modes_by_type = {
        EquipmentType.HAUL_TRUCK:  [
            FailureMode.OVERHEATING, FailureMode.BEARING_WEAR,
            FailureMode.HYDRAULIC_LEAK, FailureMode.HEALTHY,
        ],
        EquipmentType.DRILL_RIG:   [
            FailureMode.PUMP_DEGRADATION, FailureMode.VALVE_STICTION,
            FailureMode.BEARING_WEAR, FailureMode.HEALTHY,
        ],
        EquipmentType.LHD_LOADER:  [
            FailureMode.HYDRAULIC_LEAK, FailureMode.OVERHEATING,
            FailureMode.BEARING_WEAR, FailureMode.HEALTHY,
        ],
    }

    types     = list(equip_mix.keys())
    fractions = [equip_mix[t] for t in types]
    records   = []

    for i in range(n_units):
        etype_idx = int(rng.choice(len(types), p=fractions))
        etype     = types[etype_idx]
        fmodes    = failure_modes_by_type[etype]
        fmode     = fmodes[int(rng.integers(0, len(fmodes)))]

        lo, hi       = life_ranges[etype]
        total_life   = float(rng.integers(lo, hi))
        p1_frac      = float(rng.uniform(0.45, 0.65))
        p2_frac      = float(rng.uniform(0.20, 0.35))
        base_ts      = datetime(2024, 1, 1) + timedelta(days=int(rng.integers(0, 180)))

        unit = EquipmentUnit(
            unit_id      = f"{etype.value}-{i+1:03d}",
            equip_type   = etype,
            failure_mode = fmode,
            total_life_h = total_life,
            p1_fraction  = p1_frac,
            p2_fraction  = p2_frac,
            rng          = np.random.default_rng(seed + i),
        )

        h = 0.0
        while h < total_life:
            unit.current_hour = h
            readings = unit.read_sensors()
            alert    = unit.alert_status(readings)
            records.append({
                "timestamp":      (base_ts + timedelta(hours=h)).isoformat(timespec="minutes"),
                "unit_id":        unit.unit_id,
                "equipment_type": unit.equip_type.value,
                "failure_mode":   unit.failure_mode.value,
                "operating_hour": round(h, 1),
                "rul_h":          round(max(0, total_life - h), 1),
                "health_score":   round(100.0 * max(0, total_life - h) / total_life, 2),
                "phase":          unit.phase,
                "alert":          alert,
                **readings,
            })
            h += hours_per_tick

    df = pd.DataFrame(records)
    return df.sort_values(["unit_id", "operating_hour"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MineGuard sensor generator")
    parser.add_argument("--mode",   choices=["batch", "stream"], default="batch")
    parser.add_argument("--units",  type=int, default=50,  help="Units for batch mode")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", type=str, default="data/simulated/",
                        help="Output directory for batch CSV")
    # Stream mode args
    parser.add_argument("--unit-id",  default="HT-DEMO")
    parser.add_argument("--type",     default="HT",
                        choices=["HT", "DR", "LHD"])
    parser.add_argument("--failure",  default="overheating",
                        choices=[f.value for f in FailureMode])
    parser.add_argument("--life",     type=float, default=4000.0)
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between ticks in stream mode")
    args = parser.parse_args()

    if args.mode == "batch":
        print(f"Generating batch dataset: {args.units} units …")
        df = generate_lifecycle_dataset(n_units=args.units, seed=args.seed)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"mineguard_{args.units}units_seed{args.seed}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df):,} rows → {out_path}")
        print(f"Columns: {list(df.columns)}")

    else:  # stream
        import json
        unit = EquipmentUnit(
            unit_id      = args.unit_id,
            equip_type   = EquipmentType(args.type),
            failure_mode = FailureMode(args.failure),
            total_life_h = args.life,
        )
        print(f"Streaming {unit.unit_id} ({unit.equip_type.value}) — "
              f"failure mode: {unit.failure_mode.value}")
        print("Ctrl+C to stop\n")
        try:
            for record in stream_sensor_data(unit, tick_interval_s=args.interval):
                print(json.dumps(record))
        except KeyboardInterrupt:
            print("\nStream stopped.")
