"""
MineGuard — Dataset Downloader
================================
Downloads the three real industrial datasets used in this project.

Usage
-----
    python scripts/download_datasets.py              # download all
    python scripts/download_datasets.py --dataset hydraulic
    python scripts/download_datasets.py --dataset cmapss
    python scripts/download_datasets.py --dataset bearing

Datasets
--------
1. UCI Hydraulic System Condition Monitoring
   Source : https://archive.ics.uci.edu/dataset/447
   Size   : ~7 MB (2205 cycles × 17 sensors at up to 100 Hz)
   Target : cooler, valve, pump, accumulator health (4 targets)
   Why    : hydraulic systems are the heart of every mine machine

2. NASA CMAPSS Turbofan Degradation (FD001–FD004)
   Source : https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
   Mirror : https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
   Size   : ~4 MB (4 sub-datasets, 26 columns, run-to-failure)
   Target : Remaining Useful Life (RUL) in operating cycles
   Why    : industry-standard RUL benchmark; maps to haul-truck engine life

3. Case Western Reserve University Bearing Fault Dataset
   Source : https://engineering.case.edu/bearingdatacenter/download-data-file
   Mirror : Kaggle — https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis
   Size   : ~70 MB (vibration signals at 12 kHz and 48 kHz)
   Target : fault type (normal, inner race, outer race, ball) + severity
   Why    : bearing failures are the #1 cause of unplanned downtime in mining
"""

import argparse
import hashlib
import os
import zipfile
import io
from pathlib import Path

import requests
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
RAW        = ROOT / "data" / "raw"
HYD_DIR    = RAW / "hydraulic"
CMAPSS_DIR = RAW / "cmapss"
BEAR_DIR   = RAW / "bearing"


# ── Download helpers ──────────────────────────────────────────────────────────

def _download(url: str, dest: Path, desc: str = "") -> Path:
    """Stream-download url → dest with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return dest

    print(f"  Downloading {desc or dest.name} …")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"  Saved → {dest.relative_to(ROOT)}")
    return dest


def _unzip(zip_path: Path, out_dir: Path) -> None:
    """Unzip archive into out_dir, skipping if already extracted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        already = all((out_dir / n).exists() for n in names if not n.endswith("/"))
        if already:
            print(f"  [skip] {zip_path.name} already extracted")
            return
        print(f"  Extracting {zip_path.name} …")
        zf.extractall(out_dir)
        print(f"  Extracted {len(names)} files → {out_dir.relative_to(ROOT)}")


# ── Dataset 1: UCI Hydraulic System ──────────────────────────────────────────

def download_hydraulic():
    """
    UCI Hydraulic System Condition Monitoring dataset.

    The ucimlrepo package is the cleanest way to fetch this —
    it handles auth-free download and returns ready-to-use DataFrames.
    Falls back to direct HTTP if the package isn't installed.
    """
    print("\n[1/3] UCI Hydraulic System Condition Monitoring")
    print("     Source: https://archive.uci.edu/dataset/447")

    try:
        from ucimlrepo import fetch_ucirepo
        import pandas as pd

        marker = HYD_DIR / "profile.csv"
        if marker.exists():
            print("  [skip] dataset already downloaded")
            return

        print("  Fetching via ucimlrepo …")
        ds = fetch_ucirepo(id=447)

        # The dataset is structured as matrices per sensor
        # X = sensor time-series features, y = condition labels
        X = ds.data.features
        y = ds.data.targets

        HYD_DIR.mkdir(parents=True, exist_ok=True)
        X.to_csv(HYD_DIR / "sensor_features.csv", index=False)
        y.to_csv(HYD_DIR / "condition_labels.csv", index=False)

        # Also save metadata
        meta_lines = [
            "# UCI Hydraulic System Condition Monitoring",
            "# Dataset ID: 447",
            "# URL: https://archive.uci.edu/dataset/447",
            "#",
            "# Targets (condition labels):",
            "#   cooler_condition    : 3=near failure, 20=reduced, 100=full efficiency",
            "#   valve_condition     : 100=optimal, 90=small lag, 80=severe, 73=failure",
            "#   pump_leakage        : 0=none, 1=weak, 2=severe",
            "#   accumulator_pressure: 130=optimal, 115=slight, 100=severe, 90=failure",
            "#   stable_flag         : 1=stable conditions reached",
            "#",
            "# Sensor types (17 total):",
            "#   PS1-PS6  : pressure sensors (bar), 100 Hz",
            "#   EPS1     : motor power (W), 100 Hz",
            "#   FS1-FS2  : volume flow (L/min), 10 Hz",
            "#   TS1-TS4  : temperature (°C), 1 Hz",
            "#   VS1      : vibration (mm/s), 1 Hz",
            "#   CE, CP, SE: efficiency, cooling, efficiency, 1 Hz",
        ]
        (HYD_DIR / "README.md").write_text("\n".join(meta_lines))
        print(f"  Saved sensor_features.csv ({len(X)} rows × {len(X.columns)} cols)")
        print(f"  Saved condition_labels.csv ({len(y)} rows × {len(y.columns)} cols)")

    except ImportError:
        print("  ucimlrepo not installed — install with: pip install ucimlrepo")
        print("  Or download manually from: https://archive.uci.edu/dataset/447")
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Manual download: https://archive.uci.edu/dataset/447")


# ── Dataset 2: NASA CMAPSS ────────────────────────────────────────────────────

# Kaggle mirror — stable, no auth required for this specific dataset
CMAPSS_KAGGLE_URL = (
    "https://raw.githubusercontent.com/Samimust/predictive-maintenance/"
    "master/CMAPSSData.zip"
)

# Column names per the CMAPSS paper
CMAPSS_COLS = (
    ["unit_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i:02d}" for i in range(1, 22)]
)


def download_cmapss():
    """
    NASA CMAPSS Turbofan Degradation dataset (FD001–FD004).

    Tries a public GitHub mirror first. If that fails, prints
    instructions for the official NASA download.
    """
    print("\n[2/3] NASA CMAPSS Turbofan Degradation")
    print("     Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data")

    marker = CMAPSS_DIR / "train_FD001.txt"
    if marker.exists():
        print("  [skip] dataset already downloaded")
        return

    CMAPSS_DIR.mkdir(parents=True, exist_ok=True)

    # Try GitHub mirror
    mirrors = [
        "https://raw.githubusercontent.com/Samimust/predictive-maintenance/master/CMAPSSData.zip",
        "https://github.com/schwxd/LSTM-Keras-CMAPSS/raw/master/C-MAPSS-Data.zip",
    ]

    zip_path = CMAPSS_DIR / "CMAPSSData.zip"
    downloaded = False

    for url in mirrors:
        try:
            _download(url, zip_path, "CMAPSS zip")
            downloaded = True
            break
        except Exception as e:
            print(f"  Mirror {url[:60]}… failed: {e}")

    if downloaded and zip_path.exists():
        _unzip(zip_path, CMAPSS_DIR)
        zip_path.unlink()  # remove zip to save space
    else:
        print("\n  Auto-download failed. Manual steps:")
        print("  1. Go to https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data")
        print("  2. Download CMAPSSData.zip")
        print(f"  3. Unzip into: {CMAPSS_DIR.relative_to(ROOT)}/")
        print("     You should have: train_FD001.txt, test_FD001.txt, RUL_FD001.txt (×4)")

    # Write README regardless
    readme = [
        "# NASA CMAPSS Turbofan Degradation Dataset",
        "# Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data",
        "#",
        "# Files: train_FD00{1-4}.txt, test_FD00{1-4}.txt, RUL_FD00{1-4}.txt",
        "#",
        "# Column layout (26 columns, space-separated, no header):",
        "#   1  unit_id",
        "#   2  cycle (operating cycle number)",
        "#   3-5  op_setting_1, op_setting_2, op_setting_3",
        "#   6-26 sensor_01 … sensor_21",
        "#",
        "# Sub-datasets:",
        "#   FD001: 1 fault mode (HPC degradation), 1 operating condition",
        "#   FD002: 1 fault mode, 6 operating conditions",
        "#   FD003: 2 fault modes, 1 operating condition",
        "#   FD004: 2 fault modes, 6 operating conditions",
        "#",
        "# Task: predict Remaining Useful Life (RUL) for each engine in test set",
        "# True RUL values provided in RUL_FD00{1-4}.txt",
    ]
    (CMAPSS_DIR / "README.md").write_text("\n".join(readme))


# ── Dataset 3: CWRU Bearing ────────────────────────────────────────────────────

CWRU_FILES = {
    # Normal baseline (1797 RPM, drive end)
    "normal_0hp.mat": "https://engineering.case.edu/sites/default/files/Normal_0.mat",
    # Inner race faults (drive end, 0.007 inch fault)
    "ir007_0hp.mat":  "https://engineering.case.edu/sites/default/files/IR007_0.mat",
    "ir007_1hp.mat":  "https://engineering.case.edu/sites/default/files/IR007_1.mat",
    "ir007_2hp.mat":  "https://engineering.case.edu/sites/default/files/IR007_2.mat",
    # Ball faults (drive end, 0.007 inch)
    "b007_0hp.mat":   "https://engineering.case.edu/sites/default/files/B007_0.mat",
    "b007_1hp.mat":   "https://engineering.case.edu/sites/default/files/B007_1.mat",
    # Outer race faults (drive end, 0.007 inch)
    "or007_0hp.mat":  "https://engineering.case.edu/sites/default/files/OR007@6_0.mat",
    "or007_1hp.mat":  "https://engineering.case.edu/sites/default/files/OR007@6_1.mat",
}

CWRU_KAGGLE_NOTE = """
  Note: CWRU direct server is sometimes slow or rate-limited.
  Reliable alternative — Kaggle dataset (requires free Kaggle account):
    https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis

  To download via Kaggle CLI:
    pip install kaggle
    kaggle datasets download brjapon/gearbox-fault-diagnosis
    unzip gearbox-fault-diagnosis.zip -d data/raw/bearing/
"""


def download_bearing():
    """
    Case Western Reserve University Bearing Fault dataset.

    Tries direct download from CWRU server.
    Falls back to instructions for Kaggle mirror.
    """
    print("\n[3/3] CWRU Bearing Fault Dataset")
    print("     Source: https://engineering.case.edu/bearingdatacenter")

    BEAR_DIR.mkdir(parents=True, exist_ok=True)

    any_success = False
    for fname, url in CWRU_FILES.items():
        dest = BEAR_DIR / fname
        if dest.exists():
            print(f"  [skip] {fname}")
            any_success = True
            continue
        try:
            _download(url, dest, fname)
            any_success = True
        except Exception as e:
            print(f"  Failed {fname}: {e}")

    if not any_success:
        print(CWRU_KAGGLE_NOTE)

    # README
    readme = [
        "# CWRU Bearing Fault Dataset",
        "# Source: https://engineering.case.edu/bearingdatacenter",
        "#",
        "# .mat files — load with scipy.io.loadmat()",
        "# Key arrays inside each file:",
        "#   DE_time : drive end accelerometer (12 kHz or 48 kHz)",
        "#   FE_time : fan end accelerometer",
        "#   BA_time : base accelerometer (some files)",
        "#   RPM     : shaft speed",
        "#",
        "# Fault types:",
        "#   normal  : healthy baseline",
        "#   IR      : inner race fault",
        "#   B       : ball fault",
        "#   OR      : outer race fault",
        "#",
        "# Fault sizes: 0.007, 0.014, 0.021, 0.028 inch diameter",
        "# Loads: 0, 1, 2, 3 HP (motor load)",
        "#",
        "# Kaggle mirror (easier download):",
        "#   https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis",
    ]
    (BEAR_DIR / "README.md").write_text("\n".join(readme))


# ── Dataset summary printout ───────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 58)
    print("Dataset download summary")
    print("=" * 58)
    datasets = {
        "Hydraulic (UCI)": HYD_DIR,
        "CMAPSS (NASA)":   CMAPSS_DIR,
        "Bearing (CWRU)":  BEAR_DIR,
    }
    for name, path in datasets.items():
        if not path.exists():
            status = "NOT DOWNLOADED"
        else:
            files = list(path.rglob("*"))
            data_files = [f for f in files if f.is_file() and f.suffix != ".md"]
            if not data_files:
                status = "EMPTY"
            else:
                total_mb = sum(f.stat().st_size for f in data_files) / 1e6
                status = f"{len(data_files)} files  ({total_mb:.1f} MB)"
        print(f"  {name:<22} {status}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MineGuard datasets")
    parser.add_argument(
        "--dataset",
        choices=["hydraulic", "cmapss", "bearing", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    print("\nMineGuard Dataset Downloader")
    print("=" * 58)

    if args.dataset in ("hydraulic", "all"):
        download_hydraulic()
    if args.dataset in ("cmapss", "all"):
        download_cmapss()
    if args.dataset in ("bearing", "all"):
        download_bearing()

    print_summary()
