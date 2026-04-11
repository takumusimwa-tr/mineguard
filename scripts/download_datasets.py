"""
MineGuard — Dataset Downloader
=======================================
Downloads all three real industrial datasets.

Usage
-----
    python scripts/download_datasets.py              # all
    python scripts/download_datasets.py --dataset hydraulic
    python scripts/download_datasets.py --dataset cmapss
    python scripts/download_datasets.py --dataset bearing
"""

import argparse
import zipfile
import io
from pathlib import Path

import requests
from tqdm import tqdm

ROOT       = Path(__file__).resolve().parents[1]
RAW        = ROOT / "data" / "raw"
HYD_DIR    = RAW / "hydraulic"
CMAPSS_DIR = RAW / "cmapss"
BEAR_DIR   = RAW / "bearing"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download_bytes(url: str, desc: str = "") -> bytes:
    """Download URL → raw bytes with progress bar."""
    print(f"  Downloading {desc} ...")
    resp = requests.get(url, stream=True, timeout=120,
                        headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, leave=False) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            buf.write(chunk)
            bar.update(len(chunk))
    return buf.getvalue()


def _save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"  Saved  {path.relative_to(ROOT)}  ({len(data)/1e6:.1f} MB)")


def _unzip_bytes(data: bytes, out_dir: Path) -> list[str]:
    """Unzip bytes into out_dir, return list of extracted filenames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        for name in names:
            dest = out_dir / Path(name).name   # flatten — no sub-dirs
            if dest.exists():
                continue
            dest.write_bytes(zf.read(name))
        print(f"  Extracted {len(names)} files → {out_dir.relative_to(ROOT)}/")
        return names


# ── Dataset 1: UCI Hydraulic System ──────────────────────────────────────────
#
#  The dataset consists of 17 tab-delimited .txt files (one per sensor) plus
#  a profile.txt with the condition labels.
#  Official page : https://archive.ics.uci.edu/dataset/447
#  The ucimlrepo client is the cleanest way — it uses the UCI REST API.

def download_hydraulic():
    print("\n[1/3] UCI Hydraulic System Condition Monitoring")
    print("     https://archive.ics.uci.edu/dataset/447")

    marker = HYD_DIR / "profile.txt"
    if marker.exists():
        print("  [skip] already downloaded")
        return

    # ── Method A: ucimlrepo Python client (preferred) ──
    try:
        from ucimlrepo import fetch_ucirepo
        import pandas as pd

        print("  Fetching via ucimlrepo client ...")
        ds = fetch_ucirepo(id=447)
        HYD_DIR.mkdir(parents=True, exist_ok=True)

        X = ds.data.features
        y = ds.data.targets
        X.to_csv(HYD_DIR / "sensor_features.csv", index=False)
        y.to_csv(HYD_DIR / "condition_labels.csv", index=False)

        print(f"  sensor_features.csv  ({len(X)} rows × {len(X.columns)} cols)")
        print(f"  condition_labels.csv ({len(y)} rows × {len(y.columns)} cols)")

        # Write a profile.txt marker so skip-check works next time
        (HYD_DIR / "profile.txt").write_text("downloaded via ucimlrepo\n")
        return

    except Exception as e:
        print(f"  ucimlrepo failed ({e}), trying direct HTTP ...")

    # ── Method B: direct UCI zip download ──
    # UCI exposes a zip at this endpoint
    url = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
    try:
        data = _download_bytes(url, "hydraulic zip")
        _unzip_bytes(data, HYD_DIR)
        # The zip contains another nested zip — handle that
        for nested in HYD_DIR.glob("*.zip"):
            inner = _download_bytes.__doc__ and None  # just open it directly
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(HYD_DIR)
            nested.unlink()
        (HYD_DIR / "profile.txt").write_text("downloaded via direct HTTP\n")
        return
    except Exception as e:
        print(f"  Direct HTTP also failed: {e}")

    # ── Fallback: manual instructions ──
    print("""
  MANUAL DOWNLOAD REQUIRED:
  1. Go to https://archive.uci.edu/dataset/447
  2. Click 'Download' → save the zip
  3. Unzip into:  data/raw/hydraulic/
  Expected files: PS1.txt PS2.txt ... TS1.txt ... profile.txt  (17 sensor files + labels)
""")


# ── Dataset 2: NASA CMAPSS ────────────────────────────────────────────────────
#
#  Four sub-datasets (FD001–FD004), each with train/test/RUL txt files.
#  Official: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
#  The NASA open data portal requires no login for this dataset.

CMAPSS_URLS = [
    # Kaggle public mirror (most reliable, no login for this one)
    "https://raw.githubusercontent.com/deep-diver/NASA-Turbofan-Engine-RUL-Prediction/master/dataset/CMAPSSData.zip",
    # Another public mirror
    "https://raw.githubusercontent.com/TobiasRoeding/tuh-de-data/main/nasa/CMAPSSData.zip",
]

def download_cmapss():
    print("\n[2/3] NASA CMAPSS Turbofan Degradation (FD001–FD004)")
    print("     https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data")

    marker = CMAPSS_DIR / "train_FD001.txt"
    if marker.exists():
        print("  [skip] already downloaded")
        return

    CMAPSS_DIR.mkdir(parents=True, exist_ok=True)

    for url in CMAPSS_URLS:
        try:
            data = _download_bytes(url, f"CMAPSS zip ({url.split('/')[2]})")
            extracted = _unzip_bytes(data, CMAPSS_DIR)
            if any("FD001" in n for n in extracted):
                print("  Successfully downloaded CMAPSS dataset.")
                return
        except Exception as e:
            print(f"  Mirror failed: {e}")

    # Fallback
    print("""
  MANUAL DOWNLOAD REQUIRED:
  Option A (easiest — Kaggle, free account needed):
    1. Go to https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
    2. Download the zip
    3. Unzip into:  data/raw/cmapss/
    Expected: train_FD001.txt  test_FD001.txt  RUL_FD001.txt  (× 4 sub-datasets)

  Option B (NASA official):
    1. Go to https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
    2. Download CMAPSSData.zip
    3. Unzip into:  data/raw/cmapss/
""")


# ── Dataset 3: CWRU Bearing Fault ─────────────────────────────────────────────
#
#  .mat files (MATLAB format) — loaded with scipy.io.loadmat().
#  Each file contains drive-end vibration signals at 12 kHz.
#  We grab a representative subset: normal + 3 fault types × 3 loads.

CWRU_FILES = {
    # filename in data/raw/bearing/ : direct download URL
    "normal_0.mat":  "https://engineering.case.edu/sites/default/files/Normal_0.mat",
    "normal_1.mat":  "https://engineering.case.edu/sites/default/files/Normal_1.mat",
    "IR007_0.mat":   "https://engineering.case.edu/sites/default/files/IR007_0.mat",
    "IR007_1.mat":   "https://engineering.case.edu/sites/default/files/IR007_1.mat",
    "IR007_2.mat":   "https://engineering.case.edu/sites/default/files/IR007_2.mat",
    "B007_0.mat":    "https://engineering.case.edu/sites/default/files/B007_0.mat",
    "B007_1.mat":    "https://engineering.case.edu/sites/default/files/B007_1.mat",
    "OR007_0.mat":   "https://engineering.case.edu/sites/default/files/OR007@6_0.mat",
    "OR007_1.mat":   "https://engineering.case.edu/sites/default/files/OR007@6_1.mat",
    "IR014_0.mat":   "https://engineering.case.edu/sites/default/files/IR014_0.mat",
    "IR021_0.mat":   "https://engineering.case.edu/sites/default/files/IR021_0.mat",
    "B014_0.mat":    "https://engineering.case.edu/sites/default/files/B014_0.mat",
    "OR021_0.mat":   "https://engineering.case.edu/sites/default/files/OR021@6_0.mat",
}

# Kaggle mirror for bearing data — much more reliable than CWRU server
CWRU_KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/brjapon/gearbox-fault-diagnosis"


def download_bearing():
    print("\n[3/3] CWRU Bearing Fault Dataset")
    print("     https://engineering.case.edu/bearingdatacenter")

    BEAR_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(BEAR_DIR.glob("*.mat"))
    if len(existing) >= 5:
        print(f"  [skip] {len(existing)} .mat files already present")
        return

    # ── Method A: direct file downloads from CWRU ──
    succeeded = 0
    failed    = []
    for fname, url in CWRU_FILES.items():
        dest = BEAR_DIR / fname
        if dest.exists():
            succeeded += 1
            continue
        try:
            data = _download_bytes(url, fname)
            _save(dest, data)
            succeeded += 1
        except Exception as e:
            failed.append(fname)

    if succeeded >= 5:
        print(f"  Downloaded {succeeded}/{len(CWRU_FILES)} files from CWRU server.")
        if failed:
            print(f"  {len(failed)} files failed (not critical): {failed}")
        return

    # ── Method B: Kaggle CLI ──
    print(f"\n  CWRU server unreliable ({succeeded} files). Trying Kaggle CLI ...")
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "brjapon/gearbox-fault-diagnosis",
             "--path", str(BEAR_DIR), "--unzip"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            mat_files = list(BEAR_DIR.rglob("*.mat"))
            print(f"  Kaggle download succeeded — {len(mat_files)} .mat files")
            return
        else:
            print(f"  Kaggle CLI error: {result.stderr.strip()}")
    except FileNotFoundError:
        print("  Kaggle CLI not installed.")
    except Exception as e:
        print(f"  Kaggle CLI failed: {e}")

    # ── Fallback: manual instructions ──
    print("""
  MANUAL DOWNLOAD (pick one):

  Option A — Kaggle (free account, easiest):
    1. Sign up at https://kaggle.com (free)
    2. Go to https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis
    3. Click Download → unzip into:  data/raw/bearing/
    Expected: lots of .mat files (normal, IR, OR, B fault types)

  Option B — Direct from CWRU:
    1. Go to https://engineering.case.edu/bearingdatacenter/download-data-file
    2. Download files from the "12k Drive End Bearing Fault Data" section
    3. Save .mat files into:  data/raw/bearing/

  Option C — Install Kaggle CLI then re-run:
    pip install kaggle
    # Put your kaggle.json in C:\\Users\\<you>\\.kaggle\\
    python scripts/download_datasets.py --dataset bearing
""")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 58)
    print("Dataset download summary")
    print("=" * 58)

    def _status(path, extensions):
        if not path.exists():
            return "NOT DOWNLOADED"
        files = [f for f in path.rglob("*")
                 if f.is_file() and f.suffix in extensions]
        if not files:
            return "EMPTY (README only)"
        mb = sum(f.stat().st_size for f in files) / 1e6
        return f"{len(files)} files  ({mb:.1f} MB)"

    print(f"  Hydraulic (UCI)   {_status(HYD_DIR,  {'.csv','.txt'})}")
    print(f"  CMAPSS (NASA)     {_status(CMAPSS_DIR,{'.txt'})}")
    print(f"  Bearing (CWRU)    {_status(BEAR_DIR,  {'.mat', '.csv'})}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MineGuard datasets")
    parser.add_argument("--dataset",
                        choices=["hydraulic", "cmapss", "bearing", "all"],
                        default="all")
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
