#!/usr/bin/env python3
"""Run full analysis pipeline end-to-end."""

import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
NOTEBOOKS = [
    "01_data_ingestion.py",
    "02_annual_peaks.py",
    "03_threshold_area.py",
    "04_seasonal.py",
    "05_trends.py",
    "07_quantile_trends.py",
    "09_peak_background.py",
    "08_peak_background.py",
    "08_seasonal_clustering.py",
    "06_figures.py",
]

start = time.time()
for nb in NOTEBOOKS:
    t0 = time.time()
    path = ROOT / "notebooks" / nb
    print(f"\n{'='*60}")
    print(f"Running {nb}...")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, cwd=ROOT)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR in {nb}:\n{result.stderr}")
        sys.exit(1)
    print(f"Finished in {time.time()-t0:.1f}s")

print(f"\n{'='*60}")
print(f"Pipeline complete in {time.time()-start:.1f}s")
print(f"Results: {ROOT/'results'}")
print(f"Figures: {ROOT/'figures'}")
