"""
# 02 — Annual Peak Metrics
Uses the 99th percentile per timestep to avoid single-pixel outliers.
"""

import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from src.visualization.style import set_nature_style
set_nature_style()
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import DATA_DIR, MODISA_FILE
from src.preprocessing.ingestion import load_chlorophyll_dataset
from src.statistics.peaks import annual_peak_metrics

FILEPATH = DATA_DIR / MODISA_FILE

ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]
years = np.unique(chl.time.dt.year.values)
print(f"Years: {years[0]} – {years[-1]} ({len(years)} years)")

df = annual_peak_metrics(chl)
print(df.head(10))
print(f"\nchl_peak range: {df.chl_peak.min():.4f} – {df.chl_peak.max():.4f} mg/m³")

fig, ax = plt.subplots(figsize=(3.5, 2))
ax.plot(df.year, df.chl_peak, "o-", linewidth=0.5, color="steelblue",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="steelblue", markersize=3)
ax.set_xlabel("Year")
ax.set_ylabel("Annual P99 peak Chl-a (mg m⁻³)")
ax.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig("figures/02_annual_peaks.png", dpi=300, bbox_inches="tight")
plt.close(fig)

df.to_csv("results/annual_peaks.csv", index=False)
print("Saved: results/annual_peaks.csv")
print("Step 2 complete.")
