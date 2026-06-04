"""
# 02 — Annual Peak Metrics with Significance
Uses the 99th percentile per timestep to avoid single-pixel outliers.
Adds Theil–Sen + Mann–Kendall trend tests on both peak intensity
(`chl_peak`) and peak timing (`peak_doy`, circular variable).
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
from src.statistics.significance import (
    mann_kendall,
    theil_sen_with_ci,
    circular_doy_trend,
    p_value_to_stars,
)

FILEPATH = DATA_DIR / MODISA_FILE

ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]
years = np.unique(chl.time.dt.year.values)
print(f"Years: {years[0]} – {years[-1]} ({len(years)} years)")

df = annual_peak_metrics(chl)

# --- Add peak day-of-year (DOY) from time_of_peak ---
df["time_of_peak"] = pd.to_datetime(df["time_of_peak"])
df["peak_doy"] = df["time_of_peak"].dt.dayofyear

print(df.head(10))
print(f"\nchl_peak range: {df.chl_peak.min():.4f} – {df.chl_peak.max():.4f} mg/m³")
print(f"peak_doy range: {df.peak_doy.min()} – {df.peak_doy.max()} (DOY)")

# --- Trend tests ---
ts_peak = theil_sen_with_ci(df["chl_peak"])
mk_peak = mann_kendall(df["chl_peak"])
circ_peak = circular_doy_trend(df["peak_doy"])

print("\n=== chl_peak trend ===")
print(f"  Theil–Sen slope:  {ts_peak['slope']:.4f} mg/m³ per yr  "
      f"[{ts_peak['slope_lo']:.4f}, {ts_peak['slope_hi']:.4f}]")
print(f"  % per decade:     {ts_peak['slope_pct_per_decade']:.2f} %")
print(f"  Mann–Kendall:     τ={mk_peak['tau']:.3f}, p={mk_peak['p_value']:.4g}, "
      f"trend={mk_peak['trend']} ({p_value_to_stars(mk_peak['p_value'])})")

print("\n=== peak_doy trend (circular) ===")
print(f"  Mean direction:   {circ_peak['mean_direction_deg']:.1f}° (i.e. DOY "
      f"{circ_peak['mean_direction_deg']*365.0/360.0:.0f})")
print(f"  Drift:            {circ_peak['drift_deg_per_year']:.2f}° per yr "
      f"(= {circ_peak['drift_deg_per_year']*365.0/360.0:.2f} DOY/yr)")
print(f"  Rayleigh p:       {circ_peak['rayleigh_p']:.4g}")

# --- Two-panel figure ---
fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True)

# Top: peak intensity
ax = axes[0]
ax.plot(df.year, df.chl_peak, "o-", linewidth=0.5, color="steelblue",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="steelblue", markersize=3)
trend_line = ts_peak["intercept"] + ts_peak["slope"] * np.arange(len(df))
ax.plot(df.year, trend_line, "--", color="#d55e00", linewidth=0.5,
        label=f"slope={ts_peak['slope']:.3f}/yr ({p_value_to_stars(mk_peak['p_value'])})")
ax.set_ylabel("Annual P99 peak Chl-a (mg m⁻³)")
ax.legend(frameon=False, fontsize=5, loc="upper left")
ax.tick_params(labelsize=6)

# Bottom: peak day-of-year
ax = axes[1]
ax.plot(df.year, df.peak_doy, "o-", linewidth=0.5, color="#7f4f9a",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="#7f4f9a", markersize=3)
# Plot circular fit: mean direction + drift in DOY
mean_doy = circ_peak["mean_direction_deg"] * 365.0 / 360.0
ax.axhline(mean_doy, color="#d55e00", linewidth=0.5, linestyle="--",
           label=f"mean DOY ≈ {mean_doy:.0f}")
ax.set_ylabel("Peak day-of-year")
ax.set_xlabel("Year")
ax.legend(frameon=False, fontsize=5, loc="upper left")
ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig("figures/02_annual_peaks.png", dpi=300, bbox_inches="tight")
plt.close(fig)

df.to_csv("results/annual_peaks.csv", index=False)
print("Saved: results/annual_peaks.csv")
print("Saved: figures/02_annual_peaks.png")
print("Step 2 complete.")
