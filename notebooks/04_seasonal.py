"""
# 04 — Seasonal Cycle Analysis
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
import matplotlib.pyplot as plt

from src.utils.config import DATA_DIR, MODISA_FILE
from src.preprocessing.ingestion import load_chlorophyll_dataset

FILEPATH = DATA_DIR / MODISA_FILE

ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]

print(f"Data shape: {chl.shape}")

# 8-day climatology
chl_clim = chl.groupby("time.dayofyear").mean("time", skipna=True)
doys = chl_clim.dayofyear.values
spatial_mean_clim = chl_clim.mean(dim=["row", "col"], skipna=True)

fig, ax = plt.subplots(figsize=(7.2, 1.8))
ax.plot(doys, spatial_mean_clim.values, linewidth=0.5, color="#3b6992")
ax.set_xlabel("Day of year")
ax.set_ylabel("Chl-a (mg m⁻³)")
ax.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig("figures/04_climatology.png", dpi=300, bbox_inches="tight")
plt.close(fig)

peak_doy = doys[np.nanargmax(spatial_mean_clim.values)]
print(f"Peak chlorophyll day of year: {peak_doy}")

# Monthly climatology
chl_monthly = chl.resample(time="ME").mean(skipna=True)
monthly_clim = chl_monthly.groupby("time.month").mean("time", skipna=True)
spatial_monthly_clim = monthly_clim.mean(dim=["row", "col"], skipna=True)

fig2, ax2 = plt.subplots(figsize=(3.5, 2))
ax2.plot(range(1, 13), spatial_monthly_clim.values, "o-", linewidth=0.5, color="#d55e00",
         markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="#d55e00", markersize=3)
ax2.set_xlabel("Month")
ax2.set_ylabel("Chl-a (mg m⁻³)")
ax2.set_xticks(range(1, 13))
ax2.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig("figures/04_monthly_climatology.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

amplitude = float(spatial_monthly_clim.max().values - spatial_monthly_clim.min().values)
print(f"Seasonal amplitude: {amplitude:.3f} mg/m³")
print("Step 4 complete.")
