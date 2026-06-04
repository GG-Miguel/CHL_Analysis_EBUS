"""
# 01 — Data Ingestion & Exploration
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
from src.visualization.maps import plot_mean_chl_map

FILEPATH = DATA_DIR / MODISA_FILE

ds = load_chlorophyll_dataset(FILEPATH)

print(ds)
print(f"\nTime range: {ds.time.values[0]} to {ds.time.values[-1]}")
print(f"Spatial dims: {dict(ds.sizes)}")

chl = ds["chl"]
lat = ds.lat.values
lon = ds.lon.values
lat_2d = lat[:, np.newaxis] * np.ones_like(lon)

print(f"\nchl shape: {chl.shape}")
print(f"chl range: {float(chl.min().values):.4f} – {float(chl.max().values):.4f} mg/m³")
print(f"NaN fraction: {float(np.isnan(chl.values).sum()) / chl.values.size:.4f}")

# Mean map
chl_mean = chl.mean(dim="time").values
fig, ax = plot_mean_chl_map(lon, lat_2d, chl_mean, savepath="figures/01_mean_map.png")
print("Saved: figures/01_mean_map.png")

# Time series of spatial mean
ts_spatial_mean = chl.mean(dim=["row", "col"], skipna=True)
fig2, ax2 = plt.subplots(figsize=(7.2, 1.8))
ax2.plot(ts_spatial_mean.time.values, ts_spatial_mean.values, linewidth=0.4, color="#3b6992")
ax2.set_ylabel("Chl-a (mg m⁻³)")
ax2.set_xlabel("Time")
ax2.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig("figures/01_spatial_mean_ts.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
print("Saved: figures/01_spatial_mean_ts.png")

print("Step 1 complete.")
