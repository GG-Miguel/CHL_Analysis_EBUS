"""
# 03 — Multi-Threshold Patch Metrics (P85, P95, P99) with Bootstrap CIs
Computes per-year patch metrics (area, mean Chl, # patches, largest
patch) above each of the four thresholds, and adds 95% bootstrap CIs
for the per-year metrics.
"""

import os
import time
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
from skimage import measure

from src.utils.config import R_EARTH_KM, DATA_DIR, MODISA_FILE, THRESHOLD_PCTS
from src.preprocessing.ingestion import load_chlorophyll_dataset
from src.statistics.significance import bootstrap_patch_metrics
from src.visualization.maps import plot_patch_overlay, plot_metrics_timeseries

FILEPATH = DATA_DIR / MODISA_FILE
ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]
lat = ds.lat.values
lon = ds.lon.values
lat_2d = lat[:, np.newaxis] * np.ones_like(lon)

dlat_rad = np.deg2rad(np.abs(lat[1] - lat[0])) if len(lat) > 1 else np.deg2rad(0.04166)
dlon_rad = dlat_rad
lat_rad = np.deg2rad(lat_2d)
pixel_area = R_EARTH_KM**2 * np.cos(lat_rad) * dlat_rad * dlon_rad

# Bootstrap configuration
N_BOOT = int(os.environ.get("N_BOOT", 200))
SEED = 0
print(f"Bootstrap: n_boot={N_BOOT}, seed={SEED}")
print(f"Data shape: {chl.shape}")
print(f"Lat range: {lat.min():.2f} – {lat.max():.2f}")
print(f"Thresholds: {THRESHOLD_PCTS}")

years = np.unique(chl.time.dt.year.values)
records = []

THRESHOLD_COLORS = {85: "#e8c848", 90: "#c8a830", 95: "#e88a2a", 99: "#d62728"}

t_start = time.time()
for y in years:
    yearly = chl.sel(time=str(y))
    vals = yearly.values
    if np.all(np.isnan(vals)):
        continue
    chl_ann = np.nanmean(vals, axis=0)
    ann_valid = chl_ann[~np.isnan(chl_ann)]
    if len(ann_valid) == 0:
        continue

    row = {"year": y}
    for p in THRESHOLD_PCTS:
        t = float(np.percentile(ann_valid, p))
        mask = (chl_ann >= t) & (~np.isnan(chl_ann))
        area = float(np.nansum(pixel_area * mask))

        if area == 0:
            row[f"threshold_p{p}"] = t
            row[f"area_km2_p{p}"] = 0
            row[f"mean_chl_p{p}"] = np.nan
            row[f"n_patches_p{p}"] = 0
            row[f"largest_patch_p{p}"] = 0
            # Bootstrap CIs (n_boot = 0 because mask was empty)
            for k in ("lo", "hi"):
                row[f"area_km2_p{p}_{k}"] = 0
                row[f"mean_chl_p{p}_{k}"] = np.nan
                row[f"threshold_p{p}_{k}"] = t
            continue

        mean_chl = float(np.nanmean(chl_ann[mask]))
        labeled, n_labels = measure.label(mask.astype(int), connectivity=1, return_num=True)
        patch_areas = [float(np.nansum(pixel_area[labeled == i])) for i in range(1, n_labels + 1)]
        largest = max(patch_areas) if patch_areas else 0

        row[f"threshold_p{p}"] = t
        row[f"area_km2_p{p}"] = area
        row[f"mean_chl_p{p}"] = mean_chl
        row[f"n_patches_p{p}"] = n_labels
        row[f"largest_patch_p{p}"] = largest

        # Bootstrap CIs
        boot = bootstrap_patch_metrics(
            chl_ann, pixel_area,
            percentile=p, n_boot=N_BOOT, seed=SEED,
        )
        row[f"threshold_p{p}_lo"] = boot["threshold"]["lo"]
        row[f"threshold_p{p}_hi"] = boot["threshold"]["hi"]
        row[f"area_km2_p{p}_lo"] = boot["area_km2"]["lo"]
        row[f"area_km2_p{p}_hi"] = boot["area_km2"]["hi"]
        row[f"mean_chl_p{p}_lo"] = boot["mean_chl"]["lo"]
        row[f"mean_chl_p{p}_hi"] = boot["mean_chl"]["hi"]

    records.append(row)
    print(f"  year {y} done  ({time.time() - t_start:.1f}s)")

df = pd.DataFrame(records)
print(df[["year"] + [f"threshold_p{p}" for p in THRESHOLD_PCTS]].head(10))

for p in THRESHOLD_PCTS:
    print(f"\n=== P{p} ===")
    print(f"  Mean area: {df[f'area_km2_p{p}'].mean():.0f} km²")
    print(f"  Mean patches: {df[f'n_patches_p{p}'].mean():.1f}")
    print(f"  Mean chl: {df[f'mean_chl_p{p}'].mean():.3f} mg/m³")

# --- Threshold time series with CI bands ---
fig_t, ax_t = plt.subplots(figsize=(3.5, 1.8))
for p, color in THRESHOLD_COLORS.items():
    lo = df.get(f"threshold_p{p}_lo")
    hi = df.get(f"threshold_p{p}_hi")
    if lo is not None and hi is not None:
        ax_t.fill_between(df.year, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax_t.plot(df.year, df[f"threshold_p{p}"], "o-", linewidth=0.5, color=color,
              markerfacecolor="white", markeredgewidth=0.4, markeredgecolor=color, markersize=3,
              label=f"P{p}")
ax_t.set_ylabel("Threshold (mg m⁻³)")
ax_t.set_xlabel("Year")
ax_t.legend(frameon=False, fontsize=5)
ax_t.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig("figures/03_threshold_timeseries.png", dpi=300, bbox_inches="tight")
plt.close(fig_t)

# --- Metrics time series (one panel per metric, lines for each threshold) ---
metrics_to_plot = ["area_km2", "mean_chl", "n_patches", "largest_patch"]
fig, axes = plt.subplots(4, 1, figsize=(7.2, 6), sharex=True)
labels_map = {
    "area_km2": "Area above threshold (km²)",
    "mean_chl": "Mean Chl-a in patch (mg m⁻³)",
    "n_patches": "No. of patches",
    "largest_patch": "Largest patch area (km²)",
}
for ax, m in zip(axes, metrics_to_plot):
    for p, color in THRESHOLD_COLORS.items():
        col = f"{m}_p{p}"
        if col in df.columns:
            ax.plot(df["year"], df[col], "o-", linewidth=0.5, color=color,
                    markerfacecolor="white", markeredgewidth=0.3, markeredgecolor=color,
                    markersize=2, label=f"P{p}")
    ax.set_ylabel(labels_map[m], fontsize=6)
    ax.tick_params(labelsize=5)
    ax.locator_params(axis="y", nbins=4)
axes[0].legend(frameon=False, fontsize=5, ncol=3)
axes[-1].set_xlabel("Year", fontsize=6)
fig.suptitle("Multi-Threshold Patch Metrics — Canary EBUS", fontsize=7)
plt.tight_layout()
plt.savefig("figures/03_threshold_metrics_ts.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: figures/03_threshold_metrics_ts.png")

# --- Patch overlays with 3 thresholds ---
for ey in [2003, 2010, 2020]:
    yearly = chl.sel(time=str(ey))
    vals = yearly.values
    chl_ann = np.nanmean(vals, axis=0)
    ann_valid = chl_ann[~np.isnan(chl_ann)]
    if len(ann_valid) == 0:
        continue
    masks = []
    for p in THRESHOLD_PCTS:
        t = float(np.percentile(ann_valid, p))
        mask = (chl_ann >= t) & (~np.isnan(chl_ann))
        masks.append((mask, THRESHOLD_COLORS[p], f"≥ P{p}"))
    plot_patch_overlay(lon, lat_2d, chl_ann, masks, ey, savepath=f"figures/03_patch_overlay_{ey}.png")
    print(f"  Saved: figures/03_patch_overlay_{ey}.png")

df.to_csv("results/threshold_area_metrics.csv", index=False)
print(f"Saved: results/threshold_area_metrics.csv")
print(f"Bootstrap total time: {time.time() - t_start:.1f}s")
print("Step 3 complete.")
