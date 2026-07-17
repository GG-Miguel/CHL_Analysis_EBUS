"""
# 08c — PCA Component Interpretation
Visualizes and validates the physical interpretation of the 3 principal
components used in seasonal cycle clustering. Generates loadings plots,
reconstructed cycles, score maps, and extreme examples.
"""

import os
import time
import warnings
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
from sklearn.decomposition import PCA
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from src.utils.config import DATA_DIR, MODISA_FILE
from src.preprocessing.ingestion import load_chlorophyll_dataset
from src.models.clustering import (
    compute_seasonal_cycles,
    _zscore_cycles,
)

FILEPATH = DATA_DIR / MODISA_FILE
ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]
lat = ds.lat.values
lon = ds.lon.values

SUPP_DIR = Path("figures/Supplementary")
SUPP_DIR.mkdir(parents=True, exist_ok=True)

TEMPORAL_RESOLUTION = "8day"
PCT = 85
N_COMPONENTS = 3

print(f"Data shape: {chl.shape}")
print(f"Percentile: P{PCT}")
print(f"PCA components: {N_COMPONENTS}")

print("\n--- Computing seasonal cycle vectors ---")
t0 = time.time()
cycles, lat_sel, lon_sel, periods = compute_seasonal_cycles(
    chl, lat, lon,
    percentile=PCT,
    temporal_resolution=TEMPORAL_RESOLUTION,
)
print(f"Done: {time.time() - t0:.1f}s")
print(f"Valid pixels: {len(lat_sel)}")
print(f"Periods: {len(periods)}")

if len(lat_sel) == 0:
    print("No valid pixels found. Exiting.")
    sys.exit(1)

print("\n--- Z-score normalization ---")
cycles_z = _zscore_cycles(cycles)
print(f"Z-scored cycles shape: {cycles_z.shape}")

print("\n--- Fitting PCA ---")
pca = PCA(n_components=N_COMPONENTS)
scores = pca.fit_transform(cycles_z)
loadings = pca.components_
variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(variance_ratio)

print(f"Variance explained:")
for i in range(N_COMPONENTS):
    print(f"  PC{i+1}: {variance_ratio[i]:.1%} (cumulative: {cumulative_variance[i]:.1%})")

month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
period_to_month = lambda p: month_labels[(p - 1) // 4] if p <= 46 else ""

print("\n--- Figure 1: PCA Loadings ---")
fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.5), sharex=True)
pc_names = ["PC1", "PC2", "PC3"]
pc_colors = ["#d62728", "#1f77b4", "#2ca02c"]

for i, (ax, pc_name, color) in enumerate(zip(axes, pc_names, pc_colors)):
    ax.plot(periods, loadings[i], color=color, linewidth=1.0)
    ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax.fill_between(periods, loadings[i], 0, alpha=0.3, color=color, linewidth=0)
    ax.set_ylabel("Loading", fontsize=6)
    ax.set_title(f"{pc_name} ({variance_ratio[i]:.1%})", fontsize=7, loc="left")
    ax.tick_params(labelsize=5)
    ax.locator_params(axis="y", nbins=3)

axes[-1].set_xlabel("8-day period", fontsize=6)
month_ticks = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
axes[-1].set_xticks(month_ticks)
axes[-1].set_xticklabels(month_labels, fontsize=5)
plt.tight_layout()
fig.savefig(str(SUPP_DIR / "08c_pca_loadings.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {SUPP_DIR}/08c_pca_loadings.png")

print("\n--- Figure 2: Reconstructed Cycles from Individual PCs ---")
fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.5), sharex=True)

for i, (ax, pc_name, color) in enumerate(zip(axes, pc_names, pc_colors)):
    reconstructed = np.outer(scores[:, i], loadings[i])
    mean_reconstructed = np.mean(reconstructed, axis=0)
    std_reconstructed = np.std(reconstructed, axis=0)
    
    ax.plot(periods, mean_reconstructed, color=color, linewidth=1.0,
            label=f"{pc_name} mean")
    ax.fill_between(periods, 
                    mean_reconstructed - std_reconstructed,
                    mean_reconstructed + std_reconstructed,
                    color=color, alpha=0.2, linewidth=0,
                    label="±1 std")
    ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
    ax.set_ylabel("Z-score", fontsize=6)
    ax.set_title(f"Reconstructed from {pc_name} only", fontsize=7, loc="left")
    ax.tick_params(labelsize=5)
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    ax.locator_params(axis="y", nbins=3)

axes[-1].set_xlabel("8-day period", fontsize=6)
axes[-1].set_xticks(month_ticks)
axes[-1].set_xticklabels(month_labels, fontsize=5)
plt.tight_layout()
fig.savefig(str(SUPP_DIR / "08c_pca_reconstructed_cycles.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {SUPP_DIR}/08c_pca_reconstructed_cycles.png")

print("\n--- Figure 3: PCA Score Maps ---")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0),
                         subplot_kw={"projection": ccrs.PlateCarree()})

for i, (ax, pc_name, color) in enumerate(zip(axes, pc_names, pc_colors)):
    ax.set_extent([-30, -7, 10, 45], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, linestyle=":")
    
    sc = ax.scatter(
        lon_sel, lat_sel,
        c=scores[:, i], cmap="RdBu_r", s=0.3, alpha=0.5,
        transform=ccrs.PlateCarree(), rasterized=True,
        vmin=-np.percentile(np.abs(scores[:, i]), 95),
        vmax=np.percentile(np.abs(scores[:, i]), 95),
    )
    
    ax.set_title(f"{pc_name} scores", fontsize=7, pad=4)
    ax.tick_params(labelsize=5)
    
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", 
                        fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("Score", fontsize=5)
    cbar.ax.tick_params(labelsize=4)

plt.tight_layout()
fig.savefig(str(SUPP_DIR / "08c_pca_score_maps.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {SUPP_DIR}/08c_pca_score_maps.png")

print("\n--- Figure 4: Detailed Variance Analysis ---")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.5, 2.0))

all_components = np.arange(1, len(variance_ratio) + 1)
ax1.bar(all_components[:10], variance_ratio[:10], color="#3b6992", alpha=0.7)
ax1.set_xlabel("Principal component", fontsize=6)
ax1.set_ylabel("Variance explained", fontsize=6)
ax1.set_title("Scree plot (first 10 PCs)", fontsize=7)
ax1.tick_params(labelsize=5)
ax1.set_xticks(range(1, 11))

cumvar_full = np.cumsum(variance_ratio)
ax2.plot(all_components, cumvar_full, "o-", color="#3b6992", 
         markersize=2, linewidth=0.8)
ax2.axhline(0.90, color="#d55e00", linestyle="--", linewidth=0.5, alpha=0.7,
            label="90% threshold")
ax2.axhline(0.95, color="#009e73", linestyle="--", linewidth=0.5, alpha=0.7,
            label="95% threshold")
ax2.axvline(N_COMPONENTS, color="#d62728", linestyle=":", linewidth=0.5, alpha=0.7,
            label=f"k={N_COMPONENTS}")
n_90 = int(np.searchsorted(cumvar_full, 0.90) + 1)
n_95 = int(np.searchsorted(cumvar_full, 0.95) + 1)
ax2.axvline(n_90, color="#d55e00", linestyle=":", linewidth=0.5, alpha=0.5)
ax2.axvline(n_95, color="#009e73", linestyle=":", linewidth=0.5, alpha=0.5)
ax2.set_xlabel("Number of components", fontsize=6)
ax2.set_ylabel("Cumulative variance", fontsize=6)
ax2.set_title("Cumulative explained variance", fontsize=7)
ax2.tick_params(labelsize=5)
ax2.legend(frameon=False, fontsize=5, loc="lower right")

plt.tight_layout()
fig.savefig(str(SUPP_DIR / "08c_pca_variance_detailed.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {SUPP_DIR}/08c_pca_variance_detailed.png")

print("\n--- Figure 5: Extreme Examples ---")
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True)

n_extreme = 50

for i, (pc_name, color) in enumerate(zip(pc_names, pc_colors)):
    pc_idx = i
    
    high_idx = np.argsort(scores[:, pc_idx])[-n_extreme:]
    low_idx = np.argsort(scores[:, pc_idx])[:n_extreme]
    
    high_cycles = cycles[high_idx]
    low_cycles = cycles[low_idx]
    
    with np.errstate(all="ignore"):
        high_mean = np.nanmean(high_cycles, axis=0)
        high_std = np.nanstd(high_cycles, axis=0)
        low_mean = np.nanmean(low_cycles, axis=0)
        low_std = np.nanstd(low_cycles, axis=0)
    
    ax_high = axes[0, i]
    ax_high.plot(periods, high_mean, color=color, linewidth=1.0,
                 label=f"High {pc_name}")
    ax_high.fill_between(periods,
                         high_mean - high_std,
                         high_mean + high_std,
                         color=color, alpha=0.2, linewidth=0)
    ax_high.set_ylabel("Chl-a (mg m⁻³)", fontsize=6)
    ax_high.set_title(f"Highest {pc_name} scores", fontsize=7, loc="left")
    ax_high.tick_params(labelsize=5)
    ax_high.legend(frameon=False, fontsize=5)
    
    ax_low = axes[1, i]
    ax_low.plot(periods, low_mean, color=color, linewidth=1.0,
                label=f"Low {pc_name}")
    ax_low.fill_between(periods,
                        low_mean - low_std,
                        low_mean + low_std,
                        color=color, alpha=0.2, linewidth=0)
    ax_low.set_ylabel("Chl-a (mg m⁻³)", fontsize=6)
    ax_low.set_xlabel("8-day period", fontsize=6)
    ax_low.set_title(f"Lowest {pc_name} scores", fontsize=7, loc="left")
    ax_low.tick_params(labelsize=5)
    ax_low.legend(frameon=False, fontsize=5)

for ax in axes[-1]:
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels, fontsize=5)

plt.tight_layout()
fig.savefig(str(SUPP_DIR / "08c_pca_extreme_examples.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {SUPP_DIR}/08c_pca_extreme_examples.png")

print("\n--- Summary Statistics ---")
print(f"\nPCA fit summary:")
print(f"  Total pixels: {len(lat_sel)}")
print(f"  Components: {N_COMPONENTS}")
print(f"  Total variance explained: {cumulative_variance[-1]:.1%}")
print(f"\nScore ranges:")
for i in range(N_COMPONENTS):
    print(f"  PC{i+1}: [{scores[:, i].min():.2f}, {scores[:, i].max():.2f}]")

print(f"\nComponents needed for 90% variance: {n_90}")
print(f"Components needed for 95% variance: {n_95}")

print("\nStep 08c complete.")
