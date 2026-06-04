"""
# 06 — Manuscript Figures (multi-threshold) with Significance
Re-uses the per-year metrics from notebook 03 and the trend-significance
table from notebook 05 to render the final manuscript figures.
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

from src.utils.config import R_EARTH_KM, DATA_DIR, MODISA_FILE, THRESHOLD_PCTS
from src.preprocessing.ingestion import load_chlorophyll_dataset
from src.visualization.maps import plot_mean_chl_map, plot_patch_overlay

FILEPATH = DATA_DIR / MODISA_FILE
ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]
lat = ds.lat.values
lon = ds.lon.values
lat_2d = lat[:, np.newaxis] * np.ones_like(lon)

THRESHOLD_COLORS = {85: "#e8c848", 90: "#c8a830", 95: "#e88a2a", 99: "#d62728"}

# --- Fig 1: Mean map ---
chl_mean = chl.mean(dim="time").values
plot_mean_chl_map(lon, lat_2d, chl_mean, savepath="figures/Fig1_mean_map.png")
print("Saved: figures/Fig1_mean_map.png")

# --- Fig 2: Patch overlays with 4 thresholds (P85=yellow, P90=olive, P95=orange, P99=red) ---
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
    plot_patch_overlay(lon, lat_2d, chl_ann, masks, ey, savepath=f"figures/Fig2_patch_{ey}.png")
    print(f"  Saved: figures/Fig2_patch_{ey}.png")

# --- Fig 3: Multi-threshold metrics time series with CI bands ---
df = pd.read_csv("results/threshold_area_metrics.csv")
fig, axes = plt.subplots(4, 1, figsize=(7.2, 6), sharex=True)
metrics_list = ["area_km2", "mean_chl", "n_patches", "largest_patch"]
labels_map = {
    "area_km2": "Area (km²)",
    "mean_chl": "Mean Chl-a (mg m⁻³)",
    "n_patches": "No. of patches",
    "largest_patch": "Largest patch (km²)",
}
for ax, m in zip(axes, metrics_list):
    for p, c in THRESHOLD_COLORS.items():
        col = f"{m}_p{p}"
        if col in df.columns:
            ax.plot(df["year"], df[col], "o-", linewidth=0.5, color=c,
                    markerfacecolor="white", markeredgewidth=0.3, markeredgecolor=c,
                    markersize=2, label=f"P{p}")
        lo = f"{col}_lo"
        hi = f"{col}_hi"
        if lo in df.columns and hi in df.columns and m in ("area_km2", "mean_chl"):
            ax.fill_between(df["year"], df[lo], df[hi], color=c, alpha=0.12, linewidth=0)
    ax.set_ylabel(labels_map[m], fontsize=6)
    ax.tick_params(labelsize=5)
axes[0].legend(frameon=False, fontsize=5, ncol=3)
axes[-1].set_xlabel("Year", fontsize=6)
fig.suptitle("Multi-Threshold Patch Metrics — Canary EBUS", fontsize=7)
plt.tight_layout()
plt.savefig("figures/Fig3_metrics_timeseries.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: figures/Fig3_metrics_timeseries.png")

# --- Fig 4: Trend bar chart with bootstrap CIs and significance stars ---
df_trends = pd.read_csv("results/trend_significance.csv")
fig, ax = plt.subplots(figsize=(4.8, 2.7))
xpos = np.arange(len(metrics_list))
width = 0.22
for i, p in enumerate(THRESHOLD_PCTS):
    slopes, errs_lo, errs_hi, stars_list = [], [], [], []
    for m in metrics_list:
        col = f"{m}_p{p}"
        r = df_trends[df_trends.metric == col]
        if not r.empty:
            r0 = r.iloc[0]
            slopes.append(r0["slope"])
            errs_lo.append(r0["slope"] - r0["slope_lo"])
            errs_hi.append(r0["slope_hi"] - r0["slope"])
            stars_list.append(r0["stars"])
        else:
            slopes.append(0)
            errs_lo.append(0)
            errs_hi.append(0)
            stars_list.append("n.s.")
    ax.bar(xpos + i * width, slopes, width, color=THRESHOLD_COLORS[p], label=f"P{p}",
           linewidth=0.3, edgecolor="white")
    ax.errorbar(xpos + i * width, slopes,
                yerr=[errs_lo, errs_hi],
                fmt="none", ecolor="#222222", elinewidth=0.4, capsize=1.2, capthick=0.4)
    for j, s in enumerate(slopes):
        if stars_list[j] != "n.s.":
            e_hi = errs_hi[j]
            ax.text(xpos[j] + i * width,
                    s + e_hi + 0.02 * (abs(s) + 1),
                    stars_list[j], ha="center", va="bottom", fontsize=6, color="#222222")
ax.axhline(0, color="#666666", linewidth=0.4)
ax.set_xticks(xpos + width * 1.5)
ax.set_xticklabels([labels_map[m] for m in metrics_list], fontsize=5, rotation=20, ha="right")
ax.set_ylabel("Theil–Sen slope (yr⁻¹)", fontsize=6)
ax.legend(frameon=False, fontsize=5, ncol=4, loc="upper left")
ax.tick_params(labelsize=5)
plt.tight_layout()
fig.savefig("figures/Fig4_trend_bar.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: figures/Fig4_trend_bar.png")

# --- Fig 5: Significance table (reuse from notebook 05) ---
import shutil
shutil.copy("figures/05_significance_table.png", "figures/Fig5_significance_table.png")
print("Saved: figures/Fig5_significance_table.png")

print("\nAll figures saved in figures/")
print("Step 6 complete.")
