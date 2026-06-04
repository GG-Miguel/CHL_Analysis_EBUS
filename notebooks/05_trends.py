"""
# 05 — Interannual Variability & Trend Analysis (multi-threshold)
"""

import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from src.visualization.style import set_nature_style
set_nature_style()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.statistics.trends import theil_sen_trend
from src.utils.config import THRESHOLD_PCTS

df = pd.read_csv("results/threshold_area_metrics.csv")
print(f"Loaded {len(df)} years: {df.year.min()} – {df.year.max()}")
print(f"Thresholds: {THRESHOLD_PCTS}")

METRICS_BASE = ["area_km2", "mean_chl", "n_patches", "largest_patch"]
THRESHOLD_COLORS = {85: "#e8c848", 90: "#c8a830", 95: "#e88a2a", 99: "#d62728"}

# --- Trend computation ---
trend_results = []
for m in METRICS_BASE:
    for p in THRESHOLD_PCTS:
        col = f"{m}_p{p}"
        if col not in df.columns:
            continue
        result = theil_sen_trend(df[col])
        result["metric"] = f"{m}_p{p}"
        trend_results.append(result)

df_trends = pd.DataFrame(trend_results)
print("\n=== Theil–Sen Trends ===")
print(df_trends.to_string())

# --- Trend plots per threshold ---
labels_map = {
    "area_km2": "Area (km²)",
    "mean_chl": "Mean Chl-a (mg m⁻³)",
    "n_patches": "No. of patches",
    "largest_patch": "Largest patch (km²)",
}

for p in THRESHOLD_PCTS:
    fig, axes = plt.subplots(len(METRICS_BASE), 1, figsize=(7.2, 4), sharex=True)
    for ax, m in zip(axes, METRICS_BASE):
        col = f"{m}_p{p}"
        if col not in df.columns:
            continue
        ax.plot(df["year"], df[col], "o-", markersize=2.5, linewidth=0.5, color="steelblue",
                markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="steelblue")
        tr = df_trends[df_trends.metric == col].iloc[0]
        trend_line = tr["intercept"] + tr["slope"] * np.arange(len(df))
        ax.plot(df["year"], trend_line, "--", color="#d55e00", linewidth=0.5,
                label=f"slope={tr['slope']:.1f} yr⁻¹")
        ax.set_ylabel(labels_map[m], fontsize=6)
        ax.tick_params(labelsize=5)
        ax.legend(frameon=False, fontsize=5)
    axes[-1].set_xlabel("Year", fontsize=6)
    fig.suptitle(f"P{p} Threshold Metrics — Theil–Sen Trend", fontsize=7)
    plt.tight_layout()
    fig.savefig(f"figures/05_trends_p{p}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: figures/05_trends_p{p}.png")

# --- Trend bar chart (multi-threshold grouped bars) ---
fig, ax = plt.subplots(figsize=(4.5, 2.5))
xpos = np.arange(len(METRICS_BASE))
width = 0.22
for i, p in enumerate(THRESHOLD_PCTS):
    slopes = []
    for m in METRICS_BASE:
        col = f"{m}_p{p}"
        r = df_trends[df_trends.metric == col]
        slopes.append(r.slope.values[0] if len(r) > 0 else 0)
    ax.bar(xpos + i * width, slopes, width, color=THRESHOLD_COLORS[p], label=f"P{p}",
           linewidth=0.3, edgecolor="white")
ax.axhline(0, color="#666666", linewidth=0.4)
ax.set_xticks(xpos + width)
ax.set_xticklabels([labels_map[m] for m in METRICS_BASE], fontsize=5, rotation=20, ha="right")
ax.set_ylabel("Theil–Sen slope (yr⁻¹)", fontsize=6)
ax.legend(frameon=False, fontsize=5)
ax.tick_params(labelsize=5)
plt.tight_layout()
fig.savefig("figures/05_trend_bar.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: figures/05_trend_bar.png")

df_trends.to_csv("results/trends.csv", index=False)
print("Saved: results/trends.csv")
print("Step 5 complete.")
