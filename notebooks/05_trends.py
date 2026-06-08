"""
# 05 — Interannual Variability & Trend Analysis with Significance
Loads the per-year metrics from notebook 03 and runs:
  * Theil–Sen slope with 95% confidence interval (per decade)
  * Mann–Kendall trend test (p-value + τ)
  * Significance stars (*** / ** / * / n.s.)

Produces:
  * results/trend_significance.csv  — wide significance table
  * figures/05_significance_table.png — rendered table
  * figures/05_trends_p<XX>.png — per-threshold time series with CI bands
  * figures/05_trend_bar.png — bar chart with CI error bars + stars
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

from src.statistics.trends import trend_summary_table
from src.utils.config import THRESHOLD_PCTS

df = pd.read_csv("results/threshold_area_metrics.csv")
print(f"Loaded {len(df)} years: {df.year.min()} – {df.year.max()}")
print(f"Thresholds: {THRESHOLD_PCTS}")

METRICS_BASE = ["area_km2", "mean_chl", "n_patches", "largest_patch"]
THRESHOLD_COLORS = {85: "#e8c848", 90: "#c8a830", 95: "#e88a2a", 99: "#d62728"}

# --- Compute the significance table (per-decade headline) ---
value_cols = [f"{m}_p{p}" for m in METRICS_BASE for p in THRESHOLD_PCTS]
value_cols = [c for c in value_cols if c in df.columns]

df_trends = trend_summary_table(df, "year", value_cols)
print("\n=== Theil–Sen (per decade) + Mann–Kendall ===")
print(df_trends.to_string(index=False))

df_trends.to_csv("results/trends.csv", index=False)
df_trends.to_csv("results/trend_significance.csv", index=False)
print("Saved: results/trend_significance.csv")

labels_map = {
    "area_km2": "Area (km²)",
    "mean_chl": "Mean Chl-a (mg m⁻³)",
    "n_patches": "No. of patches",
    "largest_patch": "Largest patch (km²)",
}

years_arr = df["year"].astype(float).values

# --- Per-threshold time series with CI bands ---
for p in THRESHOLD_PCTS:
    fig, axes = plt.subplots(len(METRICS_BASE), 1, figsize=(7.2, 4), sharex=True)
    for ax, m in zip(axes, METRICS_BASE):
        col = f"{m}_p{p}"
        lo_col = f"{col}_lo"
        hi_col = f"{col}_hi"
        if col not in df.columns:
            continue
        ax.plot(df["year"], df[col], "o-", markersize=2.5, linewidth=0.5, color="steelblue",
                markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="steelblue",
                label="data")

        # CI band (only available for area_km2, mean_chl, threshold)
        if lo_col in df.columns and hi_col in df.columns:
            ax.fill_between(df["year"], df[lo_col], df[hi_col],
                            color="steelblue", alpha=0.15, linewidth=0,
                            label="95% bootstrap CI")

# Theil–Sen trend line (raw intercept + per-year slope × year)
        row = df_trends[df_trends.metric == col]
        if not row.empty:
            tr = row.iloc[0]
            trend_line = tr["intercept"] + tr["slope_per_year"] * years_arr
            ax.plot(df["year"], trend_line, "--", color="#d55e00", linewidth=0.6,
                    label=f"Theil–Sen {tr['slope']:.2g}/decade ({tr['stars']})")
        ax.set_ylabel(labels_map[m], fontsize=6)
        ax.tick_params(labelsize=5)
        ax.legend(frameon=False, fontsize=4, loc="best")
    axes[-1].set_xlabel("Year", fontsize=6)
    fig.suptitle(f"P{p} Threshold Metrics — Theil–Sen Trend (per decade)", fontsize=7)
    plt.tight_layout()
    fig.savefig(f"figures/05_trends_p{p}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: figures/05_trends_p{p}.png")

# --- Bar chart of trends with CIs and significance stars ---
fig, ax = plt.subplots(figsize=(4.8, 2.7))
xpos = np.arange(len(METRICS_BASE))
width = 0.22
for i, p in enumerate(THRESHOLD_PCTS):
    slopes = []
    errs_lo = []
    errs_hi = []
    stars_list = []
    for m in METRICS_BASE:
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
ax.set_xticklabels([labels_map[m] for m in METRICS_BASE], fontsize=5, rotation=20, ha="right")
ax.set_ylabel("Theil–Sen slope (per decade)", fontsize=6)
ax.legend(frameon=False, fontsize=5, ncol=4, loc="upper left")
ax.tick_params(labelsize=5)
plt.tight_layout()
fig.savefig("figures/05_trend_bar.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: figures/05_trend_bar.png")

# --- Render the significance table as a PNG ---
fig_t, ax_t = plt.subplots(figsize=(7.2, 0.32 * len(df_trends) + 0.6))
ax_t.axis("off")

disp = df_trends.copy()
disp["slope"] = disp["slope"].map(lambda x: f"{x:.3g}")
disp["slope_per_year"] = disp["slope_per_year"].map(lambda x: f"{x:.3g}")
disp["slope_lo"] = disp["slope_lo"].map(lambda x: f"{x:.3g}")
disp["slope_hi"] = disp["slope_hi"].map(lambda x: f"{x:.3g}")
disp["slope_pct_per_decade"] = disp["slope_pct_per_decade"].map(lambda x: f"{x:.2f}%")
disp["mk_tau"] = disp["mk_tau"].map(lambda x: f"{x:.3f}")
disp["mk_p"] = disp["mk_p"].map(lambda x: f"{x:.3g}")
disp = disp[["metric", "slope", "slope_per_year", "slope_lo", "slope_hi",
             "slope_pct_per_decade", "mk_tau", "mk_p", "mk_trend", "stars", "n"]]
disp.columns = ["metric", "slope/dec", "slope/yr", "slope_lo/dec", "slope_hi/dec",
                "%/decade", "MK τ", "MK p", "MK trend", "sig.", "n"]

tbl = ax_t.table(cellText=disp.values, colLabels=disp.columns, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(6)
tbl.scale(1, 1.2)
for i, stars in enumerate(df_trends["stars"]):
    color = {"***": "#fff0f0", "**": "#fff5e8", "*": "#fffce8"}.get(stars, "white")
    for j in range(len(disp.columns)):
        tbl[(i + 1, j)].set_facecolor(color)
ax_t.set_title("Multi-threshold trend significance (Theil–Sen + Mann–Kendall, per decade)",
               fontsize=7, pad=8)
plt.tight_layout()
plt.savefig("figures/05_significance_table.png", dpi=300, bbox_inches="tight")
plt.close(fig_t)
print("Saved: figures/05_significance_table.png")

print("Step 5 complete.")
