"""
# 09 — Peak vs Background Separation (Multi-Threshold)
For each threshold (P80, P85, P90, P95, P99), separates pixels into
"background" (below threshold) and "peak" (above threshold) each year,
computes their means, and runs Theil-Sen + Mann-Kendall trend tests.

Also runs quantile regression on the spatial-mean annual time series.
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

from src.utils.config import (
    DATA_DIR, MODISA_FILE,
    THRESHOLD_PCTS, QUANTILE_TAU,
)
from src.preprocessing.ingestion import load_chlorophyll_dataset
from src.statistics.background import quantile_regression_trends
from src.statistics.significance import (
    theil_sen_with_ci,
    mann_kendall,
    p_value_to_stars,
)

FILEPATH = DATA_DIR / MODISA_FILE

SUPP_DIR = Path("figures/Supplementary")
SUPP_DIR.mkdir(parents=True, exist_ok=True)

ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]

print(f"Data shape: {chl.shape}")
print(f"Thresholds: {THRESHOLD_PCTS}")
print(f"Quantile regression taus: {QUANTILE_TAU}")

# ── Compute bg/peak metrics per year per threshold ────────────────────
years = np.unique(chl.time.dt.year.values)
records = []

for y in years:
    yearly = chl.sel(time=str(y))
    vals = yearly.values
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        continue

    row = {"year": int(y), "spatial_mean": float(np.nanmean(valid))}

    for p in THRESHOLD_PCTS:
        t = float(np.percentile(valid, p))
        bg_vals = valid[valid < t]
        peak_vals = valid[valid >= t]

        row[f"bg_mean_p{p}"] = float(np.nanmean(bg_vals)) if len(bg_vals) > 0 else np.nan
        row[f"peak_mean_p{p}"] = float(np.nanmean(peak_vals)) if len(peak_vals) > 0 else np.nan
        row[f"peak_excess_p{p}"] = row[f"peak_mean_p{p}"] - row[f"bg_mean_p{p}"]
        row[f"ratio_p{p}"] = (
            row[f"peak_mean_p{p}"] / row[f"bg_mean_p{p}"]
            if row[f"bg_mean_p{p}"] > 0 else np.nan
        )
        row[f"threshold_p{p}"] = t

    records.append(row)

df = pd.DataFrame(records)
print(f"\nAnnual peak/background metrics ({len(df)} years):")
print(df[["year"] + [f"peak_mean_p{p}" for p in THRESHOLD_PCTS]].head(10))

years_arr = df["year"].astype(float).values

# ── Trend tests ───────────────────────────────────────────────────────
THRESHOLD_COLORS = {80: "#e8c848", 85: "#c8a830", 90: "#e69f00", 95: "#d55e00", 99: "#cc0000"}

all_trend_rows = []
for p in THRESHOLD_PCTS:
    for prefix in ("bg_mean", "peak_mean", "peak_excess", "ratio"):
        col = f"{prefix}_p{p}"
        if col not in df.columns:
            continue
        y = df[col].dropna()
        if len(y) < 3:
            continue
        x = df.loc[y.index, "year"].astype(float).values
        ts = theil_sen_with_ci(y, x=x)
        mk = mann_kendall(y, x=x)
        all_trend_rows.append({
            "metric": col,
            "slope": ts["slope_per_decade"],
            "slope_per_year": ts["slope"],
            "intercept": ts["intercept_at_x0"],
            "slope_lo": ts["slope_lo_per_decade"],
            "slope_hi": ts["slope_hi_per_decade"],
            "slope_pct_per_decade": ts["slope_pct_per_decade"],
            "mk_tau": mk["tau"], "mk_p": mk["p_value"],
            "mk_trend": mk["trend"],
            "stars": p_value_to_stars(mk["p_value"]),
            "n": ts["n"],
        })

df_trends = pd.DataFrame(all_trend_rows)

print("\n=== Peak trends (Theil-Sen per decade) ===")
peak_trend_rows = df_trends[df_trends.metric.str.startswith("peak_mean")]
for _, r in peak_trend_rows.iterrows():
    print(f"  {r['metric']}: slope = {r['slope']:.4f}/decade ({r['stars']})")

print("\n=== Background trends (Theil-Sen per decade) ===")
bg_trend_rows = df_trends[df_trends.metric.str.startswith("bg_mean")]
for _, r in bg_trend_rows.iterrows():
    print(f"  {r['metric']}: slope = {r['slope']:.4f}/decade ({r['stars']})")

# ── Quantile regression ───────────────────────────────────────────────
chl_annual_mean = df["spatial_mean"]
qr_df = quantile_regression_trends(chl_annual_mean, years_arr, QUANTILE_TAU)
print("\n=== Quantile regression (spatial-mean annual Chl-a) ===")
print(qr_df.to_string(index=False))

# ── Figures ───────────────────────────────────────────────────────────
n_pcts = len(THRESHOLD_PCTS)
fig, axes = plt.subplots(3, 1, figsize=(7.2, 5), sharex=True)

# Panel 0: Peak mean chl for each threshold
ax = axes[0]
for p in THRESHOLD_PCTS:
    col = f"peak_mean_p{p}"
    color = THRESHOLD_COLORS.get(p, "#d55e00")
    ax.plot(df["year"], df[col], "o-", linewidth=0.5, color=color,
            markerfacecolor="white", markeredgewidth=0.4, markeredgecolor=color,
            markersize=2, label=f"P{p}")
    row = df_trends[df_trends.metric == col]
    if not row.empty:
        tr = row.iloc[0]
        trend_line = tr["intercept"] + tr["slope_per_year"] * (years_arr - years_arr[0])
        ax.plot(df["year"], trend_line, "--", color=color, linewidth=0.5, alpha=0.7)
ax.set_ylabel("Peak mean Chl-a\n(mg m\u207b\u00b3)", fontsize=6)
ax.legend(frameon=False, fontsize=4, loc="upper right", ncol=3)
ax.tick_params(labelsize=6)

# Panel 1: Background mean chl for each threshold
ax = axes[1]
for p in THRESHOLD_PCTS:
    col = f"bg_mean_p{p}"
    color = THRESHOLD_COLORS.get(p, "#d55e00")
    ax.plot(df["year"], df[col], "o-", linewidth=0.5, color=color,
            markerfacecolor="white", markeredgewidth=0.4, markeredgecolor=color,
            markersize=2, label=f"P{p}")
    row = df_trends[df_trends.metric == col]
    if not row.empty:
        tr = row.iloc[0]
        trend_line = tr["intercept"] + tr["slope_per_year"] * (years_arr - years_arr[0])
        ax.plot(df["year"], trend_line, "--", color=color, linewidth=0.5, alpha=0.7)
ax.set_ylabel("Background mean Chl-a\n(mg m\u207b\u00b3)", fontsize=6)
ax.legend(frameon=False, fontsize=4, loc="upper right", ncol=3)
ax.tick_params(labelsize=6)

# Panel 2: Peak/BG ratio for each threshold
ax = axes[2]
for p in THRESHOLD_PCTS:
    col = f"ratio_p{p}"
    color = THRESHOLD_COLORS.get(p, "#d55e00")
    ax.plot(df["year"], df[col], "o-", linewidth=0.5, color=color,
            markerfacecolor="white", markeredgewidth=0.4, markeredgecolor=color,
            markersize=2, label=f"P{p}")
    row = df_trends[df_trends.metric == col]
    if not row.empty:
        tr = row.iloc[0]
        trend_line = tr["intercept"] + tr["slope_per_year"] * (years_arr - years_arr[0])
        ax.plot(df["year"], trend_line, "--", color=color, linewidth=0.5, alpha=0.7)
ax.set_ylabel("Peak / Background ratio", fontsize=6)
ax.set_xlabel("Year", fontsize=6)
ax.legend(frameon=False, fontsize=4, loc="upper right", ncol=3)
ax.tick_params(labelsize=6)

fig.suptitle("Peak vs Background Chlorophyll — Multi-Threshold", fontsize=7)
plt.tight_layout()
plt.savefig("figures/09_background_vs_peaks.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: figures/09_background_vs_peaks.png")

# Quantile regression figure (supplementary)
fig2, ax2 = plt.subplots(figsize=(3.5, 2.5))
ax2.plot(qr_df["quantile"], qr_df["slope_per_decade"], "o-", color="#3b6992",
         markersize=4, linewidth=0.8)
ax2.axhline(0, color="#666666", linewidth=0.4)
ax2.set_xlabel("Quantile (\u03c4)", fontsize=6)
ax2.set_ylabel("Trend slope (per decade)", fontsize=6)
ax2.set_title("Quantile Regression \u2014 Chl-a Trends", fontsize=7)
ax2.tick_params(labelsize=5)
ax2.set_xticks(QUANTILE_TAU)
plt.tight_layout()
plt.savefig(str(SUPP_DIR / "09_quantile_regression.png"), dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {SUPP_DIR}/09_quantile_regression.png")

# ── Save results ──────────────────────────────────────────────────────
df.to_csv("results/peak_background_metrics.csv", index=False)
df_trends.to_csv("results/peak_background_trends.csv", index=False)
qr_df.to_csv("results/quantile_regression.csv", index=False)
print("Saved: results/peak_background_metrics.csv")
print("Saved: results/peak_background_trends.csv")
print("Saved: results/quantile_regression.csv")

print("Step 9 complete.")
