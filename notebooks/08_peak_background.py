"""
# 08 — Peak vs Background Decomposition

Two methods for separating peaks from background:

A) Climatological P80 threshold — months above P80 of monthly climatology
   are "peak months"; the rest are background.

B) Running-mean background — a 24-month centred rolling average extracts
   the slowly-varying background; positive anomalies are peaks.

For both, Theil-Sen + Mann-Kendall trends are computed. STL decomposition
extracts the long-term trend component as an additional reference.
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
from src.statistics.peak_background import (
    peak_background_metrics,
    peak_background_trends,
    running_mean_background,
    running_mean_trends,
)
from src.statistics.seasonal_decomposition import stl_monthly_decomposition, seasonal_amplitude
from src.statistics.significance import theil_sen_with_ci, mann_kendall, p_value_to_stars

FILEPATH = DATA_DIR / MODISA_FILE

ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]
print(f"Data shape: {chl.shape}")

# ── A) Climatological P80 threshold ──────────────────────────────────
print("\n=== A) Peak / Background separation (P80 climatology) ===")
df_pb = peak_background_metrics(chl, peak_percentile=80.0)
print(df_pb.head(10))
print(f"\nMean peak Chl:  {df_pb.peak_mean_chl.mean():.4f} mg/m³")
print(f"Mean background Chl:  {df_pb.bg_mean_chl.mean():.4f} mg/m³")
print(f"Mean intensity ratio:  {df_pb.peak_intensity_ratio.mean():.2f}")
print(f"Mean peak excess:  {df_pb.peak_excess.mean():.4f} mg/m³")

print("\n=== Theil-Sen + Mann-Kendall trends (P80 split) ===")
df_pb_trends = peak_background_trends(df_pb)
print(df_pb_trends.to_string(index=False))

# ── B) Running-mean background ───────────────────────────────────────
print("\n=== B) Running-mean background (8-day, 92-step ≈ 2 yr window) ===")
df_rm = running_mean_background(chl, window_steps=92)
print(df_rm.head(10))
print(f"\nMean running-mean BG:  {df_rm.bg_total.mean():.4f} mg/m³")
print(f"Mean peak anomaly:     {df_rm.peak_anomaly_mean.mean():.4f} mg/m³")
print(f"Mean peak fraction:    {df_rm.peak_fraction.mean():.2f}")

print("\n=== Theil-Sen + Mann-Kendall trends (running-mean) ===")
df_rm_trends = running_mean_trends(df_rm)
print(df_rm_trends.to_string(index=False))

# ── STL decomposition ─────────────────────────────────────────────────
print("\n=== STL seasonal decomposition (monthly) ===")
monthly = chl.resample(time="ME").mean(skipna=True)
monthly_mean = monthly.mean(dim=["row", "col"], skipna=True)
s = pd.Series(
    monthly_mean.values.astype(float),
    index=pd.DatetimeIndex(monthly.time.values),
    dtype=float,
).dropna()
decomp = stl_monthly_decomposition(chl, period=12)
print(f"STL decomposition: {decomp.shape[0]} monthly steps")

stl_amp = seasonal_amplitude(s, period=12).dropna(subset=["amplitude"])
print(f"STL amplitude range: {stl_amp.amplitude.min():.4f} – {stl_amp.amplitude.max():.4f} mg/m³")

if len(stl_amp) >= 3:
    years_arr = stl_amp["year"].astype(float).values
    ts_amp = theil_sen_with_ci(stl_amp["amplitude"], x=years_arr)
    mk_amp = mann_kendall(stl_amp["amplitude"], x=years_arr)
    print(f"\n  STL amplitude trend: {ts_amp['slope_per_decade']:.4f} mg/m³/decade "
          f"[{ts_amp['slope_lo_per_decade']:.4f}, {ts_amp['slope_hi_per_decade']:.4f}]")
    print(f"  MK: τ={mk_amp['tau']:.3f}, p={mk_amp['p_value']:.4g} "
          f"({p_value_to_stars(mk_amp['p_value'])})")

    trend_valid = stl_amp.dropna(subset=["trend_at_year"])
    if len(trend_valid) >= 3:
        ts_trend = theil_sen_with_ci(trend_valid["trend_at_year"], x=trend_valid["year"].astype(float).values)
        mk_trend = mann_kendall(trend_valid["trend_at_year"], x=trend_valid["year"].astype(float).values)
        print(f"  STL trend component trend: {ts_trend['slope_per_decade']:.5f} mg/m³/decade "
              f"(p={mk_trend['p_value']:.4g})")

# ── Figures ───────────────────────────────────────────────────────────

years_arr_pb = df_pb["year"].astype(float).values

# Fig 8a: P80 peak/background time series
fig8a, axes = plt.subplots(3, 1, figsize=(7.2, 5), sharex=True)

ax = axes[0]
ax.plot(df_pb["year"], df_pb["peak_mean_chl"], "o-", linewidth=0.5, color="#d55e00",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="#d55e00", markersize=3,
        label="Peak months")
ax.plot(df_pb["year"], df_pb["bg_mean_chl"], "s-", linewidth=0.5, color="#3b6992",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="#3b6992", markersize=3,
        label="Background months")

for m, color in [("peak_mean_chl", "#d55e00"), ("bg_mean_chl", "#3b6992")]:
    row = df_pb_trends[df_pb_trends.metric == m]
    if not row.empty:
        tr = row.iloc[0]
        trend_line = tr["intercept"] + tr["slope_per_year"] * years_arr_pb
        ax.plot(df_pb["year"], trend_line, "--", color=color, linewidth=0.6,
                label=f"{m.replace('peak_mean_chl','Peak').replace('bg_mean_chl','BG')} "
                      f"{tr['slope']:.3g}/decade ({tr['stars']})")

ax.set_ylabel("Chl-a (mg m⁻³)")
ax.legend(frameon=False, fontsize=5, loc="upper right")
ax.tick_params(labelsize=6)

ax = axes[1]
ax.plot(df_pb["year"], df_pb["peak_intensity_ratio"], "o-", linewidth=0.5, color="#009e73",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="#009e73", markersize=3)
row = df_pb_trends[df_pb_trends.metric == "peak_intensity_ratio"] if "peak_intensity_ratio" in df_pb_trends.metric.values else pd.DataFrame()
if not row.empty:
    tr = row.iloc[0]
    trend_line = tr["intercept"] + tr["slope_per_year"] * years_arr_pb
    ax.plot(df_pb["year"], trend_line, "--", color="#d55e00", linewidth=0.6,
            label=f"{tr['slope']:.3g}/decade ({tr['stars']})")
ax.set_ylabel("Peak / BG ratio")
ax.legend(frameon=False, fontsize=5, loc="upper right")
ax.tick_params(labelsize=6)

ax = axes[2]
ax.plot(df_pb["year"], df_pb["peak_excess"], "o-", linewidth=0.5, color="#7f4f9a",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="#7f4f9a", markersize=3)
row = df_pb_trends[df_pb_trends.metric == "peak_excess"]
if not row.empty:
    tr = row.iloc[0]
    trend_line = tr["intercept"] + tr["slope_per_year"] * years_arr_pb
    ax.plot(df_pb["year"], trend_line, "--", color="#d55e00", linewidth=0.6,
            label=f"{tr['slope']:.3g}/decade ({tr['stars']})")
ax.set_ylabel("Peak excess (mg m⁻³)")
ax.set_xlabel("Year")
ax.legend(frameon=False, fontsize=5, loc="upper right")
ax.tick_params(labelsize=6)

fig8a.suptitle("Peak vs Background (P80 climatology) — Canary EBUS", fontsize=7)
plt.tight_layout()
fig8a.savefig("figures/08_peak_background_timeseries.png", dpi=300, bbox_inches="tight")
plt.close(fig8a)
print("Saved: figures/08_peak_background_timeseries.png")

# Fig 8b: Running-mean decomposition
years_arr_rm = df_rm["year"].astype(float).values
fig8b, axes = plt.subplots(3, 1, figsize=(7.2, 5), sharex=True)

ax = axes[0]
ax.plot(df_rm["year"], df_rm["obs_mean"], "o-", linewidth=0.5, color="#333333",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="#333333", markersize=3,
        label="Observed")
ax.plot(df_rm["year"], df_rm["bg_total"], "s-", linewidth=0.8, color="#3b6992",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="#3b6992", markersize=3,
        label="Running-mean BG")

for m, color in [("obs_mean", "#333333"), ("bg_total", "#3b6992")]:
    row = df_rm_trends[df_rm_trends.metric == m]
    if not row.empty:
        tr = row.iloc[0]
        trend_line = tr["intercept"] + tr["slope_per_year"] * years_arr_rm
        ax.plot(df_rm["year"], trend_line, "--", color=color, linewidth=0.6,
                label=f"{m.replace('obs_mean','Obs').replace('bg_total','BG')} "
                      f"{tr['slope']:.3g}/dec ({tr['stars']})")

ax.set_ylabel("Chl-a (mg m⁻³)")
ax.legend(frameon=False, fontsize=5, loc="upper right")
ax.tick_params(labelsize=6)

ax = axes[1]
ax.plot(df_rm["year"], df_rm["peak_anomaly_mean"], "o-", linewidth=0.5, color="#d55e00",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="#d55e00", markersize=3,
        label="Peak anomaly (+)")
row = df_rm_trends[df_rm_trends.metric == "peak_anomaly_mean"]
if not row.empty:
    tr = row.iloc[0]
    trend_line = tr["intercept"] + tr["slope_per_year"] * years_arr_rm
    ax.plot(df_rm["year"], trend_line, "--", color="#d55e00", linewidth=0.6,
            label=f"{tr['slope']:.3g}/dec ({tr['stars']})")
ax.set_ylabel("Peak anomaly (mg m⁻³)")
ax.legend(frameon=False, fontsize=5, loc="upper right")
ax.tick_params(labelsize=6)

ax = axes[2]
ax.plot(df_rm["year"], df_rm["peak_fraction"], "o-", linewidth=0.5, color="#009e73",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="#009e73", markersize=3,
        label="Peak fraction")
row = df_rm_trends[df_rm_trends.metric == "peak_fraction"]
if not row.empty:
    tr = row.iloc[0]
    trend_line = tr["intercept"] + tr["slope_per_year"] * years_arr_rm
    ax.plot(df_rm["year"], trend_line, "--", color="#009e73", linewidth=0.6,
            label=f"{tr['slope']:.3g}/dec ({tr['stars']})")
ax.set_ylabel("Fraction above BG")
ax.set_xlabel("Year")
ax.legend(frameon=False, fontsize=5, loc="upper right")
ax.tick_params(labelsize=6)

fig8b.suptitle("Running-Mean Background (24-month) — Canary EBUS", fontsize=7)
plt.tight_layout()
fig8b.savefig("figures/08_running_mean_decomposition.png", dpi=300, bbox_inches="tight")
plt.close(fig8b)
print("Saved: figures/08_running_mean_decomposition.png")

# Fig 8c: STL decomposition
if decomp is not None and not decomp.empty:
    fig8c, axes_c = plt.subplots(4, 1, figsize=(7.2, 5.5), sharex=True)
    time_idx = pd.DatetimeIndex(monthly.time.values)
    for ax_c, col, title, color in [
        (axes_c[0], "observed", "Observed", "#333333"),
        (axes_c[1], "trend", "Trend", "#d55e00"),
        (axes_c[2], "seasonal", "Seasonal", "#009e73"),
        (axes_c[3], "residual", "Residual", "#3b6992"),
    ]:
        vals = decomp[col].values if col in decomp.columns else np.full(len(time_idx), np.nan)
        ax_c.plot(time_idx, vals, linewidth=0.4, color=color)
        ax_c.set_ylabel(title, fontsize=6)
        ax_c.tick_params(labelsize=5)

    axes_c[-1].set_xlabel("Time")
    fig8c.suptitle("STL Decomposition — Monthly Mean Chl-a", fontsize=7)
    plt.tight_layout()
    fig8c.savefig("figures/08_stl_decomposition.png", dpi=300, bbox_inches="tight")
    plt.close(fig8c)
    print("Saved: figures/08_stl_decomposition.png")

# ── Save results ───────────────────────────────────────────────────────
df_pb.to_csv("results/peak_background_metrics.csv", index=False)
df_pb_trends.to_csv("results/peak_background_trends.csv", index=False)
df_rm.to_csv("results/running_mean_metrics.csv", index=False)
df_rm_trends.to_csv("results/running_mean_trends.csv", index=False)
if decomp is not None:
    decomp.to_csv("results/stl_decomposition.csv")
stl_amp.to_csv("results/stl_amplitude.csv", index=False)

print("Saved: results/peak_background_metrics.csv")
print("Saved: results/peak_background_trends.csv")
print("Saved: results/running_mean_metrics.csv")
print("Saved: results/running_mean_trends.csv")
print("Saved: results/stl_decomposition.csv")
print("Saved: results/stl_amplitude.csv")
print("Step 8 complete.")