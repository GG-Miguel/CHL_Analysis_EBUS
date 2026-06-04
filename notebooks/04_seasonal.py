"""
# 04 — Seasonal Cycle Analysis
Computes the 8-day and monthly climatologies, the per-year seasonal
amplitude and peak day-of-year, and trend-tests these (Theil–Sen +
Mann–Kendall).
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
from src.statistics.significance import (
    mann_kendall,
    theil_sen_with_ci,
    circular_doy_trend,
    p_value_to_stars,
)

FILEPATH = DATA_DIR / MODISA_FILE

ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]

print(f"Data shape: {chl.shape}")

# --- 8-day climatology (full record) ---
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

peak_doy_clim = doys[np.nanargmax(spatial_mean_clim.values)]
print(f"Climatology peak DOY: {peak_doy_clim}")

# --- Monthly climatology ---
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
print(f"Climatology seasonal amplitude: {amplitude:.3f} mg/m³")

# --- Per-year amplitude and peak-DOY from each year's 8-day climatology ---
years = np.unique(chl.time.dt.year.values)
year_rows = []
for y in years:
    ycl = chl.sel(time=str(y))
    clim = ycl.groupby("time.dayofyear").mean("time", skipna=True)
    smean = clim.mean(dim=["row", "col"], skipna=True)
    amp = float(smean.max().values - smean.min().values)
    p_doy = int(smean.dayofyear.values[np.nanargmax(smean.values)])
    year_rows.append({"year": y, "amplitude_mgm3": amp, "peak_doy": p_doy})
df_seas = pd.DataFrame(year_rows)

ts_amp = theil_sen_with_ci(df_seas["amplitude_mgm3"])
mk_amp = mann_kendall(df_seas["amplitude_mgm3"])
circ_peak = circular_doy_trend(df_seas["peak_doy"])

print("\n=== Seasonal amplitude trend ===")
print(f"  Theil–Sen slope:  {ts_amp['slope']:.4f} mg/m³/yr  "
      f"[{ts_amp['slope_lo']:.4f}, {ts_amp['slope_hi']:.4f}]")
print(f"  % per decade:     {ts_amp['slope_pct_per_decade']:.2f} %")
print(f"  Mann–Kendall:     τ={mk_amp['tau']:.3f}, p={mk_amp['p_value']:.4g}, "
      f"trend={mk_amp['trend']} ({p_value_to_stars(mk_amp['p_value'])})")

print("\n=== Per-year peak DOY trend (circular) ===")
print(f"  Mean direction:   {circ_peak['mean_direction_deg']:.1f}° (DOY "
      f"{circ_peak['mean_direction_deg']*365.0/360.0:.0f})")
print(f"  Drift:            {circ_peak['drift_deg_per_year']:.2f}°/yr "
      f"({circ_peak['drift_deg_per_year']*365.0/360.0:.2f} DOY/yr)")
print(f"  Rayleigh p:       {circ_peak['rayleigh_p']:.4g}")

# --- Two-panel figure: amplitude + peak-DOY ---
fig3, axes = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True)

ax = axes[0]
ax.plot(df_seas.year, df_seas.amplitude_mgm3, "o-", linewidth=0.5, color="#009e73",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="#009e73", markersize=3)
trend_line = ts_amp["intercept"] + ts_amp["slope"] * np.arange(len(df_seas))
ax.plot(df_seas.year, trend_line, "--", color="#d55e00", linewidth=0.5,
        label=f"slope={ts_amp['slope']:.3f}/yr ({p_value_to_stars(mk_amp['p_value'])})")
ax.set_ylabel("Seasonal amplitude (mg m⁻³)")
ax.legend(frameon=False, fontsize=5, loc="upper left")
ax.tick_params(labelsize=6)

ax = axes[1]
ax.plot(df_seas.year, df_seas.peak_doy, "o-", linewidth=0.5, color="#7f4f9a",
        markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="#7f4f9a", markersize=3)
mean_doy = circ_peak["mean_direction_deg"] * 365.0 / 360.0
ax.axhline(mean_doy, color="#d55e00", linewidth=0.5, linestyle="--",
           label=f"mean DOY ≈ {mean_doy:.0f}")
ax.set_ylabel("Peak DOY")
ax.set_xlabel("Year")
ax.legend(frameon=False, fontsize=5, loc="upper left")
ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig("figures/04_seasonal_trend.png", dpi=300, bbox_inches="tight")
plt.close(fig3)

# --- Save summary ---
df_seas["amp_slope"] = ts_amp["slope"]
df_seas["amp_slope_lo"] = ts_amp["slope_lo"]
df_seas["amp_slope_hi"] = ts_amp["slope_hi"]
df_seas["amp_mk_tau"] = mk_amp["tau"]
df_seas["amp_mk_p"] = mk_amp["p_value"]
df_seas["amp_mk_trend"] = mk_amp["trend"]
df_seas["peak_doy_drift_deg_per_yr"] = circ_peak["drift_deg_per_year"]
df_seas["peak_doy_rayleigh_p"] = circ_peak["rayleigh_p"]

# Write a single-row summary plus the per-year data
summary_row = {
    "metric": "seasonal_amplitude_mgm3",
    "slope": ts_amp["slope"],
    "slope_lo": ts_amp["slope_lo"],
    "slope_hi": ts_amp["slope_hi"],
    "slope_pct_per_decade": ts_amp["slope_pct_per_decade"],
    "mk_tau": mk_amp["tau"],
    "mk_p": mk_amp["p_value"],
    "mk_trend": mk_amp["trend"],
    "stars": p_value_to_stars(mk_amp["p_value"]),
    "n": ts_amp["n"],
}
peak_row = {
    "metric": "per_year_peak_doy",
    "slope": circ_peak["drift_deg_per_year"],
    "slope_lo": np.nan,
    "slope_hi": np.nan,
    "slope_pct_per_decade": np.nan,
    "mk_tau": np.nan,
    "mk_p": circ_peak["rayleigh_p"],
    "mk_trend": "circular (Rayleigh)",
    "stars": p_value_to_stars(circ_peak["rayleigh_p"]),
    "n": circ_peak["n"],
}
pd.DataFrame([summary_row, peak_row]).to_csv("results/seasonal_trend.csv", index=False)
df_seas.to_csv("results/seasonal_per_year.csv", index=False)

print("Saved: results/seasonal_trend.csv")
print("Saved: results/seasonal_per_year.csv")
print("Saved: figures/04_seasonal_trend.png")
print("Step 4 complete.")
