"""
# 07 — Quantile Regression: Peak vs Background Trends

Fits quantile regression (τ = 0.1, 0.25, 0.5, 0.75, 0.9, 0.95) on
both raw and log10-transformed spatial-mean chlorophyll. Low quantiles
capture background trends; high quantiles capture peak trends.

The divergence between high and low quantile slopes reveals whether
upwelling peaks are intensifying (or weakening) independently of the
background chlorophyll field.
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
from src.statistics.quantile_trends import (
    quantile_trend,
    quantile_trend_log10,
    peak_background_divergence,
    DEFAULT_QUANTILES,
)
from src.statistics.significance import theil_sen_with_ci, mann_kendall, p_value_to_stars

FILEPATH = DATA_DIR / MODISA_FILE

ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]

spatial_mean = chl.mean(dim=["row", "col"], skipna=True).values
time_vals = chl.time.values
years_8d = pd.DatetimeIndex(time_vals).year.astype(float) + (
    pd.DatetimeIndex(time_vals).dayofyear / 365.25
)

df_8d = pd.DataFrame({
    "time": time_vals,
    "year_frac": years_8d,
    "chl": spatial_mean.astype(float),
    "log10_chl": np.log10(np.where(spatial_mean > 0, spatial_mean, np.nan)).astype(float),
}).dropna(subset=["chl"])

print(f"8-day composites: {len(df_8d)} steps, {df_8d.year_frac.min():.1f} – {df_8d.year_frac.max():.1f}")

# --- Quantile regression on raw chl ---
print("\n=== Quantile regression on raw Chl (8-day) ===")
qr_raw = quantile_trend(df_8d["chl"], x=df_8d["year_frac"].values)
print(qr_raw.to_string(index=False))

# --- Quantile regression on log10(chl) ---
print("\n=== Quantile regression on log10(Chl) (8-day) ===")
qr_log = quantile_trend_log10(df_8d["chl"], x=df_8d["year_frac"].values)
print(qr_log.to_string(index=False))

# --- Peak-background divergence ---
div_raw = peak_background_divergence(qr_raw)
div_log = peak_background_divergence(qr_log)
print(f"\n=== Peak-background divergence (raw Chl) ===")
print(f"  Background (Q25) slope:  {div_raw['background_slope']:.6f} mg/m³/yr")
print(f"  Peak (Q90) slope:        {div_raw['peak_slope']:.6f} mg/m³/yr")
print(f"  Divergence (Q90-Q25):   {div_raw['divergence']:.6f} mg/m³/yr")
print(f"\n=== Peak-background divergence (log10 Chl) ===")
print(f"  Background (Q25):  {div_log['background_pct_per_decade']:.2f} %/decade")
print(f"  Peak (Q90):        {div_log['peak_pct_per_decade']:.2f} %/decade")
print(f"  Amplification ratio: {div_log['amplification_ratio']:.2f}")

# --- Annual aggregation: monthly mean chl → quantile regression ---
monthly = chl.resample(time="ME").mean(skipna=True)
monthly_mean = monthly.mean(dim=["row", "col"], skipna=True).values
monthly_time = monthly.time.values
monthly_year = pd.DatetimeIndex(monthly_time).year.astype(float) + (
    pd.DatetimeIndex(monthly_time).month.values - 0.5
) / 12.0

df_monthly = pd.DataFrame({
    "time": monthly_time,
    "year_frac": monthly_year,
    "chl": monthly_mean.astype(float),
    "log10_chl": np.log10(np.where(monthly_mean > 0, monthly_mean, np.nan)).astype(float),
}).dropna(subset=["chl"])

print(f"\nMonthly composites: {len(df_monthly)} steps")

qr_monthly_raw = quantile_trend(df_monthly["chl"], x=df_monthly["year_frac"].values)
qr_monthly_log = quantile_trend_log10(df_monthly["chl"], x=df_monthly["year_frac"].values)

# --- Figures ---

# Fig 7a: quantile slopes (raw) across quantiles
QUANTILE_COLORS = {
    0.10: "#2166ac", 0.25: "#67a9cf", 0.50: "#333333",
    0.75: "#ef8a62", 0.90: "#b2182b", 0.95: "#8c0a0a",
}

fig7a, ax7a = plt.subplots(figsize=(3.5, 2.5))
x_pos = qr_raw["quantile"].values
slopes = qr_raw["slope_per_decade"].values
lo = qr_raw["slope_lo"].values
hi = qr_raw["slope_hi"].values
err_lo = np.abs(slopes - lo)
err_hi = np.abs(hi - slopes)
colors = [QUANTILE_COLORS.get(q, "#888888") for q in x_pos]
ax7a.bar(x_pos, slopes, width=0.06, color=colors, edgecolor="white", linewidth=0.3)
ax7a.errorbar(x_pos, slopes, yerr=[err_lo, err_hi],
              fmt="none", ecolor="#222222", elinewidth=0.4, capsize=1.5, capthick=0.4)
ax7a.axhline(0, color="#666666", linewidth=0.4)
ax7a.set_xlabel("Quantile τ")
ax7a.set_ylabel("Slope (mg m⁻³ per decade)")
ax7a.set_title("Quantile regression — Raw Chl-a", fontsize=7)
ax7a.tick_params(labelsize=6)
plt.tight_layout()
fig7a.savefig("figures/07_quantile_slopes_raw.png", dpi=300, bbox_inches="tight")
plt.close(fig7a)
print("Saved: figures/07_quantile_slopes_raw.png")

# Fig 7b: quantile slopes (log10) → %/decade
fig7b, ax7b = plt.subplots(figsize=(3.5, 2.5))
pct_dec = qr_monthly_log["pct_per_decade"].values
lo_log = qr_monthly_log["slope_lo"].values
hi_log = qr_monthly_log["slope_hi"].values
pct_lo = (10 ** lo_log - 1) * 1000
pct_hi = (10 ** hi_log - 1) * 1000
err_lo_pct = np.abs(pct_dec - pct_lo)
err_hi_pct = np.abs(pct_hi - pct_dec)
colors_log = [QUANTILE_COLORS.get(q, "#888888") for q in qr_monthly_log["quantile"].values]
ax7b.bar(qr_monthly_log["quantile"].values, pct_dec, width=0.06,
          color=colors_log, edgecolor="white", linewidth=0.3)
ax7b.errorbar(qr_monthly_log["quantile"].values, pct_dec,
              yerr=[err_lo_pct, err_hi_pct],
              fmt="none", ecolor="#222222", elinewidth=0.4, capsize=1.5, capthick=0.4)
ax7b.axhline(0, color="#666666", linewidth=0.4)
ax7b.set_xlabel("Quantile τ")
ax7b.set_ylabel("% change per decade")
ax7b.set_title("Quantile regression — log₁₀(Chl-a)", fontsize=7)
ax7b.tick_params(labelsize=6)
plt.tight_layout()
fig7b.savefig("figures/07_quantile_slopes_log10.png", dpi=300, bbox_inches="tight")
plt.close(fig7b)
print("Saved: figures/07_quantile_slopes_log10.png")

# Fig 7c: time series with quantile trend lines (monthly)
fig7c, ax7c = plt.subplots(figsize=(7.2, 2.5))
ax7c.plot(df_monthly["year_frac"], df_monthly["chl"], ".", color="#aaaaaa",
           markersize=1, alpha=0.5, label="monthly mean")

x_line = np.linspace(df_monthly["year_frac"].min(), df_monthly["year_frac"].max(), 200)
for _, row in qr_monthly_raw.iterrows():
    q = row["quantile"]
    slope = row["slope"]
    intercept = row["intercept"]
    color = QUANTILE_COLORS.get(q, "#888888")
    y_line = intercept + slope * x_line
    ax7c.plot(x_line, y_line, "-", color=color, linewidth=0.8,
              label=f"τ={q:.2f}")

ax7c.set_ylabel("Chl-a (mg m⁻³)")
ax7c.set_xlabel("Year")
ax7c.set_title("Quantile regression trends — monthly mean Chl-a", fontsize=7)
ax7c.legend(frameon=False, fontsize=5, ncol=3, loc="upper right")
ax7c.tick_params(labelsize=6)
plt.tight_layout()
fig7c.savefig("figures/07_quantile_timeseries.png", dpi=300, bbox_inches="tight")
plt.close(fig7c)
print("Saved: figures/07_quantile_timeseries.png")

# --- Save results ---
qr_raw.to_csv("results/quantile_trends_raw_8d.csv", index=False)
qr_log.to_csv("results/quantile_trends_log10_8d.csv", index=False)
qr_monthly_raw.to_csv("results/quantile_trends_raw_monthly.csv", index=False)
qr_monthly_log.to_csv("results/quantile_trends_log10_monthly.csv", index=False)

div_summary = pd.DataFrame([{
    **{f"raw_{k}": v for k, v in div_raw.items()},
    **{f"log10_{k}": v for k, v in div_log.items()},
}])
div_summary.to_csv("results/peak_background_divergence.csv", index=False)
print("Saved: results/quantile_trends_*.csv")
print("Saved: results/peak_background_divergence.csv")
print("Step 7 complete.")