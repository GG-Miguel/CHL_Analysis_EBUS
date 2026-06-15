"""
Peak / background separation for chlorophyll time series.

Two methods are provided:

1. **Climatological threshold** — months where the climatological mean
   exceeds a percentile (default P80) are "peak months".

2. **Running-mean background** — a centred rolling window (default 24 months)
   smooths the monthly series to define the slowly-varying background;
   anomalies above the running mean are "peaks" and anomalies below are
   suppressed background.

All trends are reported per decade (x10 over the per-year rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from src.statistics.significance import (
    mann_kendall,
    p_value_to_stars,
    theil_sen_with_ci,
)


def monthly_climatology(chl: xr.DataArray) -> xr.DataArray:
    """Compute month-by-month spatial-mean climatology."""
    monthly = chl.resample(time="ME").mean(skipna=True)
    clim = monthly.groupby("time.month").mean("time", skipna=True)
    return clim.mean(dim=["row", "col"])


def peak_background_mask(
    chl: xr.DataArray,
    peak_percentile: float = 80.0,
) -> xr.DataArray:
    """
    Identify peak months in the 8-day series using a climatological threshold.

    For each 8-day composite, computes the spatial mean; then computes the
    climatological mean per month-of-year. Months where the climatological
    mean exceeds the given percentile are flagged as "peak months".

    Returns
    -------
    xr.DataArray of bool with dimension ``time``, True for peak months.
    """
    spatial_mean = chl.mean(dim=["row", "col"], skipna=True)
    monthly = spatial_mean.resample(time="ME").mean(skipna=True)
    monthly_vals = monthly.values
    months = monthly.time.dt.month.values

    threshold = float(np.nanpercentile(monthly_vals, peak_percentile))
    peak_months_of_year = set(int(m) for m in months[~np.isnan(monthly_vals)] if monthly_vals[~np.isnan(monthly_vals)][np.where(months == m)[0]].mean() >= threshold) if False else set()

    month_means = {}
    valid = ~np.isnan(monthly_vals)
    for m in range(1, 13):
        idx = months == m
        idx_valid = idx & valid
        if idx_valid.any():
            month_means[m] = float(np.nanmean(monthly_vals[idx]))
        else:
            month_means[m] = np.nan

    values = np.array([month_means.get(m, np.nan) for m in range(1, 13)])
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        threshold_val = np.nan
    else:
        threshold_val = float(np.nanpercentile(valid_values, peak_percentile))

    peak_month_set = set()
    for m in range(1, 13):
        if not np.isnan(values[m - 1]) and values[m - 1] >= threshold_val:
            peak_month_set.add(m)

    is_peak = np.isin(chl.time.dt.month.values, list(peak_month_set))
    return xr.DataArray(is_peak, dims=["time"], coords={"time": chl.time})


def peak_background_metrics(
    chl: xr.DataArray,
    peak_percentile: float = 80.0,
) -> pd.DataFrame:
    """
    Compute per-year peak and background chlorophyll statistics.

    For each year, separates 8-day composites into "peak months" and
    "background months" based on whether the month's climatological
    spatial-mean chl exceeds the given percentile.

    Returns
    -------
    DataFrame with columns:
        year, peak_mean_chl, bg_mean_chl, peak_max_chl, bg_min_chl,
        peak_months_count, peak_intensity_ratio, peak_excess,
        peak_area_fraction
    """
    spatial = chl.mean(dim=["row", "col"], skipna=True)
    spatial_vals = spatial.values
    years_all = chl.time.dt.year.values
    months_all = chl.time.dt.month.values

    monthly = chl.resample(time="ME").mean(skipna=True)
    monthly_spatial = monthly.mean(dim=["row", "col"], skipna=True)
    monthly_months = monthly.time.dt.month.values
    monthly_vals = monthly_spatial.values
    valid_m = ~np.isnan(monthly_vals)
    month_means = {}
    for m in range(1, 13):
        idx = monthly_months == m
        idx_v = idx & valid_m
        if idx_v.any():
            month_means[m] = float(np.nanmean(monthly_vals[idx_v]))
        else:
            month_means[m] = np.nan

    mvals = np.array([month_means.get(m, np.nan) for m in range(1, 13)])
    mvals_valid = mvals[~np.isnan(mvals)]
    if len(mvals_valid) == 0:
        thresh = np.nan
    else:
        thresh = float(np.nanpercentile(mvals_valid, peak_percentile))

    peak_month_set = set()
    for m in range(1, 13):
        if not np.isnan(mvals[m - 1]) and mvals[m - 1] >= thresh:
            peak_month_set.add(m)

    print(f"  Peak percentile threshold (P{peak_percentile:.0f}): {thresh:.4f} mg/m³")
    print(f"  Peak months: {sorted(peak_month_set)}")

    years_unique = np.unique(years_all)
    records = []
    for y in years_unique:
        yr_mask = years_all == y
        is_peak_month = np.isin(months_all[yr_mask], list(peak_month_set))

        peak_vals = spatial_vals[yr_mask][is_peak_month]
        bg_vals = spatial_vals[yr_mask][~is_peak_month]

        peak_valid = peak_vals[~np.isnan(peak_vals)]
        bg_valid = bg_vals[~np.isnan(bg_vals)]

        records.append({
            "year": int(y),
            "peak_mean_chl": float(np.nanmean(peak_vals)) if len(peak_valid) > 0 else np.nan,
            "bg_mean_chl": float(np.nanmean(bg_vals)) if len(bg_valid) > 0 else np.nan,
            "peak_max_chl": float(np.nanmax(peak_vals)) if len(peak_valid) > 0 else np.nan,
            "bg_min_chl": float(np.nanmin(bg_vals)) if len(bg_valid) > 0 else np.nan,
            "peak_months_count": int(is_peak_month.sum()),
            "peak_intensity_ratio": (
                float(np.nanmean(peak_vals)) / float(np.nanmean(bg_vals))
                if len(peak_valid) > 0 and len(bg_valid) > 0 and float(np.nanmean(bg_vals)) > 0
                else np.nan
            ),
            "peak_excess": (
                float(np.nanmean(peak_vals)) - float(np.nanmean(bg_vals))
                if len(peak_valid) > 0 and len(bg_valid) > 0
                else np.nan
            ),
        })

    df = pd.DataFrame(records)

    total_composites = sum(r["peak_months_count"] for r in records)
    total_all = len(spatial_vals)
    df["peak_area_fraction"] = total_composites / total_all if total_all > 0 else np.nan

    return df


def running_mean_background(
    chl: xr.DataArray,
    window_steps: int = 92,
) -> pd.DataFrame:
    """
    Separate peak from background using a centred running mean on
    the raw 8-day composites.

    The spatial-mean chlorophyll at native 8-day cadence is smoothed with
    a centred rolling window of ``window_steps`` composites (default 92
    ≈ 2 years × 46 composites/year) to extract the slowly-varying
    background. The residual (observed − running_mean) is the anomaly;
    positive anomalies are "peak" contributions and the running mean
    itself is the background.

    Parameters
    ----------
    chl : xr.DataArray
        Chlorophyll with dims (time, row, col).
    window_steps : int
        Width of the centred rolling window in 8-day composites
        (default 92 ≈ 2 years).

    Returns
    -------
    DataFrame with columns:
        year, bg_total (annual mean of running background),
        anomaly_mean, peak_anomaly_mean (mean positive anomaly),
        peak_fraction (fraction of composites above background),
        peak_excess, obs_mean, obs_max
    """
    spatial = chl.mean(dim=["row", "col"], skipna=True)
    s = pd.Series(
        spatial.values.astype(float),
        index=pd.DatetimeIndex(chl.time.values),
        dtype=float,
    )
    s = s.dropna()

    bg = s.rolling(window=window_steps, center=True, min_periods=window_steps // 2).mean()
    anomaly = s - bg

    years = s.index.year
    years_unique = sorted(set(years))

    records = []
    for y in years_unique:
        mask = years == y
        bg_y = bg[mask].dropna()
        anom_y = anomaly[mask].dropna()
        s_y = s[mask].dropna()

        if len(bg_y) == 0:
            continue

        positive_anom = anom_y[anom_y > 0]
        records.append({
            "year": y,
            "bg_total": float(bg_y.mean()),
            "anomaly_mean": float(anom_y.mean()) if len(anom_y) > 0 else np.nan,
            "peak_anomaly_mean": float(positive_anom.mean()) if len(positive_anom) > 0 else np.nan,
            "peak_fraction": float((anom_y > 0).sum() / len(anom_y)) if len(anom_y) > 0 else np.nan,
            "peak_excess": float(positive_anom.mean()) if len(positive_anom) > 0 else np.nan,
            "obs_mean": float(s_y.mean()),
            "obs_max": float(s_y.max()),
        })

    return pd.DataFrame(records)


def running_mean_trends(
    df: pd.DataFrame,
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Theil-Sen + Mann-Kendall trends on running-mean decomposition metrics.
    """
    metrics = [
        "bg_total", "peak_anomaly_mean", "peak_excess",
        "peak_fraction", "obs_mean", "obs_max",
    ]
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        y = df[m].dropna()
        if len(y) < 3:
            continue
        x = df.loc[y.index, year_col].astype(float).values
        ts = theil_sen_with_ci(y, x=x)
        mk = mann_kendall(y, x=x)
        rows.append({
            "metric": m,
            "slope": ts["slope_per_decade"],
            "slope_per_year": ts["slope"],
            "intercept": ts["intercept"],
            "intercept_at_x0": ts["intercept_at_x0"],
            "slope_lo": ts["slope_lo_per_decade"],
            "slope_hi": ts["slope_hi_per_decade"],
            "slope_pct_per_decade": ts["slope_pct_per_decade"],
            "mk_tau": mk["tau"],
            "mk_p": mk["p_value"],
            "stars": p_value_to_stars(mk["p_value"]),
            "n": ts["n"],
        })
    return pd.DataFrame(rows)


def peak_background_trends(
    df: pd.DataFrame,
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Compute Theil-Sen + Mann-Kendall trends for peak and background metrics.

    Parameters
    ----------
    df : DataFrame from ``peak_background_metrics``.
    year_col : str, name of the year column.

    Returns
    -------
    DataFrame with one row per metric and columns:
        metric, slope (decade), slope_per_year, intercept, intercept_at_x0,
        slope_lo, slope_hi, slope_pct_per_decade, mk_tau, mk_p, stars, n
    """
    metrics = ["peak_mean_chl", "bg_mean_chl", "peak_max_chl", "peak_intensity_ratio", "peak_excess"]
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        y = df[m].dropna()
        if len(y) < 3:
            continue
        x = df.loc[y.index, year_col].astype(float).values
        ts = theil_sen_with_ci(y, x=x)
        mk = mann_kendall(y, x=x)
        rows.append({
            "metric": m,
            "slope": ts["slope_per_decade"],
            "slope_per_year": ts["slope"],
            "intercept": ts["intercept"],
            "intercept_at_x0": ts["intercept_at_x0"],
            "slope_lo": ts["slope_lo_per_decade"],
            "slope_hi": ts["slope_hi_per_decade"],
            "slope_pct_per_decade": ts["slope_pct_per_decade"],
            "mk_tau": mk["tau"],
            "mk_p": mk["p_value"],
            "stars": p_value_to_stars(mk["p_value"]),
            "n": ts["n"],
        })
    return pd.DataFrame(rows)