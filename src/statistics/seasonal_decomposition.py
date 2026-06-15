"""
Seasonal decomposition for chlorophyll time series.

Uses STL (Seasonal and Trend decomposition using Loess) to separate
trend, seasonal, and residual components. The trend component captures
background evolution; the seasonal amplitude captures peak intensity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stl_decomposition(
    series: pd.Series,
    period: int = 46,
    trend_deg: int = 1,
    seasonal_deg: int = 1,
    robust: bool = True,
) -> pd.DataFrame:
    """
    Decompose a time series into trend, seasonal, and residual components
    using STL (statsmodels).

    Parameters
    ----------
    series : pd.Series
        Time series with a datetime-like index or integer index.
        Must be regularly spaced (no gaps).
    period : int
        Period of the seasonal cycle. Default 46 (≈ 46 8-day composites per
        year → 365/8 ≈ 45.6). For monthly data, use 12.
    trend_deg : int
        Polynomial degree for the trend LOESS. Default 1 (linear).
    seasonal_deg : int
        Polynomial degree for the seasonal LOESS. Default 1.
    robust : bool
        Whether to use robust fitting (resistant to outliers). Default True.

    Returns
    -------
    pd.DataFrame with columns: observed, trend, seasonal, residual
    """
    from statsmodels.tsa.seasonal import STL

    values = series.values.astype(float)
    mask = ~np.isnan(values)
    if mask.sum() < 2 * period:
        return pd.DataFrame({
            "observed": values,
            "trend": np.nan,
            "seasonal": np.nan,
            "residual": np.nan,
        }, index=series.index)

    s_clean = pd.Series(values, index=series.index, dtype=float)
    s_clean = s_clean.interpolate(method="linear").ffill().bfill()

    stl = STL(s_clean, period=period, trend_deg=trend_deg,
              seasonal_deg=seasonal_deg, robust=robust)
    result = stl.fit()

    df_out = pd.DataFrame({
        "observed": values,
        "trend": result.trend.values,
        "seasonal": result.seasonal.values,
        "residual": result.resid.values,
    }, index=series.index)

    df_out.loc[~mask, ["observed", "trend", "seasonal", "residual"]] = np.nan
    return df_out


def seasonal_amplitude(
    series: pd.Series,
    period: int = 46,
) -> pd.DataFrame:
    """
    Compute per-year seasonal amplitude from STL decomposition.

    For each year, amplitude = max(seasonal) − min(seasonal) within
    that year's composites.

    Parameters
    ----------
    series : pd.Series with datetime index
    period : int, seasonal period (default 46 for 8-day data)

    Returns
    -------
    DataFrame with columns: year, amplitude, trend_at_year
    """
    decomp = stl_decomposition(series, period=period)
    if decomp["seasonal"].isna().all():
        return pd.DataFrame(columns=["year", "amplitude", "trend_at_year"])

    years = series.index.year if hasattr(series.index, "year") else pd.Series(series.index).dt.year.values
    years_unique = np.unique(years)

    records = []
    for y in years_unique:
        idx = years == y
        seas = decomp.loc[idx, "seasonal"]
        seas_valid = seas.dropna()
        if len(seas_valid) == 0:
            records.append({"year": y, "amplitude": np.nan, "trend_at_year": np.nan})
            continue
        amp = float(seas_valid.max() - seas_valid.min())
        trend_val = float(decomp.loc[idx, "trend"].dropna().mean()) if decomp.loc[idx, "trend"].notna().any() else np.nan
        records.append({"year": y, "amplitude": amp, "trend_at_year": trend_val})

    return pd.DataFrame(records)


def stl_monthly_decomposition(
    chl: "xr.DataArray",
    period: int = 12,
) -> pd.DataFrame:
    """
    Run STL on monthly spatial-mean chlorophyll.

    Parameters
    ----------
    chl : xr.DataArray with dims (time, row, col)
    period : int, seasonal period (12 for monthly)

    Returns
    -------
    DataFrame with columns: time, observed, trend, seasonal, residual
    """
    monthly = chl.resample(time="ME").mean(skipna=True)
    spatial_mean = monthly.mean(dim=["row", "col"], skipna=True)
    s = pd.Series(spatial_mean.values, index=pd.DatetimeIndex(monthly.time.values), dtype=float)
    return stl_decomposition(s, period=period)