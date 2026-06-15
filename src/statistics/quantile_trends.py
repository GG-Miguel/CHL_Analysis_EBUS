"""
Quantile regression for separating peak vs background chlorophyll trends.

Fits linear trends at multiple quantiles (τ = 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
using statsmodels QuantReg. Low quantiles capture background trends; high
quantiles capture peak trends. Slopes in log10-space are convertible to
percent-per-year changes.

All slopes are reported per decade (×10) for climate reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


DEFAULT_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def quantile_trend(
    y: pd.Series,
    x: np.ndarray | None = None,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """
    Fit quantile regression at multiple quantiles.

    Parameters
    ----------
    y : pd.Series
        Response variable (e.g., log10(chl) or raw chl per 8-day step).
    x : array-like, optional
        Time coordinate in years. Defaults to 0, 1, 2, … .
    quantiles : list of float, optional
        Quantiles at which to fit. Defaults to [0.1, 0.25, 0.5, 0.75, 0.9, 0.95].

    Returns
    -------
    pd.DataFrame with columns:
        quantile, slope, slope_per_decade, intercept, intercept_at_x0,
        slope_lo (95% CI), slope_hi (95% CI), p_value, n
    """
    quantiles = quantiles or DEFAULT_QUANTILES

    arr = np.asarray(y, dtype=float)
    mask = ~np.isnan(arr)
    y_ok = arr[mask]
    n = int(mask.sum())
    if n < 3:
        rows = []
        for q in quantiles:
            rows.append({
                "quantile": q, "slope": np.nan, "slope_per_decade": np.nan,
                "intercept": np.nan, "intercept_at_x0": np.nan,
                "slope_lo": np.nan, "slope_hi": np.nan, "p_value": np.nan, "n": n,
            })
        return pd.DataFrame(rows)

    if x is None:
        x_ok = np.arange(n, dtype=float)
    else:
        x_ok = np.asarray(x, dtype=float)[mask]

    X = sm.add_constant(x_ok)
    rows = []
    for q in quantiles:
        try:
            mod = sm.QuantReg(y_ok, X)
            res = mod.fit(q=q, max_iter=1000)
            slope = float(res.params[1])
            intercept = float(res.params[0])
            p_value = float(res.pvalues[1])
            ci = res.conf_int(alpha=0.05)
            slope_lo = float(ci[1, 0])
            slope_hi = float(ci[1, 1])
        except Exception:
            slope = np.nan
            intercept = np.nan
            p_value = np.nan
            slope_lo = np.nan
            slope_hi = np.nan

        x0 = float(x_ok[0])
        intercept_at_x0 = intercept + slope * x0 if not np.isnan(intercept) else np.nan

        rows.append({
            "quantile": q,
            "slope": slope,
            "slope_per_decade": slope * 10 if not np.isnan(slope) else np.nan,
            "intercept": intercept,
            "intercept_at_x0": intercept_at_x0,
            "slope_lo": slope_lo,
            "slope_hi": slope_hi,
            "p_value": p_value,
            "n": n,
        })

    return pd.DataFrame(rows)


def quantile_trend_log10(
    chl_series: pd.Series,
    x: np.ndarray | None = None,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """
    Quantile regression on log10(chl).

    Same as ``quantile_trend`` but applies log10 to the series first
    (dropping zeros/negatives). The slope in log10-units per year can be
    converted to approximate percent change per year:

        %/year ≈ (10^slope − 1) × 100

    Returns an additional column ``pct_per_year`` and ``pct_per_decade``.
    """
    quantiles = quantiles or DEFAULT_QUANTILES
    y = np.log10(chl_series[chl_series > 0])
    df = quantile_trend(y, x=x, quantiles=quantiles)
    df["pct_per_year"] = (10 ** df["slope"] - 1) * 100
    df["pct_per_decade"] = df["pct_per_year"] * 10
    return df


def peak_background_divergence(
    df: pd.DataFrame,
    low_quantile: float = 0.25,
    high_quantile: float = 0.90,
) -> dict:
    """
    Compute the divergence between peak (high quantile) and background
    (low quantile) trends from a quantile_trend result DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``quantile_trend`` or ``quantile_trend_log10``.
    low_quantile, high_quantile : float
        Quantiles representing background and peak, respectively.

    Returns
    -------
    dict with keys:
        background_slope, peak_slope, divergence,
        background_pct_per_decade, peak_pct_per_decade,
        amplification_ratio (peak_pct / background_pct)
    """
    low = df[df["quantile"] == low_quantile]
    high = df[df["quantile"] == high_quantile]

    if low.empty or high.empty:
        return {
            "background_slope": np.nan, "peak_slope": np.nan,
            "divergence": np.nan,
            "background_pct_per_decade": np.nan,
            "peak_pct_per_decade": np.nan,
            "amplification_ratio": np.nan,
        }

    bg_slope = float(low.iloc[0]["slope"])
    pk_slope = float(high.iloc[0]["slope"])
    divergence = pk_slope - bg_slope

    bg_pct = float(low.iloc[0].get("pct_per_decade", np.nan))
    pk_pct = float(high.iloc[0].get("pct_per_decade", np.nan))

    bg_pct_yr = float(low.iloc[0].get("pct_per_year", np.nan))
    pk_pct_yr = float(high.iloc[0].get("pct_per_year", np.nan))
    ratio = pk_pct_yr / bg_pct_yr if bg_pct_yr != 0 and not np.isnan(bg_pct_yr) and not np.isnan(pk_pct_yr) else np.nan

    return {
        "background_slope": bg_slope,
        "peak_slope": pk_slope,
        "divergence": divergence,
        "background_pct_per_decade": bg_pct,
        "peak_pct_per_decade": pk_pct,
        "amplification_ratio": ratio,
    }