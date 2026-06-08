"""
Trend-analysis helpers (back-compat wrapper layer).

All new significance work lives in :mod:`src.statistics.significance`.
The functions here are thin shims that keep the legacy ``theil_sen_trend``
and ``ols_ar1_trend`` API alive while delegating to the new module, and
add a ``trend_summary_table`` helper used by the trend notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics.significance import (
    mann_kendall,
    p_value_to_stars,
    theil_sen_with_ci,
)


def theil_sen_trend(y: pd.Series, x=None) -> dict:
    """
    Back-compat wrapper. The headline ``slope`` is now **per decade**
    (matching the user-facing climate reporting convention); the raw
    per-year slope and CIs are kept as ``slope_per_year`` / ``slope_lo``
    / ``slope_hi`` for back-compat.

    ``intercept`` is the raw Theil–Sen intercept at x = 0 (use
    ``intercept + slope_per_year * year`` for the trend line).
    ``intercept_at_x0`` is the y-value at the first x (for reporting).
    """
    ts = theil_sen_with_ci(y, x=x)
    mk = mann_kendall(y, x=x)
    return {
        "slope": ts["slope_per_decade"],
        "slope_per_year": ts["slope"],
        "intercept": ts["intercept"],
        "intercept_at_x0": ts["intercept_at_x0"],
        "p_value": mk["p_value"],
        "slope_pct_per_decade": ts["slope_pct_per_decade"],
        "slope_lo": ts["slope_lo_per_decade"],
        "slope_hi": ts["slope_hi_per_decade"],
        "slope_lo_per_year": ts["slope_lo"],
        "slope_hi_per_year": ts["slope_hi"],
        "n": ts["n"],
    }


def ols_ar1_trend(y: pd.Series) -> dict:
    """OLS with Newey–West (HAC) standard errors, lag-1. Per year."""
    import statsmodels.api as sm

    x = np.arange(len(y))
    mask = ~y.isna()
    if mask.sum() < 3:
        nan = float("nan")
        return {"slope": nan, "p_value": nan, "rsquared": nan}

    x_ok = sm.add_constant(x[mask])
    y_ok = y[mask].values
    model = sm.OLS(y_ok, x_ok).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
    slope = model.params.iloc[1] if hasattr(model.params, "iloc") else model.params[1]
    p_value = model.pvalues.iloc[1] if hasattr(model.pvalues, "iloc") else model.pvalues[1]
    return {"slope": float(slope), "p_value": float(p_value),
            "rsquared": float(model.rsquared)}


def trend_summary_table(df: pd.DataFrame, year_col: str, value_cols: list[str]) -> pd.DataFrame:
    """
    Build a wide significance table.

    The headline ``slope`` column is the **per-decade** Theil–Sen slope
    (the value to cite in the manuscript). The per-year slope and the
    95% confidence intervals on both scalings are included as well.

    For each column in ``value_cols``, returns one row with:
        metric, slope (per decade), slope_per_year,
        intercept (raw, for plotting), intercept_at_x0 (for reporting),
        slope_lo (per decade), slope_hi (per decade),
        slope_pct_per_decade,
        mk_tau, mk_z, mk_p, mk_trend, stars, n
    """
    rows = []
    for col in value_cols:
        y = df[col]
        x = df[year_col].astype(float) if year_col in df.columns else None
        ts = theil_sen_with_ci(y, x=x)
        mk = mann_kendall(y, x=x)
        rows.append({
            "metric": col,
            "slope": ts["slope_per_decade"],
            "slope_per_year": ts["slope"],
            "intercept": ts["intercept"],
            "intercept_at_x0": ts["intercept_at_x0"],
            "slope_lo": ts["slope_lo_per_decade"],
            "slope_hi": ts["slope_hi_per_decade"],
            "slope_pct_per_decade": ts["slope_pct_per_decade"],
            "mk_tau": mk["tau"],
            "mk_z": mk["z"],
            "mk_p": mk["p_value"],
            "mk_trend": mk["trend"],
            "stars": p_value_to_stars(mk["p_value"]),
            "n": ts["n"],
        })
    return pd.DataFrame(rows)
