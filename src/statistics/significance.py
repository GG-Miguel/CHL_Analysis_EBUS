"""
Statistical significance helpers for interannual trend and patch-metric
analysis.

All functions accept a pandas Series / 1-D array-like for `y` and an
optional `x` array of the same length giving the time coordinate
(in years, e.g. 2002.0, 2003.0, ..., 2025.0). The original data lives at
``datetime64[ns]`` precision at 8-day cadence, but the trend is taken
across annual aggregates, so the natural x-axis is "calendar year".
Passing the actual year (not the integer index) is what makes
``slope`` express "units of y per year" — and ``slope_per_decade``
("units of y per decade") is then ``slope * 10``.

Conventions
-----------
* ``slope`` is reported in original units of `y` per "year step" (i.e.
  per unit of the supplied ``x``, which is expected to be in years).
* ``slope_per_decade`` is ``slope * 10`` and is the headline number for
  climate reporting.
* ``slope_pct_per_decade`` is computed as
  ``100 * (slope / median(y)) * 10`` — the relative change per decade.
* ``intercept_at_x0`` is the trend line evaluated at the first ``x``
  value (i.e. a "starting value" that is much more interpretable than
  the raw Theil–Sen intercept at ``x = 0``).
* Mann–Kendall uses ``pymannkendall`` (Hamed & Rao 1998 variance
  correction is applied automatically when ties are present); the
  per-year slope reported alongside the MK statistics is re-computed
  with Theil–Sen using the supplied ``x`` (pymannkendall itself
  assumes integer-spaced x).
* For circular variables (e.g. day-of-year) we use cos/sin decomposition
  + Theil–Sen on each component, then convert the drift vector back to
  a direction in degrees.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

import pymannkendall as mk

from skimage import measure


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _resolve_x(x: Optional[Iterable], y_clean_len: int, full_len: int) -> np.ndarray:
    """
    Return x as a float array.

    If ``x`` is None, the cleaned y gets a fresh 0, 1, 2, ... index
    (i.e. a fresh integer index for the NaN-dropped series).
    If ``x`` is given, it is used as-is; the caller is expected to have
    already applied the same mask that was applied to y.
    """
    if x is None:
        return np.arange(y_clean_len, dtype=float)
    return np.asarray(x, dtype=float)


def _nan_dict() -> dict:
    nan = float("nan")
    return {
        "slope": nan, "intercept": nan, "intercept_at_x0": nan,
        "slope_lo": nan, "slope_hi": nan,
        "slope_per_decade": nan,
        "slope_lo_per_decade": nan, "slope_hi_per_decade": nan,
        "slope_pct_per_decade": nan, "sign_p_value": nan, "n": 0,
    }


# ---------------------------------------------------------------------------
# Theil–Sen with confidence interval
# ---------------------------------------------------------------------------
def theil_sen_with_ci(
    y: pd.Series,
    x: Optional[Iterable] = None,
    alpha: float = 0.05,
) -> dict:
    """
    Theil–Sen slope with 95% confidence interval.

    Parameters
    ----------
    y : pd.Series
        Annual time series. NaNs are dropped (along with the corresponding
        entries in ``x``).
    x : array-like, optional
        Time coordinate in the same units you want the slope to be
        expressed in. For climate work, pass the actual year
        (e.g. 2002, 2003, ...) so the slope is "units of y per year".
        Defaults to ``np.arange(len(y))``.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict with keys
        slope, intercept, intercept_at_x0,
        slope_lo, slope_hi,
        slope_per_decade, slope_lo_per_decade, slope_hi_per_decade,
        slope_pct_per_decade, sign_p_value, n
    """
    arr = np.asarray(y, dtype=float)
    mask = ~np.isnan(arr)
    n = int(mask.sum())
    if n < 3:
        return _nan_dict()

    y_ok = arr[mask]
    if x is None:
        x_ok = np.arange(n, dtype=float)
    else:
        x_ok = np.asarray(x, dtype=float)[mask]

    res = stats.theilslopes(y_ok, x_ok, alpha=alpha)
    if len(res) == 5:
        slope, intercept, slope_lo, slope_hi, sign_p = res
    else:
        slope, intercept, slope_lo, slope_hi = res
        sign_p = float("nan")

    # Per-decade scaling (assumes x is in years — the climatology use case)
    slope_pd = float(slope) * 10.0
    slope_lo_pd = float(slope_lo) * 10.0
    slope_hi_pd = float(slope_hi) * 10.0

    # Intercept at the first x value (more interpretable than at x = 0)
    x0 = float(x_ok[0])
    intercept_at_x0 = float(intercept + slope * x0)

    med = float(np.nanmedian(y_ok))
    slope_pct = 100.0 * (slope_pd / med) if med > 0 else float("nan")

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "intercept_at_x0": intercept_at_x0,
        "slope_lo": float(slope_lo),
        "slope_hi": float(slope_hi),
        "slope_per_decade": slope_pd,
        "slope_lo_per_decade": slope_lo_pd,
        "slope_hi_per_decade": slope_hi_pd,
        "slope_pct_per_decade": float(slope_pct),
        "sign_p_value": float(sign_p),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Mann–Kendall
# ---------------------------------------------------------------------------
def mann_kendall(
    y: pd.Series,
    x: Optional[Iterable] = None,
    alpha: float = 0.05,
) -> dict:
    """
    Two-sided Mann–Kendall trend test (Hamed & Rao 1998 variance) with
    Theil–Sen slope re-computed against the supplied ``x`` so the
    slope is expressed in "units of y per year" (and per-decade via
    ``slope_per_decade``).

    Parameters
    ----------
    y : pd.Series
    x : array-like, optional
        Time coordinate in years. Defaults to ``np.arange(len(y))``.
    alpha : float

    Returns
    -------
    dict with keys
        tau, z, p_value, h, trend,
        slope, intercept, intercept_at_x0,
        slope_lo, slope_hi,
        slope_per_decade, slope_lo_per_decade, slope_hi_per_decade,
        n
    """
    arr_full = np.asarray(y, dtype=float)
    mask = ~np.isnan(arr_full)
    arr = arr_full[mask]
    n = int(arr.size)
    if n < 4:
        nan = float("nan")
        return {
            "tau": nan, "z": nan, "p_value": nan,
            "h": False, "trend": "no trend",
            "slope": nan, "intercept": nan, "intercept_at_x0": nan,
            "slope_lo": nan, "slope_hi": nan,
            "slope_per_decade": nan,
            "slope_lo_per_decade": nan, "slope_hi_per_decade": nan,
            "n": n,
        }

    res = mk.original_test(arr, alpha=alpha)

    # pymannkendall assumes integer-spaced x; recompute Theil–Sen with
    # the user's x so the slope is in "units of y per year".
    if x is None:
        x_ok = np.arange(n, dtype=float)
    else:
        x_ok = np.asarray(x, dtype=float)[mask]
    slope, intercept, slope_lo, slope_hi = stats.theilslopes(arr, x_ok, alpha=alpha)

    x0 = float(x_ok[0])
    intercept_at_x0 = float(intercept + slope * x0)

    return {
        "tau": float(res.Tau),
        "z": float(res.z),
        "p_value": float(res.p),
        "h": bool(res.h),
        "trend": str(res.trend),
        "slope": float(slope),
        "intercept": float(intercept),
        "intercept_at_x0": intercept_at_x0,
        "slope_lo": float(slope_lo),
        "slope_hi": float(slope_hi),
        "slope_per_decade": float(slope) * 10.0,
        "slope_lo_per_decade": float(slope_lo) * 10.0,
        "slope_hi_per_decade": float(slope_hi) * 10.0,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Circular day-of-year trend
# ---------------------------------------------------------------------------
def circular_doy_trend(
    peak_doy: pd.Series,
    x: Optional[Iterable] = None,
) -> dict:
    """
    Trend test on a circular (day-of-year) variable.

    DOY is converted to angle θ = 2π·DOY/365; Theil–Sen is fitted on
    cos θ and sin θ independently with the supplied ``x`` (in years).
    The combined drift vector is then converted to degrees.

    Significance of the mean direction is tested with a Rayleigh test
    (uniform-phase null). Drift magnitude is given in degrees per year
    and degrees per decade.

    Parameters
    ----------
    peak_doy : pd.Series
        Annual peak day-of-year (1–366). NaNs dropped.
    x : array-like, optional
        Time coordinate in years. Defaults to ``np.arange(len(y))``.

    Returns
    -------
    dict with keys
        mean_direction_deg, mean_doy,
        drift_deg_per_year, drift_deg_per_decade,
        drift_doy_per_year, drift_doy_per_decade,
        cos_slope, sin_slope, rayleigh_p, n
    """
    arr_full = np.asarray(peak_doy, dtype=float)
    mask = ~np.isnan(arr_full)
    arr = arr_full[mask]
    n = int(arr.size)
    if n < 3:
        nan = float("nan")
        return {
            "mean_direction_deg": nan, "mean_doy": nan,
            "drift_deg_per_year": nan, "drift_deg_per_decade": nan,
            "drift_doy_per_year": nan, "drift_doy_per_decade": nan,
            "cos_slope": nan, "sin_slope": nan,
            "rayleigh_p": nan, "n": n,
        }

    if x is None:
        x_ok = np.arange(n, dtype=float)
    else:
        x_ok = np.asarray(x, dtype=float)[mask]

    theta = 2.0 * np.pi * (arr % 365.0) / 365.0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    cos_slope, _, _, _ = stats.theilslopes(cos_t, x_ok)
    sin_slope, _, _, _ = stats.theilslopes(sin_t, x_ok)

    # Mean direction of the cloud of points
    mean_cos = cos_t.mean()
    mean_sin = sin_t.mean()
    mean_dir_rad = np.arctan2(mean_sin, mean_cos)
    mean_dir_deg = (np.degrees(mean_dir_rad)) % 360.0
    mean_doy = mean_dir_deg * 365.0 / 360.0

    # Drift vector (per unit of x — which is in years here)
    drift_rad_per_year = np.hypot(cos_slope, sin_slope)
    drift_deg_per_year = float(np.degrees(drift_rad_per_year))
    drift_deg_per_decade = drift_deg_per_year * 10.0
    drift_doy_per_year = drift_deg_per_year * 365.0 / 360.0
    drift_doy_per_decade = drift_deg_per_year * 10.0 * 365.0 / 360.0

    # Rayleigh test (uniform-phase null)
    R = np.hypot(mean_cos, mean_sin) * n
    rayleigh_z = R**2 / n
    if n < 50:
        terms = [
            (rayleigh_z**k) / math.factorial(k)
            for k in range(1, n + 1)
        ]
        rayleigh_p = float(np.exp(-rayleigh_z) * (1.0 + 2.0 * sum(terms)))
    else:
        rayleigh_p = float(np.exp(-rayleigh_z))
    rayleigh_p = float(np.clip(rayleigh_p, 0.0, 1.0))

    return {
        "mean_direction_deg": float(mean_dir_deg),
        "mean_doy": float(mean_doy),
        "drift_deg_per_year": drift_deg_per_year,
        "drift_deg_per_decade": drift_deg_per_decade,
        "drift_doy_per_year": drift_doy_per_year,
        "drift_doy_per_decade": drift_doy_per_decade,
        "cos_slope": float(cos_slope),
        "sin_slope": float(sin_slope),
        "rayleigh_p": rayleigh_p,
        "n": n,
    }


def _factorial(k: int) -> int:
    f = 1
    for i in range(2, k + 1):
        f *= i
    return f


# ---------------------------------------------------------------------------
# Bootstrap CIs for annual patch metrics
# ---------------------------------------------------------------------------
def bootstrap_patch_metrics(
    chl_ann_2d: np.ndarray,
    pixel_area_2d: np.ndarray,
    percentile: float = 85.0,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """
    Bootstrap 95% CIs for the per-year patch metrics derived from a single
    2-D chlorophyll map. Resamples pixels (with replacement) holding the
    pixel-area array fixed.

    Returns
    -------
    dict of metric -> {median, lo, hi, n_boot}
    """
    rng = np.random.default_rng(seed)
    chl = np.asarray(chl_ann_2d, dtype=float)
    area = np.asarray(pixel_area_2d, dtype=float)

    valid = ~np.isnan(chl)
    if not valid.any():
        nan = float("nan")
        return {
            "threshold": {"median": nan, "lo": nan, "hi": nan, "n_boot": 0},
            "area_km2": {"median": 0.0, "lo": 0.0, "hi": 0.0, "n_boot": 0},
            "mean_chl": {"median": nan, "lo": nan, "hi": nan, "n_boot": 0},
            "n_patches": {"median": 0, "lo": 0, "hi": 0, "n_boot": 0},
            "largest_patch": {"median": 0.0, "lo": 0.0, "hi": 0.0, "n_boot": 0},
        }

    chl_v = chl[valid]
    area_v = area[valid]
    n_pix = chl_v.size

    boot_thr = np.empty(n_boot, dtype=float)
    boot_area = np.empty(n_boot, dtype=float)
    boot_mean = np.empty(n_boot, dtype=float)
    boot_n = np.empty(n_boot, dtype=int)
    boot_largest = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n_pix, size=n_pix)
        c = chl_v[idx]
        a = area_v[idx]
        thr = float(np.percentile(c, percentile))
        mask = c >= thr
        if not mask.any():
            boot_thr[b] = thr
            boot_area[b] = 0.0
            boot_mean[b] = float("nan")
            boot_n[b] = 0
            boot_largest[b] = 0.0
            continue
        a_masked = a[mask]
        c_masked = c[mask]
        boot_thr[b] = thr
        boot_area[b] = float(a_masked.sum())
        boot_mean[b] = float(c_masked.mean())
        # Pixel-resampling breaks the spatial layout, so we report "1 patch"
        # as the conservative structural choice for the bootstrap estimate.
        boot_n[b] = 1
        boot_largest[b] = float(a_masked.sum())

    def _ci(x):
        x = np.asarray(x)
        if np.all(np.isnan(x)):
            nan = float("nan")
            return {"median": nan, "lo": nan, "hi": nan, "n_boot": int(x.size)}
        return {
            "median": float(np.nanmedian(x)),
            "lo": float(np.nanpercentile(x, 2.5)),
            "hi": float(np.nanpercentile(x, 97.5)),
            "n_boot": int(x.size),
        }

    return {
        "threshold": _ci(boot_thr),
        "area_km2": _ci(boot_area),
        "mean_chl": _ci(boot_mean),
        "n_patches": _ci(boot_n),
        "largest_patch": _ci(boot_largest),
    }


# ---------------------------------------------------------------------------
# p-value → significance stars
# ---------------------------------------------------------------------------
def p_value_to_stars(p: float) -> str:
    """Map a p-value to a star string (*** / ** / * / n.s.)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."
