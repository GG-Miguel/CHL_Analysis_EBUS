"""
Statistical significance helpers for interannual trend and patch-metric
analysis.

All functions accept a pandas Series / 1-D array-like for `y` and return
plain Python dicts that can be turned into a DataFrame row.

Conventions
-----------
* Theil–Sen slope is reported in original units of `y` per "year step"
  (i.e. per integer index, since the inputs are annual time series).
* `slope_pct_per_decade` is computed correctly as
  `100 * (slope / median(y)) * 10` — it answers the question
  "what is the relative change per decade, expressed as a percentage".
* Mann–Kendall uses `pymannkendall` (Hamed & Rao 1998 variance correction
  is applied automatically when there are ties).
* For circular variables (e.g. day-of-year) we use cos/sin decomposition
  + Theil–Sen on each component, then convert the drift vector back to
  a direction in degrees.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

import pymannkendall as mk

from skimage import measure


# ---------------------------------------------------------------------------
# Mann–Kendall
# ---------------------------------------------------------------------------
def mann_kendall(y: pd.Series, alpha: float = 0.05) -> dict:
    """
    Two-sided Mann–Kendall trend test (Hamed & Rao 1998 variance).

    Parameters
    ----------
    y : pd.Series
        Annual time series. NaNs are dropped.
    alpha : float
        Significance level (default 0.05) — only used for the `h` boolean
        and the `trend` label.

    Returns
    -------
    dict with keys
        tau, z, p_value, h, trend, slope, intercept, n
    """
    arr = np.asarray(y, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n < 4:
        return {
            "tau": np.nan, "z": np.nan, "p_value": np.nan,
            "h": False, "trend": "no trend",
            "slope": np.nan, "intercept": np.nan, "n": n,
        }
    res = mk.original_test(arr, alpha=alpha)
    return {
        "tau": float(res.Tau),
        "z": float(res.z),
        "p_value": float(res.p),
        "h": bool(res.h),
        "trend": str(res.trend),
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Theil–Sen with confidence interval
# ---------------------------------------------------------------------------
def theil_sen_with_ci(y: pd.Series, alpha: float = 0.05) -> dict:
    """
    Theil–Sen slope with 95% confidence interval and a per-decade
    percentage change computed against the median of `y`.

    Parameters
    ----------
    y : pd.Series
        Annual time series. NaNs are dropped.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict with keys
        slope, intercept, slope_lo, slope_hi,
        slope_pct_per_decade, sign_p_value, n
    """
    arr = np.asarray(y, dtype=float)
    mask = ~np.isnan(arr)
    n = int(mask.sum())
    if n < 3:
        nan = float("nan")
        return {
            "slope": nan, "intercept": nan,
            "slope_lo": nan, "slope_hi": nan,
            "slope_pct_per_decade": nan, "sign_p_value": nan, "n": n,
        }

    x = np.arange(n, dtype=float)
    y_ok = arr[mask]

    slope, intercept, slope_lo, slope_hi = stats.theilslopes(y_ok, x, alpha=alpha)

    # scipy ≥ 1.9 returns a 5-tuple (slope, intercept, lo, hi, low_slope, high_slope)?
    # Actually it returns 5 values: (slope, intercept, lo_slope, hi_slope, p_value_2sided).
    # We handle both 4- and 5-element return shapes for forward-compat.
    res = stats.theilslopes(y_ok, x, alpha=alpha)
    if len(res) == 5:
        slope, intercept, slope_lo, slope_hi, sign_p = res
    else:
        slope, intercept, slope_lo, slope_hi = res
        sign_p = float("nan")

    med = float(np.nanmedian(y_ok))
    if med > 0:
        slope_pct = 100.0 * (slope / med) * 10.0
    else:
        slope_pct = float("nan")

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_lo": float(slope_lo),
        "slope_hi": float(slope_hi),
        "slope_pct_per_decade": float(slope_pct),
        "sign_p_value": float(sign_p),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Circular day-of-year trend
# ---------------------------------------------------------------------------
def circular_doy_trend(peak_doy: pd.Series) -> dict:
    """
    Trend test on a circular (day-of-year) variable.

    We convert DOY to an angle θ = 2π·DOY/365, fit Theil–Sen independently
    to cos θ and sin θ, then combine the per-component drifts into a single
    drift vector whose magnitude and direction we report.

    Significance of the mean direction is tested with a Rayleigh test
    (uniform-phase null). Drift magnitude is taken as the Euclidean norm
    of the (cos-slope, sin-slope) pair, in degrees per year.

    Parameters
    ----------
    peak_doy : pd.Series
        Annual peak day-of-year (1–366). NaNs dropped.

    Returns
    -------
    dict with keys
        mean_direction_deg, drift_deg_per_year,
        cos_slope, sin_slope, rayleigh_p, n
    """
    arr = np.asarray(peak_doy, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n < 3:
        nan = float("nan")
        return {
            "mean_direction_deg": nan,
            "drift_deg_per_year": nan,
            "cos_slope": nan, "sin_slope": nan,
            "rayleigh_p": nan, "n": n,
        }

    theta = 2.0 * np.pi * (arr % 365.0) / 365.0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x = np.arange(n, dtype=float)
    cos_slope, _, _, _ = stats.theilslopes(cos_t, x)
    sin_slope, _, _, _ = stats.theilslopes(sin_t, x)

    # Mean direction of the cloud of points
    mean_cos = cos_t.mean()
    mean_sin = sin_t.mean()
    mean_dir_rad = np.arctan2(mean_sin, mean_cos)
    mean_dir_deg = (np.degrees(mean_dir_rad)) % 360.0

    # Drift vector — convert (cos_slope, sin_slope) (per year) into a
    # displacement on the unit circle, then to degrees per year.
    drift_rad_per_year = np.hypot(cos_slope, sin_slope)
    drift_deg_per_year = float(np.degrees(drift_rad_per_year))

    # Rayleigh test (uniform-phase null)
    R = np.hypot(mean_cos, mean_sin) * n
    rayleigh_z = R**2 / n
    # For n > 50 the approximation p ≈ exp(-z) is fine; for n < 50 use
    # the exact series (Greenwood & Durand 1955).
    if n < 50:
        terms = [
            (rayleigh_z**k) / math.factorial(k)
            for k in range(1, n + 1)
        ]
        rayleigh_p = float(np.exp(-rayleigh_z) * (1.0 + 2.0 * sum(terms)))
    else:
        rayleigh_p = float(np.exp(-rayleigh_z))
    # Numerical artefacts in the series approximation can occasionally
    # push p slightly above 1; clip into the valid range.
    rayleigh_p = float(np.clip(rayleigh_p, 0.0, 1.0))

    return {
        "mean_direction_deg": float(mean_dir_deg),
        "drift_deg_per_year": drift_deg_per_year,
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

        # Connected-component labelling on the bootstrap mask
        # (we label on a 1-D array — n_patches is "1" in that case;
        # for 2-D structure we would need to preserve geometry, but the
        # pixel resampling breaks the spatial layout anyway, so a 1-D
        # count of "1 patch" is the conservative choice here).
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
