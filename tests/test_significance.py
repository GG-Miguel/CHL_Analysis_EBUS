"""
Unit tests for src/statistics/significance.py

Run from the repo root:
    python -m pytest tests/test_significance.py -v
or, if pytest is not installed:
    python tests/test_significance.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.statistics.significance import (
    bootstrap_patch_metrics,
    circular_doy_trend,
    mann_kendall,
    p_value_to_stars,
    theil_sen_with_ci,
)


# ---------------------------------------------------------------------------
# Theil–Sen
# ---------------------------------------------------------------------------
def test_theil_sen_monotonic_series():
    """Slope on a strictly increasing series should be ~1, p small."""
    y = pd.Series(np.arange(20, dtype=float))
    res = theil_sen_with_ci(y)
    assert abs(res["slope"] - 1.0) < 1e-6, f"expected slope=1, got {res['slope']}"
    assert res["n"] == 20
    assert res["slope_lo"] <= res["slope"] <= res["slope_hi"]


def test_theil_sen_handles_nans():
    """NaN entries should be dropped, not crash."""
    y = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0])
    res = theil_sen_with_ci(y)
    assert res["n"] == 5
    assert res["slope"] == 1.0


def test_theil_sen_too_few_points():
    """n < 3 should return NaNs, not crash."""
    y = pd.Series([1.0, 2.0])
    res = theil_sen_with_ci(y)
    assert np.isnan(res["slope"])


def test_theil_sen_per_decade_is_10x_per_year():
    """The per-decade slope must be exactly 10× the per-year slope."""
    y = pd.Series(np.arange(20, dtype=float) + 5)
    res = theil_sen_with_ci(y)
    assert abs(res["slope_per_decade"] - 10.0 * res["slope"]) < 1e-9
    assert abs(res["slope_lo_per_decade"] - 10.0 * res["slope_lo"]) < 1e-9
    assert abs(res["slope_hi_per_decade"] - 10.0 * res["slope_hi"]) < 1e-9


def test_theil_sen_with_year_x_gives_per_year_slope():
    """When x is the actual year, slope should equal per-year rate."""
    y = pd.Series(np.arange(20, dtype=float) * 2.5)
    years = pd.Series(np.arange(2000, 2020, dtype=float))
    res = theil_sen_with_ci(y, x=years)
    # y increases by 2.5 per year-step; with year x, slope is units/yr = 2.5
    assert abs(res["slope"] - 2.5) < 1e-6
    assert abs(res["slope_per_decade"] - 25.0) < 1e-6


def test_theil_sen_intercept_at_x0():
    """intercept_at_x0 should be the y-value at the first x."""
    y = pd.Series(np.arange(20, dtype=float))
    years = pd.Series(np.arange(2000, 2020, dtype=float))
    res = theil_sen_with_ci(y, x=years)
    # y[0] = 0, slope = 1, so at x0=2000 the line gives 0
    assert abs(res["intercept_at_x0"] - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# Mann–Kendall
# ---------------------------------------------------------------------------
def test_mann_kendall_increasing():
    """Strictly increasing series: p < 0.05, tau = 1, trend = 'increasing'."""
    y = pd.Series(np.arange(15, dtype=float))
    res = mann_kendall(y)
    assert res["p_value"] < 0.05, f"expected p<0.05, got {res['p_value']}"
    assert res["h"] is True
    assert res["trend"] == "increasing"
    assert res["tau"] == 1.0


def test_mann_kendall_random():
    """White noise should not be significant."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=30))
    res = mann_kendall(y)
    assert res["p_value"] > 0.01, f"expected p>0.01 for noise, got {res['p_value']}"


def test_mann_kendall_too_few_points():
    y = pd.Series([1.0, 2.0, 3.0])
    res = mann_kendall(y)
    assert res["n"] == 3
    assert np.isnan(res["p_value"])


def test_mann_kendall_per_decade_with_year_x():
    """MK slope_per_decade with year x should be 10× the per-year slope."""
    y = pd.Series(np.arange(15, dtype=float))
    years = pd.Series(np.arange(2000, 2015, dtype=float))
    res = mann_kendall(y, x=years)
    assert abs(res["slope"] - 1.0) < 1e-6
    assert abs(res["slope_per_decade"] - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# Circular DOY
# ---------------------------------------------------------------------------
def test_circular_doy_no_drift_for_wrap():
    """A series that wraps around the new year should have small drift."""
    doy = pd.Series([350, 355, 360, 5, 10, 15, 20])
    res = circular_doy_trend(doy)
    # Drift in degrees/year should be small (the points all lie near
    # the same physical position on the circle, just labelled in two ways).
    assert res["drift_deg_per_year"] < 30, (
        f"expected small drift for wrapping series, got {res['drift_deg_per_year']}"
    )


def test_circular_doy_monotonic_shift():
    """A steadily shifting series should have non-zero drift and a
    significant Rayleigh test."""
    doy = pd.Series([30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
    res = circular_doy_trend(doy)
    assert res["drift_deg_per_year"] > 0
    assert res["drift_deg_per_decade"] == res["drift_deg_per_year"] * 10


def test_circular_doy_with_year_x_doesnt_change_slope_magnitude():
    """The drift in degrees per year should be invariant under
    x-translation when x is in years with unit step."""
    doy = pd.Series([30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
    res_default = circular_doy_trend(doy)
    res_with_x = circular_doy_trend(doy, x=np.arange(2010, 2020, dtype=float))
    assert abs(res_default["drift_deg_per_year"] - res_with_x["drift_deg_per_year"]) < 1e-9


# ---------------------------------------------------------------------------
# Bootstrap patch metrics
# ---------------------------------------------------------------------------
def test_bootstrap_returns_sensible_cis():
    """A simple synthetic 2-D map should produce finite CIs."""
    rng = np.random.default_rng(0)
    chl = rng.lognormal(mean=0.0, sigma=1.0, size=(50, 50))
    area = np.ones_like(chl) * 1.0
    res = bootstrap_patch_metrics(chl, area, percentile=85.0, n_boot=50, seed=0)
    assert "area_km2" in res
    assert res["area_km2"]["n_boot"] == 50
    a = res["area_km2"]
    assert a["lo"] <= a["median"] <= a["hi"]


def test_bootstrap_handles_all_nan():
    """A NaN-filled array should not crash and should return NaNs/0s."""
    chl = np.full((10, 10), np.nan)
    area = np.ones_like(chl)
    res = bootstrap_patch_metrics(chl, area, n_boot=5, seed=0)
    assert res["area_km2"]["n_boot"] == 0


# ---------------------------------------------------------------------------
# Stars
# ---------------------------------------------------------------------------
def test_p_value_to_stars():
    assert p_value_to_stars(0.0001) == "***"
    assert p_value_to_stars(0.005) == "**"
    assert p_value_to_stars(0.02) == "*"
    assert p_value_to_stars(0.2) == "n.s."
    assert p_value_to_stars(np.nan) == "n.s."
    assert p_value_to_stars(None) == "n.s."


# ---------------------------------------------------------------------------
# Manual runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_theil_sen_monotonic_series,
        test_theil_sen_handles_nans,
        test_theil_sen_too_few_points,
        test_theil_sen_per_decade_is_10x_per_year,
        test_theil_sen_with_year_x_gives_per_year_slope,
        test_theil_sen_intercept_at_x0,
        test_mann_kendall_increasing,
        test_mann_kendall_random,
        test_mann_kendall_too_few_points,
        test_mann_kendall_per_decade_with_year_x,
        test_circular_doy_no_drift_for_wrap,
        test_circular_doy_monotonic_shift,
        test_circular_doy_with_year_x_doesnt_change_slope_magnitude,
        test_bootstrap_returns_sensible_cis,
        test_bootstrap_handles_all_nan,
        test_p_value_to_stars,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
