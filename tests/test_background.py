"""
Unit tests for src/statistics/background.py

Run from the repo root:
    python -m pytest tests/test_background.py -v
or:
    python tests/test_background.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.statistics.background import (
    annual_percentile_metrics,
    quantile_regression_trends,
)


def _make_chl_da(n_years=10, n_time_per_year=46, n_row=20, n_col=30, seed=0):
    rng = np.random.default_rng(seed)
    times = []
    for y in range(2000, 2000 + n_years):
        for t in range(n_time_per_year):
            times.append(f"{y}-01-01")
    times = pd.date_range("2000-01-01", periods=n_years * n_time_per_year, freq="8D")
    data = rng.lognormal(mean=0.0, sigma=1.0, size=(len(times), n_row, n_col))
    da = xr.DataArray(
        data,
        dims=["time", "row", "col"],
        coords={"time": times},
    )
    return da


def test_annual_percentile_metrics_returns_correct_columns():
    da = _make_chl_da()
    df = annual_percentile_metrics(da, background_pcts=[10, 25], peak_pcts=[90, 95, 99])
    assert "year" in df.columns
    assert "bg_p10" in df.columns
    assert "bg_p25" in df.columns
    assert "peak_p90" in df.columns
    assert "peak_p95" in df.columns
    assert "peak_p99" in df.columns
    assert "peak_amplitude" in df.columns
    assert "spatial_mean" in df.columns
    assert len(df) >= 10


def test_annual_percentile_metrics_ordering():
    da = _make_chl_da()
    df = annual_percentile_metrics(da, background_pcts=[10, 25], peak_pcts=[90, 95, 99])
    for _, row in df.iterrows():
        assert row["bg_p10"] <= row["bg_p25"]
        assert row["bg_p25"] <= row["peak_p90"]
        assert row["peak_p90"] <= row["peak_p95"]
        assert row["peak_p95"] <= row["peak_p99"]
        assert row["peak_amplitude"] > 0


def test_annual_percentile_metrics_handles_all_nan():
    times = pd.date_range("2000-01-01", periods=46, freq="8D")
    data = np.full((46, 10, 10), np.nan)
    da = xr.DataArray(data, dims=["time", "row", "col"], coords={"time": times})
    df = annual_percentile_metrics(da)
    assert len(df) == 0


def test_quantile_regression_trends_returns_correct_shape():
    rng = np.random.default_rng(42)
    y = pd.Series(np.arange(20, dtype=float) * 0.5 + rng.normal(0, 0.5, 20))
    x = np.arange(2000, 2020, dtype=float)
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    df = quantile_regression_trends(y, x, quantiles=quantiles)
    assert len(df) == len(quantiles)
    assert "quantile" in df.columns
    assert "slope" in df.columns
    assert "slope_per_decade" in df.columns
    assert "p_value" in df.columns


def test_quantile_regression_positive_trend():
    y = pd.Series(np.arange(20, dtype=float) * 2.0)
    x = np.arange(2000, 2020, dtype=float)
    df = quantile_regression_trends(y, x, quantiles=[0.5])
    assert abs(df.iloc[0]["slope"] - 2.0) < 0.1
    assert abs(df.iloc[0]["slope_per_decade"] - 20.0) < 1.0


def test_quantile_regression_too_few_points():
    y = pd.Series([1.0, 2.0, 3.0])
    x = np.array([2000.0, 2001.0, 2002.0])
    df = quantile_regression_trends(y, x, quantiles=[0.5])
    assert len(df) == 0


def test_quantile_regression_handles_nans():
    y = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    x = np.arange(2000, 2010, dtype=float)
    df = quantile_regression_trends(y, x, quantiles=[0.5])
    assert len(df) == 1
    assert df.iloc[0]["n"] == 8


if __name__ == "__main__":
    tests = [
        test_annual_percentile_metrics_returns_correct_columns,
        test_annual_percentile_metrics_ordering,
        test_annual_percentile_metrics_handles_all_nan,
        test_quantile_regression_trends_returns_correct_shape,
        test_quantile_regression_positive_trend,
        test_quantile_regression_too_few_points,
        test_quantile_regression_handles_nans,
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
