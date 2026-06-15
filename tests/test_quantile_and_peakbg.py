"""
Unit tests for src/statistics/quantile_trends.py,
src/statistics/peak_background.py, and
src/statistics/seasonal_decomposition.py

Run from the repo root:
    python -m pytest tests/test_quantile_and_peakbg.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.statistics.quantile_trends import (
    quantile_trend,
    quantile_trend_log10,
    peak_background_divergence,
    DEFAULT_QUANTILES,
)
from src.statistics.peak_background import running_mean_trends
from src.statistics.seasonal_decomposition import stl_decomposition, seasonal_amplitude


class TestQuantileTrend:
    def test_increasing_series(self):
        np.random.seed(42)
        x = np.arange(2003, 2024, dtype=float)
        y = 0.01 * (x - 2003) + np.random.normal(0, 0.05, len(x))
        y = pd.Series(y)
        df = quantile_trend(y, x=x)
        assert len(df) == len(DEFAULT_QUANTILES)
        assert all(df["quantile"].values > 0)
        slope_q50 = df.loc[df["quantile"] == 0.5, "slope"].values[0]
        assert slope_q50 > 0

    def test_per_decade_scaling(self):
        np.random.seed(0)
        x = np.arange(2003, 2024, dtype=float)
        y = pd.Series(np.random.normal(0, 1, len(x)))
        df = quantile_trend(y, x=x)
        for _, row in df.iterrows():
            if not np.isnan(row["slope"]):
                assert abs(row["slope_per_decade"] - row["slope"] * 10) < 1e-10

    def test_custom_quantiles(self):
        y = pd.Series(np.random.normal(0, 1, 50))
        df = quantile_trend(y, quantiles=[0.5, 0.9])
        assert len(df) == 2
        assert list(df["quantile"]) == [0.5, 0.9]

    def test_too_few_points(self):
        y = pd.Series([1.0, 2.0])
        df = quantile_trend(y, x=np.array([1.0, 2.0]))
        assert len(df) == len(DEFAULT_QUANTILES)
        assert all(pd.isna(df["slope"]))


class TestQuantileTrendLog10:
    def test_log_transform_applied(self):
        np.random.seed(1)
        chl = pd.Series(np.abs(np.random.normal(1.0, 0.3, 30)))
        df = quantile_trend_log10(chl)
        assert "pct_per_year" in df.columns
        assert "pct_per_decade" in df.columns
        assert not all(pd.isna(df["pct_per_year"]))

    def test_pct_conversion_formula(self):
        np.random.seed(1)
        chl = pd.Series(np.abs(np.random.normal(1.0, 0.3, 30)))
        df = quantile_trend_log10(chl)
        for _, row in df.iterrows():
            if not np.isnan(row["slope"]):
                expected_pct = (10 ** row["slope"] - 1) * 100
                assert abs(row["pct_per_year"] - expected_pct) < 1e-8


class TestPeakBackgroundDivergence:
    def test_divergence_positive(self):
        df = pd.DataFrame({
            "quantile": [0.25, 0.90],
            "slope": [0.001, 0.005],
            "pct_per_year": [0.23, 1.15],
            "pct_per_decade": [2.3, 11.5],
        })
        div = peak_background_divergence(df)
        assert div["divergence"] == pytest.approx(0.004)
        assert div["amplification_ratio"] == pytest.approx(1.15 / 0.23, rel=1e-6)

    def test_missing_quantiles(self):
        df = pd.DataFrame({
            "quantile": [0.5],
            "slope": [0.0],
            "pct_per_year": [0.0],
            "pct_per_decade": [0.0],
        })
        div = peak_background_divergence(df)
        assert np.isnan(div["background_slope"])


class TestRunningMeanTrends:
    def test_running_mean_trends_basic(self):
        np.random.seed(42)
        years = np.arange(2003, 2024)
        df = pd.DataFrame({
            "year": years,
            "bg_total": 0.3 + 0.002 * (years - 2003) + np.random.normal(0, 0.01, len(years)),
            "peak_anomaly_mean": 0.15 + 0.001 * (years - 2003) + np.random.normal(0, 0.005, len(years)),
            "peak_excess": 0.15 + 0.001 * (years - 2003) + np.random.normal(0, 0.005, len(years)),
            "peak_fraction": 0.5 + np.random.normal(0, 0.02, len(years)),
            "obs_mean": 0.45 + np.random.normal(0, 0.01, len(years)),
            "obs_max": 1.0 + np.random.normal(0, 0.05, len(years)),
        })
        trends = running_mean_trends(df)
        assert len(trends) >= 4
        assert "slope" in trends.columns
        assert "stars" in trends.columns

    def test_running_mean_trends_too_few(self):
        df = pd.DataFrame({
            "year": [2020, 2021],
            "bg_total": [0.3, 0.31],
            "peak_anomaly_mean": [0.15, 0.16],
        })
        trends = running_mean_trends(df)
        assert len(trends) == 0


class TestSTLDecomposition:
    def test_stl_basic(self):
        np.random.seed(42)
        n = 120
        t = np.arange(n, dtype=float)
        seasonal = 0.3 * np.sin(2 * np.pi * t / 12)
        trend = 0.001 * t
        noise = np.random.normal(0, 0.05, n)
        y = pd.Series(trend + seasonal + noise, index=pd.date_range("2003-01", periods=n, freq="ME"))
        decomp = stl_decomposition(y, period=12)
        assert "trend" in decomp.columns
        assert "seasonal" in decomp.columns
        assert "residual" in decomp.columns
        assert len(decomp) == n

    def test_seasonal_amplitude(self):
        np.random.seed(42)
        n = 120
        t = np.arange(n, dtype=float)
        seasonal = 0.3 * np.sin(2 * np.pi * t / 12)
        trend = 0.001 * t + 0.5
        noise = np.random.normal(0, 0.02, n)
        y = pd.Series(trend + seasonal + noise, index=pd.date_range("2003-01", periods=n, freq="ME"))
        amp = seasonal_amplitude(y, period=12)
        assert "amplitude" in amp.columns
        assert len(amp) >= 5