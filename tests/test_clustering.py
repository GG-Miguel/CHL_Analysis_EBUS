import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.clustering import (
    extract_high_chl_pixels,
    cluster_pixels_hdbscan,
    cluster_pixels_kmeans,
    compute_cluster_centers,
    select_optimal_k,
    compute_seasonal_cycles,
    compute_correlation_distance,
    cluster_seasonal_hdbscan,
    cluster_seasonal_kmeans,
    select_optimal_k_seasonal,
)


def _make_two_blob_data(n_per_blob=500, seed=0):
    rng = np.random.default_rng(seed)
    lat1 = rng.normal(loc=26.0, scale=1.0, size=n_per_blob)
    lon1 = rng.normal(loc=-15.0, scale=1.0, size=n_per_blob)
    chl1 = rng.lognormal(mean=0.5, sigma=0.3, size=n_per_blob)

    lat2 = rng.normal(loc=21.0, scale=1.0, size=n_per_blob)
    lon2 = rng.normal(loc=-18.0, scale=1.0, size=n_per_blob)
    chl2 = rng.lognormal(mean=0.5, sigma=0.3, size=n_per_blob)

    lat = np.concatenate([lat1, lat2])
    lon = np.concatenate([lon1, lon2])
    chl = np.concatenate([chl1, chl2])
    return lat, lon, chl


def test_extract_high_chl_pixels_basic():
    rng = np.random.default_rng(0)
    chl_2d = rng.lognormal(0, 1, (50, 50))
    lat = np.linspace(10, 45, 50)
    lon = np.linspace(-30, -7, 50)
    lat_sel, lon_sel, chl_sel, threshold = extract_high_chl_pixels(chl_2d, lat, lon, 85)
    assert len(lat_sel) > 0
    assert len(lat_sel) == len(lon_sel) == len(chl_sel)
    assert threshold > 0
    assert np.all(chl_sel >= threshold)


def test_extract_high_chl_pixels_all_nan():
    chl_2d = np.full((10, 10), np.nan)
    lat = np.linspace(10, 20, 10)
    lon = np.linspace(-30, -20, 10)
    lat_sel, lon_sel, chl_sel, threshold = extract_high_chl_pixels(chl_2d, lat, lon, 85)
    assert len(lat_sel) == 0
    assert np.isnan(threshold)


def test_extract_high_chl_pixels_2d_lon():
    rng = np.random.default_rng(0)
    chl_2d = rng.lognormal(0, 1, (50, 50))
    lat = np.linspace(10, 45, 50)
    lon_1d = np.linspace(-30, -7, 50)
    lon_2d = lon_1d[np.newaxis, :] * np.ones((50, 1))
    lat_sel, lon_sel, chl_sel, threshold = extract_high_chl_pixels(chl_2d, lat, lon_2d, 85)
    assert len(lat_sel) > 0


def test_cluster_pixels_kmeans_two_blobs():
    lat, lon, chl = _make_two_blob_data()
    labels = cluster_pixels_kmeans(lat, lon, k=2, seed=0)
    assert len(labels) == len(lat)
    unique = np.unique(labels)
    assert len(unique) == 2


def test_cluster_pixels_kmeans_too_few_points():
    labels = cluster_pixels_kmeans(np.array([21.0]), np.array([-18.0]), k=2)
    assert np.all(labels == -1)


def test_cluster_pixels_hdbscan_two_blobs():
    lat, lon, chl = _make_two_blob_data(n_per_blob=500)
    labels, probs = cluster_pixels_hdbscan(lat, lon, min_cluster_size=50, min_samples=20)
    assert len(labels) == len(lat)
    assert len(probs) == len(lat)
    n_clusters = len(np.unique(labels[labels >= 0]))
    assert n_clusters >= 1


def test_cluster_pixels_hdbscan_too_few_points():
    lat = np.array([21.0, 22.0])
    lon = np.array([-18.0, -17.0])
    labels, probs = cluster_pixels_hdbscan(lat, lon, min_cluster_size=50)
    assert np.all(labels == -1)


def test_compute_cluster_centers_basic():
    lat, lon, chl = _make_two_blob_data()
    labels = cluster_pixels_kmeans(lat, lon, k=2, seed=0)
    centers = compute_cluster_centers(labels, lat, lon, chl)
    assert len(centers) == 2
    assert "center_lat" in centers.columns
    assert "center_lon" in centers.columns
    assert "mean_chl" in centers.columns
    assert "n_pixels" in centers.columns
    assert centers["n_pixels"].sum() == len(lat)


def test_compute_cluster_centers_all_noise():
    labels = np.full(100, -1, dtype=int)
    lat = np.random.default_rng(0).normal(25, 1, 100)
    lon = np.random.default_rng(0).normal(-15, 1, 100)
    chl = np.random.default_rng(0).lognormal(0, 1, 100)
    centers = compute_cluster_centers(labels, lat, lon, chl)
    assert len(centers) == 0


def test_compute_cluster_centers_sorted_by_latitude():
    lat, lon, chl = _make_two_blob_data()
    labels = cluster_pixels_kmeans(lat, lon, k=2, seed=0)
    centers = compute_cluster_centers(labels, lat, lon, chl)
    assert centers["center_lat"].iloc[0] >= centers["center_lat"].iloc[1]


def test_select_optimal_k():
    lat, lon, chl = _make_two_blob_data()
    k_df = select_optimal_k(lat, lon, k_range=range(2, 5), seed=0)
    assert len(k_df) == 3
    assert "k" in k_df.columns
    assert "silhouette" in k_df.columns
    assert "inertia" in k_df.columns
    assert k_df["silhouette"].max() > 0


def test_compute_correlation_distance_basic():
    rng = np.random.default_rng(0)
    cycles = rng.normal(size=(10, 12))
    dist = compute_correlation_distance(cycles)
    assert dist.shape == (10, 10)
    assert np.allclose(np.diag(dist), 0.0)
    assert np.all(dist >= 0) and np.all(dist <= 2)


def test_compute_correlation_distance_identical():
    cycle = np.sin(np.linspace(0, 2*np.pi, 12))
    cycles = np.tile(cycle, (5, 1))
    dist = compute_correlation_distance(cycles)
    assert np.allclose(dist, 0.0, atol=1e-10)


def test_cluster_seasonal_kmeans_basic():
    rng = np.random.default_rng(0)
    cycles = rng.normal(size=(50, 12))
    labels = cluster_seasonal_kmeans(cycles, k=2, seed=0)
    assert len(labels) == 50
    assert len(np.unique(labels)) == 2


def test_cluster_seasonal_kmeans_too_few():
    cycles = np.array([[1.0, 2.0, 3.0]])
    labels = cluster_seasonal_kmeans(cycles, k=2)
    assert np.all(labels == -1)


def test_cluster_seasonal_hdbscan_basic():
    rng = np.random.default_rng(0)
    cycles = rng.normal(size=(200, 12))
    labels, probs = cluster_seasonal_hdbscan(cycles, min_cluster_size=50, min_samples=10)
    assert len(labels) == 200
    assert len(probs) == 200
    assert labels.dtype == int


def test_cluster_seasonal_hdbscan_too_few():
    cycles = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    labels, probs = cluster_seasonal_hdbscan(cycles, min_cluster_size=50)
    assert np.all(labels == -1)


def test_select_optimal_k_seasonal_basic():
    rng = np.random.default_rng(0)
    cycles = rng.normal(size=(50, 12))
    k_df = select_optimal_k_seasonal(cycles, k_range=range(2, 5), seed=0)
    assert len(k_df) == 3
    assert "k" in k_df.columns
    assert "silhouette" in k_df.columns
    assert k_df["silhouette"].max() > 0


if __name__ == "__main__":
    tests = [
        test_extract_high_chl_pixels_basic,
        test_extract_high_chl_pixels_all_nan,
        test_extract_high_chl_pixels_2d_lon,
        test_cluster_pixels_kmeans_two_blobs,
        test_cluster_pixels_kmeans_too_few_points,
        test_cluster_pixels_hdbscan_two_blobs,
        test_cluster_pixels_hdbscan_too_few_points,
        test_compute_cluster_centers_basic,
        test_compute_cluster_centers_all_noise,
        test_compute_cluster_centers_sorted_by_latitude,
        test_select_optimal_k,
        test_compute_correlation_distance_basic,
        test_compute_correlation_distance_identical,
        test_cluster_seasonal_kmeans_basic,
        test_cluster_seasonal_kmeans_too_few,
        test_cluster_seasonal_hdbscan_basic,
        test_cluster_seasonal_hdbscan_too_few,
        test_select_optimal_k_seasonal_basic,
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
