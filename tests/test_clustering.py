import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.clustering import (
    compute_correlation_distance,
    cluster_seasonal_hdbscan,
    cluster_seasonal_kmeans,
    select_optimal_k_seasonal,
    cluster_seasonal_hdbscan_pca,
    search_hdbscan_parameters,
)


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


def test_cluster_seasonal_hdbscan_pca_basic():
    rng = np.random.default_rng(0)
    cycles = rng.normal(size=(200, 12))
    labels, probs, clusterer = cluster_seasonal_hdbscan_pca(
        cycles, min_cluster_size=50, min_samples=10, n_components=3,
    )
    assert len(labels) == 200
    assert len(probs) == 200
    assert labels.dtype == int
    assert clusterer is not None


def test_cluster_seasonal_hdbscan_pca_too_few():
    cycles = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    labels, probs, clusterer = cluster_seasonal_hdbscan_pca(
        cycles, min_cluster_size=50,
    )
    assert np.all(labels == -1)
    assert clusterer is None


def test_search_hdbscan_parameters_basic():
    rng = np.random.default_rng(0)
    cycles = rng.normal(size=(200, 12))
    results = search_hdbscan_parameters(
        cycles,
        min_cluster_sizes=[20, 50],
        min_samples_list=[5, 10],
        pca_components=[3],
        selection_methods=["eom"],
    )
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert "min_cluster_size" in results.columns
    assert "min_samples" in results.columns
    assert "n_pca_components" in results.columns
    assert "dbcv" in results.columns
    assert "n_clusters" in results.columns
    assert "noise_ratio" in results.columns


if __name__ == "__main__":
    tests = [
        test_compute_correlation_distance_basic,
        test_compute_correlation_distance_identical,
        test_cluster_seasonal_kmeans_basic,
        test_cluster_seasonal_kmeans_too_few,
        test_cluster_seasonal_hdbscan_basic,
        test_cluster_seasonal_hdbscan_too_few,
        test_select_optimal_k_seasonal_basic,
        test_cluster_seasonal_hdbscan_pca_basic,
        test_cluster_seasonal_hdbscan_pca_too_few,
        test_search_hdbscan_parameters_basic,
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
