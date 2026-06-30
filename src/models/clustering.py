import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import hdbscan
from hdbscan.validity import validity_index as _dbcv_validity_index


def compute_seasonal_cycles(
    chl_da,
    lat: np.ndarray,
    lon: np.ndarray,
    percentile: float = 97,
    temporal_resolution: str = "8day",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute climatological seasonal cycle vectors for each pixel.

    Parameters
    ----------
    chl_da : xr.DataArray
        Chlorophyll time series
    lat, lon : np.ndarray
        Coordinate arrays
    percentile : float
        Only include pixels above this percentile in the mean climatology
    temporal_resolution : str
        "8day" (46 periods/year) or "monthly" (12 periods/year)

    Returns
    -------
    cycles : np.ndarray
        Shape (n_valid_pixels, n_periods) - seasonal cycle for each pixel
    lat_sel, lon_sel : np.ndarray
        Coordinates of valid pixels
    periods : np.ndarray
        Period indices (1-46 for 8day, 1-12 for monthly)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        chl_clim = np.nanmean(chl_da.values, axis=0)

    valid = ~np.isnan(chl_clim)
    if not valid.any():
        return np.array([]), np.array([]), np.array([]), np.array([])

    threshold = float(np.nanpercentile(chl_clim[valid], percentile))
    mask = (chl_clim >= threshold) & valid

    if lat.ndim == 1 and lon.ndim == 2:
        lat_2d = lat[:, np.newaxis] * np.ones_like(lon)
        lon_2d = lon
    elif lat.ndim == 1 and lon.ndim == 1:
        lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
    else:
        lat_2d, lon_2d = lat, lon

    ys, xs = np.where(mask)
    lat_sel = lat_2d[ys, xs]
    lon_sel = lon_2d[ys, xs]

    if temporal_resolution == "monthly":
        periods = np.arange(1, 13)
        groupby = "time.month"
    else:
        periods = np.arange(1, 47)
        groupby = "time.dayofyear"

    cycles_list = []
    for period in periods:
        if temporal_resolution == "monthly":
            period_data = chl_da.sel(time=chl_da.time.dt.month == period)
        else:
            doy_start = (period - 1) * 8 + 1
            doy_end = period * 8
            period_data = chl_da.sel(
                time=(chl_da.time.dt.dayofyear >= doy_start) &
                     (chl_da.time.dt.dayofyear <= doy_end)
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            period_mean = np.nanmean(period_data.values, axis=0)

        period_values = period_mean[ys, xs]
        cycles_list.append(period_values)

    cycles = np.column_stack(cycles_list)
    return cycles, lat_sel, lon_sel, periods


def compute_correlation_distance(cycles: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix using 1 - Pearson correlation.

    Parameters
    ----------
    cycles : np.ndarray
        Shape (n_pixels, n_periods)

    Returns
    -------
    dist_matrix : np.ndarray
        Shape (n_pixels, n_pixels) - distance matrix
    """
    from scipy.spatial.distance import pdist, squareform

    cycles_clean = cycles.copy()
    nan_mask = np.isnan(cycles_clean)
    row_means = np.nanmean(cycles_clean, axis=1, keepdims=True)
    cycles_clean[nan_mask] = 0

    centered = cycles_clean - row_means
    norms = np.sqrt(np.sum(centered**2, axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    normalized = centered / norms

    corr_matrix = normalized @ normalized.T
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    dist_matrix = 1.0 - corr_matrix
    np.fill_diagonal(dist_matrix, 0.0)

    return dist_matrix.astype(np.float64)


def _zscore_cycles(cycles: np.ndarray) -> np.ndarray:
    """Z-score normalize each pixel's seasonal cycle (row-wise)."""
    out = cycles.copy()
    nan_mask = np.isnan(out)
    row_mean = np.nanmean(out, axis=1, keepdims=True)
    row_std = np.nanstd(out, axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    out = (out - row_mean) / row_std
    out[nan_mask] = 0
    return out.astype(np.float64)


def cluster_seasonal_hdbscan(
    cycles: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster pixels by seasonal cycle similarity using HDBSCAN.
    Z-scores cycles internally and uses Euclidean distance
    (equivalent to correlation distance on normalized data).

    Parameters
    ----------
    cycles : np.ndarray
        Shape (n_pixels, n_periods) - raw seasonal cycle vectors
    min_cluster_size, min_samples : int
        HDBSCAN parameters

    Returns
    -------
    labels : np.ndarray
        Cluster labels (-1 for noise)
    probabilities : np.ndarray
        Membership probabilities
    """
    n = cycles.shape[0]
    if n < min_cluster_size:
        return np.full(n, -1, dtype=int), np.zeros(n, dtype=float)

    cycles_z = _zscore_cycles(cycles)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(cycles_z)
    probabilities = clusterer.probabilities_
    return labels, probabilities


def cluster_seasonal_kmeans(
    cycles: np.ndarray,
    k: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """
    Cluster pixels by seasonal cycle using K-means on normalized cycles.

    Parameters
    ----------
    cycles : np.ndarray
        Shape (n_pixels, n_periods)
    k : int
        Number of clusters
    seed : int
        Random seed

    Returns
    -------
    labels : np.ndarray
        Cluster labels
    """
    n = cycles.shape[0]
    if n < k:
        return np.full(n, -1, dtype=int)

    cycles_norm = cycles.copy()
    for i in range(n):
        valid = ~np.isnan(cycles_norm[i])
        if valid.any():
            mean_val = np.nanmean(cycles_norm[i, valid])
            std_val = np.nanstd(cycles_norm[i, valid])
            if std_val > 0:
                cycles_norm[i, valid] = (cycles_norm[i, valid] - mean_val) / std_val

    nan_mask = np.isnan(cycles_norm)
    cycles_norm[nan_mask] = 0

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(cycles_norm)
    return labels


def select_optimal_k_seasonal(
    cycles: np.ndarray,
    k_range: range = range(2, 6),
    seed: int = 0,
) -> pd.DataFrame:
    """
    Select optimal k for seasonal cycle clustering using silhouette score.
    """
    cycles_norm = cycles.copy()
    n = cycles.shape[0]
    for i in range(n):
        valid = ~np.isnan(cycles_norm[i])
        if valid.any():
            mean_val = np.nanmean(cycles_norm[i, valid])
            std_val = np.nanstd(cycles_norm[i, valid])
            if std_val > 0:
                cycles_norm[i, valid] = (cycles_norm[i, valid] - mean_val) / std_val

    nan_mask = np.isnan(cycles_norm)
    cycles_norm[nan_mask] = 0

    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(cycles_norm)
        sil = silhouette_score(cycles_norm, labels)
        records.append({"k": k, "silhouette": sil, "inertia": km.inertia_})

    return pd.DataFrame(records)


def cluster_seasonal_hdbscan_pca(
    cycles: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    n_components: int | None = None,
    cluster_selection_method: str = "eom",
) -> tuple[np.ndarray, np.ndarray, "hdbscan.HDBSCAN"]:
    """
    Cluster pixels by seasonal cycle similarity using HDBSCAN on PCA-reduced
    z-scored cycles.

    Parameters
    ----------
    cycles : np.ndarray
        Shape (n_pixels, n_periods) - raw seasonal cycle vectors
    min_cluster_size, min_samples : int
        HDBSCAN parameters
    n_components : int or None
        Number of PCA components. If None, keeps enough to explain >=90%
        variance.
    cluster_selection_method : str
        "eom" or "leaf"

    Returns
    -------
    labels : np.ndarray
        Cluster labels (-1 for noise)
    probabilities : np.ndarray
        Membership probabilities
    clusterer : hdbscan.HDBSCAN
        Fitted HDBSCAN object
    """
    n = cycles.shape[0]
    if n < min_cluster_size:
        return np.full(n, -1, dtype=int), np.zeros(n, dtype=float), None

    cycles_z = _zscore_cycles(cycles)

    if n_components is not None:
        n_comp = min(n_components, n - 1, cycles_z.shape[1])
    else:
        pca_full = PCA()
        pca_full.fit(cycles_z)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cumvar, 0.90) + 1)
        n_comp = min(n_comp, n - 1, cycles_z.shape[1])

    pca = PCA(n_components=n_comp)
    cycles_pca = pca.fit_transform(cycles_z)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(cycles_pca)
    probabilities = clusterer.probabilities_
    return labels, probabilities, clusterer


def search_hdbscan_parameters(
    cycles: np.ndarray,
    min_cluster_sizes: list[int] | None = None,
    min_samples_list: list[int] | None = None,
    pca_components: list[int | None] | None = None,
    selection_methods: list[str] | None = None,
) -> pd.DataFrame:
    """
    Grid search over HDBSCAN parameters for seasonal cycle clustering.

    Evaluates each combination using DBCV (Density-Based Clustering
    Validation), noise ratio, number of clusters, and mean cluster
    persistence.

    Parameters
    ----------
    cycles : np.ndarray
        Shape (n_pixels, n_periods) - raw seasonal cycle vectors
    min_cluster_sizes : list of int
    min_samples_list : list of int
    pca_components : list of (int or None)
        None means auto-select for >=90% variance explained
    selection_methods : list of str

    Returns
    -------
    results : pd.DataFrame
        Sorted by DBCV score (descending)
    """
    if min_cluster_sizes is None:
        min_cluster_sizes = [20, 50, 100, 150, 200, 300, 500]
    if min_samples_list is None:
        min_samples_list = [5, 10, 20, 50, 100]
    if pca_components is None:
        pca_components = [None, 10, 5, 3]
    if selection_methods is None:
        selection_methods = ["eom", "leaf"]

    cycles_z = _zscore_cycles(cycles)
    n = cycles_z.shape[0]

    records = []
    for mcs in min_cluster_sizes:
        for ms in min_samples_list:
            for n_comp in pca_components:
                for csm in selection_methods:
                    if n < mcs:
                        continue

                    if n_comp is not None:
                        nc = min(n_comp, n - 1, cycles_z.shape[1])
                    else:
                        pca_full = PCA()
                        pca_full.fit(cycles_z)
                        cumvar = np.cumsum(
                            pca_full.explained_variance_ratio_
                        )
                        nc = int(np.searchsorted(cumvar, 0.90) + 1)
                        nc = min(nc, n - 1, cycles_z.shape[1])

                    pca = PCA(n_components=nc)
                    X = pca.fit_transform(cycles_z)

                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=mcs,
                        min_samples=ms,
                        metric="euclidean",
                        cluster_selection_method=csm,
                    )
                    labels = clusterer.fit_predict(X)
                    probs = clusterer.probabilities_

                    unique_labels = np.unique(labels)
                    unique_labels = unique_labels[unique_labels >= 0]
                    n_clusters = len(unique_labels)
                    noise_ratio = float(np.sum(labels < 0) / n)

                    dbcv = np.nan
                    if n_clusters >= 2:
                        try:
                            dbcv = float(
                                _dbcv_validity_index(X, labels)
                            )
                        except Exception:
                            dbcv = np.nan

                    mean_persistence = np.nan
                    if (
                        hasattr(clusterer, "cluster_persistence_")
                        and clusterer.cluster_persistence_ is not None
                        and len(clusterer.cluster_persistence_) > 0
                    ):
                        mean_persistence = float(
                            np.nanmean(clusterer.cluster_persistence_)
                        )

                    records.append({
                        "min_cluster_size": mcs,
                        "min_samples": ms,
                        "n_pca_components": nc,
                        "pca_requested": n_comp,
                        "cluster_selection_method": csm,
                        "n_clusters": n_clusters,
                        "noise_ratio": noise_ratio,
                        "dbcv": dbcv,
                        "mean_persistence": mean_persistence,
                        "n_pixels": n,
                    })

    results = pd.DataFrame(records)
    if len(results) > 0:
        results = results.sort_values(
            "dbcv", ascending=False
        ).reset_index(drop=True)
    return results
