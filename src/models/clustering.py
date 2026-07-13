import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import hdbscan

from src.utils.config import R_EARTH_KM


def extract_high_chl_pixels(
    chl_2d: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    percentile: float = 85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    valid = ~np.isnan(chl_2d)
    if not valid.any():
        return np.array([]), np.array([]), np.array([]), np.nan

    threshold = float(np.nanpercentile(chl_2d[valid], percentile))
    mask = (chl_2d >= threshold) & valid

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
    chl_sel = chl_2d[ys, xs]

    return lat_sel, lon_sel, chl_sel, threshold


def cluster_pixels_hdbscan(
    lat: np.ndarray,
    lon: np.ndarray,
    min_cluster_size: int = 300,
    min_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    if len(lat) < min_cluster_size:
        return np.full(len(lat), -1, dtype=int), np.zeros(len(lat), dtype=float)

    coords = np.column_stack([lat, lon])
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(coords)
    probabilities = clusterer.probabilities_
    return labels, probabilities


def cluster_pixels_kmeans(
    lat: np.ndarray,
    lon: np.ndarray,
    k: int = 2,
    seed: int = 0,
) -> np.ndarray:
    if len(lat) < k:
        return np.full(len(lat), -1, dtype=int)

    coords = np.column_stack([lat, lon])
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(coords_scaled)
    return labels


def compute_cluster_centers(
    labels: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    chl: np.ndarray,
    pixel_area_km2: float | None = None,
) -> pd.DataFrame:
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    if len(unique_labels) == 0:
        return pd.DataFrame(
            columns=[
                "cluster", "center_lat", "center_lon", "mean_chl",
                "median_chl", "n_pixels", "area_km2", "lat_std", "lon_std",
            ]
        )

    records = []
    for cl in unique_labels:
        mask = labels == cl
        cl_lat = lat[mask]
        cl_lon = lon[mask]
        cl_chl = chl[mask]

        if pixel_area_km2 is not None:
            area = float(np.sum(mask) * pixel_area_km2)
        else:
            dlat = np.abs(np.median(np.diff(np.sort(np.unique(lat))))) if len(np.unique(lat)) > 1 else 0.04166
            mean_lat_rad = np.deg2rad(np.mean(cl_lat))
            dlat_rad = np.deg2rad(dlat)
            dlon_rad = dlat_rad
            pix_area = R_EARTH_KM**2 * np.cos(mean_lat_rad) * dlat_rad * dlon_rad
            area = float(np.sum(mask) * pix_area)

        records.append({
            "cluster": int(cl),
            "center_lat": float(np.mean(cl_lat)),
            "center_lon": float(np.mean(cl_lon)),
            "mean_chl": float(np.nanmean(cl_chl)),
            "median_chl": float(np.nanmedian(cl_chl)),
            "n_pixels": int(np.sum(mask)),
            "area_km2": area,
            "lat_std": float(np.std(cl_lat)),
            "lon_std": float(np.std(cl_lon)),
        })

    return pd.DataFrame(records).sort_values("center_lat", ascending=False).reset_index(drop=True)


def select_optimal_k(
    lat: np.ndarray,
    lon: np.ndarray,
    k_range: range = range(2, 6),
    seed: int = 0,
) -> pd.DataFrame:
    coords = np.column_stack([lat, lon])
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(coords_scaled)
        sil = silhouette_score(coords_scaled, labels)
        records.append({"k": k, "silhouette": sil, "inertia": km.inertia_})

    return pd.DataFrame(records)


def _match_clusters_by_latitude(centers_df: pd.DataFrame) -> pd.DataFrame:
    if len(centers_df) == 0:
        return centers_df
    return centers_df.sort_values("center_lat", ascending=False).reset_index(drop=True)


def _match_to_reference(
    centers_df: pd.DataFrame,
    ref_centers: np.ndarray,
) -> pd.DataFrame:
    if len(centers_df) == 0:
        return centers_df

    result = centers_df.copy()
    assigned = set()
    new_labels = {}

    for ref_idx in range(len(ref_centers)):
        ref_lat, ref_lon = ref_centers[ref_idx]
        best_dist = np.inf
        best_cl = None
        for _, row in result.iterrows():
            cl = row["cluster"]
            if cl in assigned:
                continue
            dist = np.hypot(row["center_lat"] - ref_lat, row["center_lon"] - ref_lon)
            if dist < best_dist:
                best_dist = dist
                best_cl = cl
        if best_cl is not None:
            new_labels[best_cl] = ref_idx
            assigned.add(best_cl)

    result["cluster"] = result["cluster"].map(new_labels).fillna(-1).astype(int)
    return result[result["cluster"] >= 0].reset_index(drop=True)


def run_yearly_clustering(
    chl_da,
    lat: np.ndarray,
    lon: np.ndarray,
    percentile: float = 85,
    method: str = "hdbscan",
    k: int = 2,
    min_cluster_size: int = 300,
    min_samples: int = 50,
    seed: int = 0,
    reference_centers: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = np.unique(chl_da.time.dt.year.values)

    if reference_centers is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chl_clim = np.nanmean(chl_da.values, axis=0)
        lat_ref, lon_ref, chl_ref, _ = extract_high_chl_pixels(chl_clim, lat, lon, percentile)
        if len(lat_ref) > 0:
            if method == "hdbscan":
                ref_labels, _ = cluster_pixels_hdbscan(
                    lat_ref, lon_ref,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )
            else:
                ref_labels = cluster_pixels_kmeans(lat_ref, lon_ref, k=k, seed=seed)
            ref_df = compute_cluster_centers(ref_labels, lat_ref, lon_ref, chl_ref)
            if len(ref_df) > 0:
                reference_centers = ref_df[["center_lat", "center_lon"]].values

    yearly_records = []
    all_yearly_centers = []

    for y in years:
        yearly = chl_da.sel(time=str(y))
        vals = yearly.values
        if np.all(np.isnan(vals)):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chl_ann = np.nanmean(vals, axis=0)
        ann_valid = chl_ann[~np.isnan(chl_ann)]
        if len(ann_valid) == 0:
            continue

        lat_sel, lon_sel, chl_sel, threshold = extract_high_chl_pixels(
            chl_ann, lat, lon, percentile
        )

        if len(lat_sel) == 0:
            continue

        if method == "hdbscan":
            labels, _ = cluster_pixels_hdbscan(
                lat_sel, lon_sel,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )
        elif method == "kmeans":
            labels = cluster_pixels_kmeans(lat_sel, lon_sel, k=k, seed=seed)
        else:
            raise ValueError(f"Unknown method: {method}")

        n_clusters = len(np.unique(labels[labels >= 0]))
        centers = compute_cluster_centers(labels, lat_sel, lon_sel, chl_sel)

        yearly_records.append({
            "year": y,
            "threshold": threshold,
            "n_high_chl_pixels": len(lat_sel),
            "n_clusters": n_clusters,
            "method": method,
        })

        if len(centers) > 0:
            if reference_centers is not None:
                centers_matched = _match_to_reference(centers, reference_centers)
            else:
                centers_matched = _match_clusters_by_latitude(centers)
            centers_matched["year"] = y
            centers_matched["method"] = method
            all_yearly_centers.append(centers_matched)

    yearly_df = pd.DataFrame(yearly_records)

    if len(all_yearly_centers) == 0:
        return yearly_df, pd.DataFrame()

    centers_all = pd.concat(all_yearly_centers, ignore_index=True)

    agg_records = []
    for cl_id in sorted(centers_all["cluster"].unique()):
        cl_data = centers_all[centers_all["cluster"] == cl_id]
        agg_records.append({
            "cluster": cl_id,
            "mean_center_lat": float(cl_data["center_lat"].mean()),
            "mean_center_lon": float(cl_data["center_lon"].mean()),
            "std_center_lat": float(cl_data["center_lat"].std()),
            "std_center_lon": float(cl_data["center_lon"].std()),
            "mean_chl": float(cl_data["mean_chl"].mean()),
            "mean_area_km2": float(cl_data["area_km2"].mean()),
            "n_years": len(cl_data),
            "method": method,
        })

    aggregated_df = pd.DataFrame(agg_records)
    return yearly_df, aggregated_df


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


def cluster_seasonal_hdbscan_pca(
    cycles: np.ndarray,
    min_cluster_size: int = 500,
    min_samples: int = 100,
    n_components: int | None = 3,
    cluster_selection_method: str = "leaf",
) -> tuple[np.ndarray, np.ndarray, object, object]:
    """
    Cluster pixels by seasonal cycle similarity using HDBSCAN on PCA-reduced
    z-scored cycles.

    Parameters
    ----------
    cycles : np.ndarray
        Shape (n_pixels, n_periods) - raw seasonal cycle vectors
    min_cluster_size : int
        Minimum cluster size for HDBSCAN
    min_samples : int
        Minimum samples for core distance
    n_components : int or None
        Number of PCA components. If None, auto-selects components explaining
        >=90% variance. Default 3 (optimal per DBCV grid search).
    cluster_selection_method : str
        "eom" (Excess of Mass) or "leaf" (fine-grained). Default "leaf".

    Returns
    -------
    labels : np.ndarray
        Cluster labels (-1 for noise)
    probabilities : np.ndarray
        Membership probabilities
    pca : PCA
        Fitted PCA object
    clusterer : HDBSCAN
        Fitted HDBSCAN object
    """
    from sklearn.decomposition import PCA

    n = cycles.shape[0]
    if n < min_cluster_size:
        return np.full(n, -1, dtype=int), np.zeros(n, dtype=float), None, None

    cycles_z = _zscore_cycles(cycles)

    if n_components is None:
        pca_full = PCA()
        pca_full.fit(cycles_z)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, 0.90) + 1)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(cycles_z)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(scores)
    probabilities = clusterer.probabilities_
    return labels, probabilities, pca, clusterer


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
