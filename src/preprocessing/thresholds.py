import numpy as np
import xarray as xr
import pandas as pd
from skimage import measure
from src.utils.config import R_EARTH_KM


def pooled_percentile_threshold(chl: xr.DataArray, percentile: float = 85) -> float:
    valid = chl.values[~np.isnan(chl.values)]
    if len(valid) == 0:
        return np.nan
    return float(np.percentile(valid, percentile))


def compute_pixel_areas(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if lat.ndim == 1 and lon.ndim == 2:
        lat_2d = lat[:, np.newaxis] * np.ones_like(lon)
    elif lat.ndim == 1 and lon.ndim == 1:
        lat_2d = lat[:, np.newaxis] * np.ones((len(lat), len(lon)))
    else:
        lat_2d = lat
    lat_rad = np.deg2rad(lat_2d)
    dlat = np.abs(np.diff(lat_rad, axis=0)).mean()
    dlon = np.abs(np.diff(lat_rad, axis=1)).mean() if lat_rad.shape[1] > 1 else dlat
    dlat_deg = np.abs(lat[1] - lat[0]) if len(lat) > 1 else 0.04166
    dlon_deg = dlat_deg
    dlat_rad = np.deg2rad(dlat_deg)
    dlon_rad = np.deg2rad(dlon_deg)
    area = R_EARTH_KM**2 * np.cos(lat_rad) * dlat_rad * dlon_rad
    return area


def threshold_area_metrics(chl: xr.DataArray, lat: np.ndarray, lon: np.ndarray,
                           percentile: float = 85) -> dict:
    threshold = pooled_percentile_threshold(chl, percentile)
    if np.isnan(threshold):
        return {"threshold": np.nan, "area_km2": 0, "mean_chl": np.nan,
                "integrated_chl": 0, "n_patches": 0, "largest_patch_area": 0,
                "centroid_lat": np.nan, "centroid_lon": np.nan}

    pixel_areas = compute_pixel_areas(lat, lon)
    chl_values = chl.values.squeeze()
    if chl_values.ndim > 2:
        chl_values = np.nanmean(chl_values, axis=0) if chl_values.shape[0] > 1 else chl_values[0]

    mask = (chl_values >= threshold) & (~np.isnan(chl_values))
    area_km2 = float(np.nansum(pixel_areas * mask))
    if area_km2 == 0:
        return {"threshold": threshold, "area_km2": 0, "mean_chl": np.nan,
                "integrated_chl": 0, "n_patches": 0, "largest_patch_area": 0,
                "centroid_lat": np.nan, "centroid_lon": np.nan}

    mean_chl = float(np.nanmean(chl_values[mask]))
    integrated_chl = float(np.nansum(chl_values[mask] * pixel_areas[mask]))

    labeled, n_labels = measure.label(mask.astype(int), connectivity=1, return_num=True)
    n_patches = int(n_labels)
    patch_areas = []
    centroids = []
    for i in range(1, n_labels + 1):
        pmask = labeled == i
        pa = float(np.nansum(pixel_areas[pmask]))
        patch_areas.append(pa)
        ys, xs = np.where(pmask)
        if lat.ndim == 1 and lon.ndim == 2:
            c_lat = float(np.mean(lat[ys]))
            c_lon = float(np.mean(lon[ys, xs]))
        elif lat.ndim == 2 and lon.ndim == 2:
            c_lat = float(np.mean(lat[ys, xs]))
            c_lon = float(np.mean(lon[ys, xs]))
        else:
            c_lat = float(np.mean(lat[ys]))
            c_lon = float(np.mean(lon[xs]))
        centroids.append((c_lat, c_lon))

    patch_areas = np.array(patch_areas)
    largest_idx = int(np.argmax(patch_areas))
    largest_patch_area = float(patch_areas[largest_idx])
    centroid_lat, centroid_lon = centroids[largest_idx]

    return {
        "threshold": threshold,
        "area_km2": area_km2,
        "mean_chl": mean_chl,
        "integrated_chl": integrated_chl,
        "n_patches": n_patches,
        "largest_patch_area": largest_patch_area,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
    }
