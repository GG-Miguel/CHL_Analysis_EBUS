import numpy as np
import xarray as xr


def compute_pixel_area_weights(lat: np.ndarray) -> np.ndarray:
    from src.utils.config import R_EARTH_KM
    lat_rad = np.deg2rad(lat)
    if len(lat) > 1:
        dlat = np.abs(np.diff(lat_rad)).mean()
    else:
        dlat = np.deg2rad(0.04166)
    dlon = dlat
    area = R_EARTH_KM**2 * np.cos(lat_rad) * dlat * dlon
    return area


def build_bbox_mask(ds: xr.Dataset, bbox) -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = bbox
    mask = (
        (ds.lon >= lon_min)
        & (ds.lon <= lon_max)
        & (ds.lat >= lat_min)
        & (ds.lat <= lat_max)
    )
    return mask
