import numpy as np
import xarray as xr


def standardize_longitude(ds: xr.Dataset, lon_name: str = "lon") -> xr.Dataset:
    if lon_name not in ds.coords and lon_name not in ds.dims:
        if lon_name in ds.data_vars:
            ds = ds.assign_coords({lon_name: ds[lon_name]})
        else:
            return ds
    lon = ds[lon_name]
    if lon.values.max() > 180:
        new_lon = ((lon + 180) % 360) - 180
        ds = ds.assign_coords({lon_name: new_lon}).sortby(lon_name)
    return ds


def compute_pixel_area(lat: np.ndarray, r: float = 6371.0) -> np.ndarray:
    from src.utils.config import R_EARTH_KM, NATIVE_CADENCE_DAYS
    lat_rad = np.deg2rad(lat)
    dlat = np.abs(np.diff(lat_rad)).mean() if len(lat_rad) > 1 else np.deg2rad(0.04166)
    dlon = dlat
    area = r**2 * np.cos(lat_rad) * dlat * dlon
    return area
