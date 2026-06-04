import numpy as np
import xarray as xr


def log10_transform_chl(chl: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(chl, 1e-6))


def clip_chl_outliers(
    ds: xr.Dataset,
    varname: str = "chl",
    vmin: float = 0.001,
    vmax: float = 20.0,
) -> xr.Dataset:
    """
    Clip chlorophyll to a physically realistic range.
    Values outside [vmin, vmax] are set to NaN.
    """
    da = ds[varname]
    ds[varname] = da.where((da >= vmin) & (da <= vmax))
    return ds
