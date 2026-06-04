from pathlib import Path
import xarray as xr
import numpy as np
from src.utils.config import CHL_VARNAME, LAT_NAME, LON_NAME, CHL_CLIP_MAX


def load_chlorophyll_dataset(filepath: str | Path, clip: bool = True) -> xr.Dataset:
    ds = xr.open_dataset(filepath)

    if LAT_NAME in ds.data_vars:
        ds = ds.assign_coords({LAT_NAME: ds[LAT_NAME]})
    if LON_NAME in ds.data_vars:
        ds = ds.assign_coords({LON_NAME: ds[LON_NAME]})

    if CHL_VARNAME in ds.data_vars:
        ds = ds.rename({CHL_VARNAME: "chl"})

    if clip and "chl" in ds.data_vars:
        ds["chl"] = ds["chl"].where(ds["chl"] <= CHL_CLIP_MAX)

    return ds
