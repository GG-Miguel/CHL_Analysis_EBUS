"""
# 00 — Data Cleaning
Clip chlorophyll to a physically realistic range [0.001, 20] mg/m³.
Values outside this range (bad retrievals, cloud edges) are set to NaN.
Creates a cleaned copy of the merged file.
"""

import xarray as xr
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

from src.utils.config import DATA_DIR, MODISA_FILE, CHL_CLIP_MIN, CHL_CLIP_MAX
from src.preprocessing.transforms import clip_chl_outliers

src_path = DATA_DIR / MODISA_FILE
stem = src_path.stem
clean_path = src_path.with_name(f"{stem}_clean.nc")

print(f"Source: {src_path}")
print(f"Output: {clean_path}")

ds = xr.open_dataset(src_path)
print(f"Before: chl range [{float(ds['chlor_a'].min()):.4f}, {float(ds['chlor_a'].max()):.4f}]")
print(f"  NaNs: {float(np.isnan(ds['chlor_a'].values).sum())}")

ds = clip_chl_outliers(ds, varname="chlor_a", vmin=CHL_CLIP_MIN, vmax=CHL_CLIP_MAX)

print(f"After:  chl range [{float(ds['chlor_a'].min()):.4f}, {float(ds['chlor_a'].max()):.4f}]")
print(f"  NaNs: {float(np.isnan(ds['chlor_a'].values).sum())}")
print(f"  Outliers removed: {float((ds['chlor_a'].values < CHL_CLIP_MIN).sum() + (ds['chlor_a'].values > CHL_CLIP_MAX).sum())}")

ds.to_netcdf(clean_path)
ds.close()
print(f"Saved: {clean_path}")
print("Step 0 complete.")
