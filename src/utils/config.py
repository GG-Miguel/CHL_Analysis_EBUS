from pathlib import Path
import os

CANARY_BBOX = {"lon_min": -30.0, "lon_max": -7.0, "lat_min": 10.0, "lat_max": 45.0}

THRESHOLD_PERCENTILE = 85

NATIVE_CADENCE_DAYS = 8

# Allow the same codebase to run on the laptop and the server without
# hard-coding a machine-specific path. Set PAPER3_DATA_DIR to override.
_DEFAULT_LAPTOP_DIR = Path("/Users/GGMig/paper3/data_local/")
_DEFAULT_SERVER_DIR = Path("/data/mgg/Satelite/CHLa/MODIS_AQUA")
_DEFAULT_DATA_DIR = _DEFAULT_LAPTOP_DIR if _DEFAULT_LAPTOP_DIR.exists() else _DEFAULT_SERVER_DIR
DATA_DIR = Path(os.environ.get("PAPER3_DATA_DIR", _DEFAULT_DATA_DIR))

FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")
NOTEBOOKS_DIR = Path("notebooks")

MODISA_FILE = "MODIS_AQUA_CHL_8D_timeseries_canary_ebus.nc"

PRODUCT_NAME = "MODIS-Aqua"
CHL_VARNAME = "chlor_a"
LAT_NAME = "lat"
LON_NAME = "lon"
TIME_NAME = "time"

R_EARTH_KM = 6371.0

CHL_CLIP_MAX = 80.0

THRESHOLD_PCTS = [85, 90, 95, 99]
