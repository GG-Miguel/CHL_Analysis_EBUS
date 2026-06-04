#!/usr/bin/env python3
"""
Merge MODIS Aqua L3b 8-day chlorophyll into a time series
on the full integerized sinusoidal grid (NO spatial cropping).

Output convention:
  - Latitude  DECREASING north → south  (row 0 = northernmost)
  - Longitude INCREASING west → east    (2-D per row, sinusoidal)
"""

import os
from glob import glob
from datetime import datetime, timezone
import numpy as np
import netCDF4 as nc

# ---------------- CONFIG ----------------
FOLDER = "/data/mgg/Satelite/CHLa/MODIS_AQUA"
OUTPATH = os.path.join(FOLDER, "MODIS_AQUA_CHL_8D_timeseries_global.nc")


# ---------------- TIME ----------------
def parse_time(s):
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


# ---------------- DISCOVER FILES ----------------
files = sorted(glob(os.path.join(FOLDER, "AQUA_MODIS.*.nc")))

records = []
for f in files:
    with nc.Dataset(f) as ds:
        if getattr(ds, "temporal_range", None) != "8-day":
            continue
        records.append((f, ds.time_coverage_start, ds.time_coverage_end))

records.sort(key=lambda x: parse_time(x[1]))

if not records:
    raise SystemExit("No valid 8-day files found")


# ---------------- GRID INFO (from first file) ----------------
with nc.Dataset(records[0][0]) as ds:
    g = ds.groups["level-3_binned_data"]
    bi = g.variables["BinIndex"][:]

start_num = bi["start_num"].astype(np.int64)
max_bins = bi["max"].astype(np.int64)
nrows = start_num.shape[0]  # full ISIN row count (~1979)


# ---------------- ROW LATITUDE (NASA ISIN: row 0 = south) ----------------
row_lat = -90.0 + (np.arange(nrows) + 0.5) * 180.0 / nrows

# Keep all rows
ri = np.arange(nrows)
rl_sn = row_lat[ri].copy()
rsb_sn = start_num[ri].copy()
rmb_sn = max_bins[ri].copy()
nrow = len(ri)


# ---------------- FULL COLUMN RANGE PER ROW ----------------
col_start = np.zeros(nrow, dtype=np.int32)
col_count = np.zeros(nrow, dtype=np.int32)
for i in range(nrow):
    n = int(rmb_sn[i])
    col_start[i] = 0
    col_count[i] = n

ncol = int(col_count.max())
if ncol == 0:
    raise SystemExit("Empty grid")


# ---------------- 2-D LONGITUDE (south→north order) ----------------
lon2d = np.full((nrow, ncol), np.nan, dtype=np.float32)
for i in range(nrow):
    n = int(rmb_sn[i])
    nc_i = int(col_count[i])
    if nc_i > 0 and n > 0:
        cols = np.arange(nc_i)
        lon2d[i, :nc_i] = (360.0 * (cols + 0.5) / n - 180.0).astype(np.float32)


# ---------------- FLIP TO NORTH→SOUTH FOR OUTPUT ----------------
rl_ns = rl_sn[::-1].copy()
rsb_ns = rsb_sn[::-1].copy()
rmb_ns = rmb_sn[::-1].copy()
col_start_ns = col_start[::-1].copy()
col_count_ns = col_count[::-1].copy()
lon2d_ns = lon2d[::-1, :].copy()


# ---------------- OUTPUT FILE ----------------
if os.path.exists(OUTPATH):
    os.remove(OUTPATH)

ds_out = nc.Dataset(OUTPATH, "w", format="NETCDF4")

ds_out.createDimension("time", None)
ds_out.createDimension("row", nrow)
ds_out.createDimension("col", ncol)

v_time = ds_out.createVariable("time", "f8", ("time",))
v_time.units = "days since 1970-01-01 00:00:00"
v_time.calendar = "standard"
v_time.standard_name = "time"

ds_out.createVariable("row", "i4", ("row",))[:] = np.arange(nrow)
ds_out.createVariable("col", "i4", ("col",))[:] = np.arange(ncol)

v_lat = ds_out.createVariable("lat", "f4", ("row",))
v_lat.units = "degrees_north"
v_lat.standard_name = "latitude"
v_lat[:] = rl_ns

v_lon = ds_out.createVariable("lon", "f4", ("row", "col"))
v_lon.units = "degrees_east"
v_lon.standard_name = "longitude"
v_lon[:] = lon2d_ns

ds_out.createVariable("row_max_bins", "i4", ("row",))[:] = rmb_ns
ds_out.createVariable("row_start_bin", "i4", ("row",))[:] = rsb_ns
ds_out.createVariable("row_col_start", "i4", ("row",))[:] = col_start_ns
ds_out.createVariable("row_col_count", "i4", ("row",))[:] = col_count_ns

v_chl = ds_out.createVariable(
    "chlor_a", "f4",
    ("time", "row", "col"),
    zlib=True, complevel=4,
    fill_value=np.nan,
)
v_chl.units = "mg m^-3"
v_chl.standard_name = "mass_concentration_of_chlorophyll_a_in_sea_water"

ds_out.sync()


# ---------------- MAIN LOOP ----------------
for t, (fpath, tstart, tend) in enumerate(records):

    with nc.Dataset(fpath) as ds:
        g = ds.groups["level-3_binned_data"]

        bi2 = g.variables["BinIndex"][:]
        if bi2["start_num"].shape[0] != nrows:
            raise SystemExit(
                f"{fpath}: BinIndex has {bi2['start_num'].shape[0]} rows, "
                f"expected {nrows}"
            )

        bl = g.variables["BinList"][:]
        ch = g.variables["chlor_a"][:]

        bn = bl["bin_num"].astype(np.int64)
        wg = bl["weights"].astype(np.float32)
        sm = ch["sum"].astype(np.float32)

        vals = np.full_like(sm, np.nan, dtype=np.float32)
        ok = wg > 0
        vals[ok] = sm[ok] / wg[ok]

        bin_row = np.searchsorted(start_num, bn, side="right") - 1

        br_sn = bin_row                  # local row 0 … nrow-1 (south → north)
        gn = bn
        vk = vals

        gc = gn - start_num[bin_row]
        lc_sn = gc - col_start[br_sn]
        kc = (lc_sn >= 0) & (lc_sn < col_count[br_sn])

        data_sn = np.full((nrow, ncol), np.nan, dtype=np.float32)
        data_sn[br_sn[kc], lc_sn[kc]] = vk[kc]

        data_ns = data_sn[::-1, :].copy()

    t0 = parse_time(tstart)
    t1 = parse_time(tend)
    tm = (t0 + (t1 - t0) / 2).replace(tzinfo=None)

    v_time[t] = nc.date2num(tm, units=v_time.units, calendar="standard")
    v_chl[t] = data_ns

    if (t + 1) % 10 == 0:
        ds_out.sync()
        print(f"{t+1}/{len(records)} processed")

ds_out.close()
print("DONE:", OUTPATH)
