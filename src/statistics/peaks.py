import numpy as np
import xarray as xr
import pandas as pd


PCT = 99


def annual_peak_metrics(chl: xr.DataArray) -> pd.DataFrame:
    years = np.unique(chl.time.dt.year.values)
    records = []
    for y in years:
        yearly = chl.sel(time=str(y))
        vals = yearly.values

        peak_val = -1.0
        peak_time = None
        for t in range(vals.shape[0]):
            ts = vals[t]
            valid = ts[~np.isnan(ts)]
            if len(valid) == 0:
                continue
            p99 = float(np.percentile(valid, PCT))
            if p99 > peak_val:
                peak_val = p99
                peak_time = yearly.time.values[t]

        records.append({
            "year": y,
            "chl_peak": peak_val if peak_val >= 0 else np.nan,
            "time_of_peak": peak_time,
        })
    return pd.DataFrame(records)
