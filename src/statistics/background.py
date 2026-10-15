import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.regression.quantile_regression import QuantReg


def annual_percentile_metrics(
    chl: xr.DataArray,
    background_pcts: list[float] = [10, 25],
    peak_pcts: list[float] = [90, 95, 99],
) -> pd.DataFrame:
    years = np.unique(chl.time.dt.year.values)
    records = []
    for y in years:
        yearly = chl.sel(time=str(y))
        vals = yearly.values
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            continue
        row = {"year": int(y), "spatial_mean": float(np.nanmean(valid))}
        for p in background_pcts:
            row[f"bg_p{p}"] = float(np.percentile(valid, p))
        for p in peak_pcts:
            row[f"peak_p{p}"] = float(np.percentile(valid, p))
        all_pcts = sorted(background_pcts + peak_pcts)
        if len(all_pcts) >= 2:
            lo = float(np.percentile(valid, all_pcts[0]))
            hi = float(np.percentile(valid, all_pcts[-1]))
            row["peak_amplitude"] = hi - lo
        records.append(row)
    return pd.DataFrame(records)


def quantile_regression_trends(
    y: pd.Series,
    x: np.ndarray,
    quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
) -> pd.DataFrame:
    arr = np.asarray(y, dtype=float)
    mask = ~np.isnan(arr)
    y_ok = arr[mask]
    x_ok = np.asarray(x, dtype=float)[mask]
    n = len(y_ok)
    if n < 5:
        return pd.DataFrame(columns=["quantile", "slope", "intercept", "p_value", "n"])
    X = np.column_stack([np.ones(n), x_ok])
    records = []
    for q in quantiles:
        model = QuantReg(y_ok, X)
        res = model.fit(q=q)
        records.append({
            "quantile": q,
            "intercept": float(res.params[0]),
            "slope": float(res.params[1]),
            "slope_per_decade": float(res.params[1] * 10.0),
            "p_value": float(res.pvalues[1]),
            "n": n,
        })
    return pd.DataFrame(records)
