import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf


def theil_sen_trend(y: pd.Series) -> dict:
    x = np.arange(len(y))
    mask = ~y.isna()
    if mask.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "p_value": np.nan,
                "slope_pct_per_decade": np.nan}
    x_ok = x[mask]
    y_ok = y[mask].values
    res = stats.theilslopes(y_ok, x_ok)
    slope, intercept, lo_slope, hi_slope = res
    _, p_value = stats.mannwhitneyu(y_ok[:len(y_ok)//2], y_ok[len(y_ok)//2:],
                                     alternative='two-sided') if len(y_ok) >= 6 else (np.nan, np.nan)
    median_y = np.nanmedian(y_ok)
    slope_pct = ((10**slope - 1) * 100) if median_y > 0 else np.nan
    return {
        "slope": slope,
        "intercept": intercept,
        "p_value": p_value,
        "slope_pct_per_decade": slope_pct * 10 if not np.isnan(slope_pct) else np.nan,
    }


def ols_ar1_trend(y: pd.Series) -> dict:
    import statsmodels.api as sm
    x = np.arange(len(y))
    mask = ~y.isna()
    if mask.sum() < 3:
        return {"slope": np.nan, "p_value": np.nan, "rsquared": np.nan}
    x_ok = sm.add_constant(x[mask])
    y_ok = y[mask].values
    model = sm.OLS(y_ok, x_ok).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
    slope = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
    p_value = model.pvalues.iloc[1] if hasattr(model.pvalues, 'iloc') else model.pvalues[1]
    return {"slope": slope, "p_value": p_value, "rsquared": float(model.rsquared)}
