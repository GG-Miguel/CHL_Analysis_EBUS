# CHL_Analysis_EBUS — MODIS-Aqua Chlorophyll-a Pipeline (Canary EBUS)

Reproducible analysis pipeline for satellite-derived chlorophyll-a (Chl-a)
concentration in the **Canary Eastern Boundary Upwelling System** (Canary
EBUS), based on **MODIS-Aqua** L3b 8-day binned data.

The pipeline ingests the merged binned data, computes a sequence of
interannual metrics (annual peaks, multi-threshold patch areas, seasonal
cycles), trend-tests every series with non-parametric statistics
(Theil–Sen + Mann–Kendall), and renders a complete figure set ready for
a manuscript.

---

## What the pipeline does

| Step | Notebook | Inputs | Outputs | What it does |
|------|----------|--------|---------|--------------|
| 0 | `00_data_cleaning.py` | merged `.nc` | `*_clean.nc` | Clips Chl-a to a physically realistic range (sets outliers to NaN). |
| 1 | `01_data_ingestion.py` | clean `.nc` | `01_mean_map.png`, `01_spatial_mean_ts.png` | Loads the dataset, plots the multi-year mean Chl-a map and the spatial-mean time series. |
| 2 | `02_annual_peaks.py` | clean `.nc` | `results/annual_peaks.csv`, `02_annual_peaks.png` | Per-year 99th-percentile peak + Theil–Sen + Mann–Kendall on peak intensity; circular DOY test on peak timing. |
| 3 | `03_threshold_area.py` | clean `.nc` | `results/threshold_area_metrics.csv`, patch-overlay PNGs | Per-year patch metrics above the P85, P90, P95, P99 percentiles, **with bootstrap 95% CIs** for each annual metric. |
| 4 | `04_seasonal.py` | clean `.nc` | `results/seasonal_per_year.csv`, `results/seasonal_trend.csv`, climatology + trend PNGs | 8-day / monthly climatology, per-year seasonal amplitude and peak-DOY, trend-tested. |
| 5 | `05_trends.py` | threshold CSV | `results/trend_significance.csv`, `05_*.png` | Full Theil–Sen + Mann–Kendall table for every metric × threshold, with rendered table image and CI bands on the time series. |
| 6 | `06_figures.py` | all above | `Fig1_…`–`Fig5_*.png` | Final manuscript-ready figure set. |

`run_all.py` runs steps 1–6 in order.

---

## Statistical methodology

All trend tests live in `src/statistics/significance.py`:

* **`theil_sen_with_ci(y)`** — Theil–Sen slope with 95% confidence
  interval. `slope_pct_per_decade = 100 × (slope / median(y)) × 10`,
  the relative change per decade in the units of `y`.
* **`mann_kendall(y)`** — Two-sided Mann–Kendall test (Hamed & Rao 1998
  variance correction is applied automatically when ties are present).
  Returns τ, z, p, and a trend label.
* **`circular_doy_trend(peak_doy)`** — For circular variables like
  day-of-year. Converts DOY to an angle, fits Theil–Sen independently on
  cos θ and sin θ, and reports the mean direction, drift magnitude in
  degrees/year, and a Rayleigh p-value.
* **`bootstrap_patch_metrics(chl_2d, area_2d, percentile, n_boot=200)`**
  — Pixel-level bootstrap (with replacement) of the per-year patch
  metrics. Returns 95% CIs on the threshold, area, mean Chl, etc.
* **`p_value_to_stars(p)`** — `***` p<0.001, `**` p<0.01, `*` p<0.05,
  `n.s.` otherwise.

Convenience: `src/statistics/trends.py` keeps the legacy
`theil_sen_trend` API as a thin wrapper and exposes
`trend_summary_table(df, year_col, value_cols)` for building wide
significance tables.

---

## Repository layout

```
paper3/
├── README.md
├── requirements.txt
├── run_all.py
├── merge_modis_aqua_full_grid.py
├── .gitignore
├── data_local/                # symlink to the merged MODIS NetCDF
├── notebooks/                 # 00–06, run in order
├── src/
│   ├── utils/                 # config, helpers
│   ├── preprocessing/         # ingestion, transforms, thresholds
│   ├── statistics/            # peaks, trends, SIGNIFICANCE
│   └── visualization/         # style, maps
├── tests/                     # test_significance.py (11 unit tests)
├── results/                   # generated CSVs (gitignored)
└── figures/                   # generated PNGs (gitignored)
```

---

## Inputs

The pipeline expects a single NetCDF file at the path defined in
`src/utils/config.py` (`MODISA_FILE`), cropped to the Canary EBUS
bounding box (`CANARY_BBOX`):

| Variable | Default value |
|----------|---------------|
| `MODISA_FILE` | `MODIS_AQUA_CHL_8D_timeseries_canary_ebus.nc` |
| `CANARY_BBOX` | `lon ∈ [−30, −7]°, lat ∈ [10, 45]°` |
| `THRESHOLD_PCTS` | `[85, 90, 95, 99]` |
| `CHL_CLIP_MAX` | `80.0` mg m⁻³ |

The file should expose `lat`, `lon`, `chlor_a` (or `chl` after
renaming) and a `time` dimension at 8-day cadence.

The raw data is built by `merge_modis_aqua_full_grid.py` from the
NASA OBPG L3b binned files, then cropped to the bbox in
`data_local/`.

The path can be overridden with the `CHL_DATA_DIR` environment
variable:

```bash
export CHL_DATA_DIR=/path/to/CHLa
```

---

## How to run

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. (optional) point at your data
export CHL_DATA_DIR=/Users/GGMig

# 3. Run the full pipeline
python run_all.py
```

To skip the slow bootstrap step (saves ~5 minutes), set
`N_BOOT=0` (or any small number):

```bash
N_BOOT=50 python run_all.py
```

To run a single step:

```bash
python notebooks/02_annual_peaks.py
python notebooks/03_threshold_area.py
# ...
```

### Tests

```bash
python tests/test_significance.py
```

Expected: `11/11 tests passed`. (The test runner has no `pytest`
dependency.)

---

## Outputs (regenerated each run)

### CSV tables (`results/`, gitignored)

| File | What it contains |
|------|------------------|
| `annual_peaks.csv` | `year, chl_peak, time_of_peak, peak_doy` |
| `threshold_area_metrics.csv` | Per-year metrics × 4 thresholds, with `_lo` / `_hi` 95% bootstrap CI columns |
| `seasonal_per_year.csv` | Per-year seasonal amplitude and peak DOY |
| `seasonal_trend.csv` | One-row trend summary for amplitude and peak DOY |
| `trend_significance.csv` | Theil–Sen slope, CI, MK τ/p, stars for every metric × threshold |

### Figures (`figures/`, gitignored)

```
01_mean_map.png            03_threshold_timeseries.png
01_spatial_mean_ts.png     03_threshold_metrics_ts.png
02_annual_peaks.png        03_patch_overlay_2003.png
04_climatology.png         03_patch_overlay_2010.png
04_monthly_climatology.png 03_patch_overlay_2020.png
04_seasonal_trend.png
05_trends_p85.png          05_significance_table.png
05_trends_p90.png          05_trend_bar.png
05_trends_p95.png
05_trends_p99.png
Fig1_mean_map.png          Fig4_trend_bar.png
Fig2_patch_2003.png        Fig5_significance_table.png
Fig2_patch_2010.png
Fig2_patch_2020.png
Fig3_metrics_timeseries.png
```

The `Fig*` files are the manuscript-ready set; the `0X_*` files are the
working versions produced by each notebook step.

---

## Headline findings (Canary EBUS, 2002–2025, MODIS-Aqua L3b 8-day)

* **Annual P99 peak intensity** shows a non-significant positive trend
  (+0.07 mg m⁻³/yr, MK p=0.41).
* **Per-year peak day-of-year** has a small drift (~1.3 DOY/yr) but the
  Rayleigh test is non-significant (no preferred season in the drift
  direction).
* **Per-year seasonal amplitude** is essentially flat (+0.003 mg m⁻³/yr,
  MK p=0.75).
* **At the P99 threshold**, the *largest patch* has shrunk significantly
  over 2002–2025 (−459 km²/yr, MK p=0.027, *).
* **At the P90 threshold**, the *number of patches* has increased
  significantly (+3.3 patches/yr, MK p=0.047, *), consistent with the
  largest-patch contraction: the high-Chl area is fragmenting rather
  than shifting in magnitude.
* All other metric × threshold combinations are not significant
  (p > 0.05) over the 24-year record.

These results are illustrative of the pipeline; please re-run
`python run_all.py` and inspect `results/trend_significance.csv` for
the current numbers.

---

## License

Research code; please add a license before public distribution.
