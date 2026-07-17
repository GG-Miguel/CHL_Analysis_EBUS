# Paper 3 — Cluster Trend Analysis

Per-cluster trend analysis for chlorophyll in the Canarian upwelling region using the **same clustering methodology as notebook 08**.

## Methodology

1. **Clustering**: Reuses seasonal-cycle clustering from notebook 08
   - HDBSCAN (non-PCA): `min_cluster_size=100`, `min_samples=50`
   - K-means: `k=3`
   - Percentiles: P85, P90, P95, P99

2. **Background/Peak Separation**: For each cluster and year, applies the percentile threshold method (same as notebook 03):
   - **Background**: values below the percentile threshold
   - **Peak**: values at or above the threshold
   - Metrics computed:
     - `bg_mean`: mean background chlorophyll
     - `peak_mean`: mean peak chlorophyll
     - `integrated_chl`: integrated chlorophyll in peak area (chl × area)

3. **Trend Analysis**: For each cluster's annual time series:
   - Theil-Sen slope with 95% CI (per decade)
   - Mann-Kendall trend test
   - Significance stars (*** / ** / * / n.s.)

## Files

- `cluster_trends_all_percentiles.py` — Main analysis script
- `figures/` — Output figures (maps, time series, trend maps)
- `results/` — Output CSV files with metrics and trends

## Usage

```bash
cd /home/mgg/CHL_Analysis_EBUS
python paper3/cluster_trends_all_percentiles.py
```

## Output

- **Cluster maps**: `seasonal_map_{method}_P{pct}.png`
- **Seasonal cycles**: `seasonal_cycles_{method}_P{pct}.png`
- **Trend maps**: `trend_map_{method}_{metric}_P{pct}.png`
- **Time series**: `timeseries_{method}_{metric}_P{pct}.png`
- **Metrics CSV**: `results/cluster_metrics_all.csv`
- **Trends CSV**: `results/cluster_trends_all.csv`

Where:
- `method` = `hdbscan` or `kmeans`
- `metric` = `bg_mean`, `peak_mean`, or `integrated_chl`
- `pct` = 85, 90, 95, or 99

## Key Differences from Notebook 09

Notebook 09 only analyzes P99 and uses HDBSCAN-PCA (with PCA dimensionality reduction). This analysis:
- Covers all percentiles (P85, P90, P95, P99)
- Uses the same HDBSCAN method as notebook 08 (no PCA)
- Uses the same clustering parameters as notebook 08
