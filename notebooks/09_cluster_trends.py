"""
# 09 — Per-Cluster Trend Analysis with Background/Peak Separation
For each seasonal-cycle cluster (HDBSCAN-PCA and K-means), compute per-year
background and peak chlorophyll metrics using the P99 threshold method,
then run Theil-Sen + Mann-Kendall trend tests. Produces maps annotated
with trend direction and significance.
"""

import os
import time
import warnings
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from src.visualization.style import set_nature_style
set_nature_style()
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import DATA_DIR, MODISA_FILE, R_EARTH_KM
from src.preprocessing.ingestion import load_chlorophyll_dataset
from src.models.clustering import (
    compute_seasonal_cycles,
    cluster_seasonal_hdbscan_pca,
    cluster_seasonal_kmeans,
    _zscore_cycles,
)
from src.statistics.significance import theil_sen_with_ci, mann_kendall, p_value_to_stars
from src.visualization.cluster_plots import (
    plot_seasonal_cluster_map,
    plot_seasonal_cycles_by_cluster,
    plot_cluster_trend_map,
    CLUSTER_COLORS,
)

FILEPATH = DATA_DIR / MODISA_FILE
ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]
lat = ds.lat.values
lon = ds.lon.values
lat_2d = lat[:, np.newaxis] * np.ones_like(lon)

PERCENTILE = 99
K = 3
MIN_CLUSTER_SIZE = 500
MIN_SAMPLES = 100
N_PCA_COMPONENTS = 3
SEED = 0

dlat_rad = np.deg2rad(np.abs(lat[1] - lat[0])) if len(lat) > 1 else np.deg2rad(0.04166)
dlon_rad = dlat_rad
lat_rad = np.deg2rad(lat_2d)
pixel_area = R_EARTH_KM**2 * np.cos(lat_rad) * dlat_rad * dlon_rad

years = np.unique(chl.time.dt.year.values)

print(f"Data shape: {chl.shape}")
print(f"Percentile: P{PERCENTILE}")
print(f"HDBSCAN-PCA: min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}, "
      f"n_components={N_PCA_COMPONENTS}")
print(f"K-means k: {K}")
print(f"Years: {years[0]}–{years[-1]}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    chl_clim = np.nanmean(chl.values, axis=0)

print("\n--- Computing seasonal cycle vectors (P99) ---")
t0 = time.time()
cycles, lat_sel, lon_sel, periods, ys_idx, xs_idx = compute_seasonal_cycles(
    chl, lat, lon,
    percentile=PERCENTILE,
    temporal_resolution="8day",
)
print(f"Seasonal cycles computed: {time.time() - t0:.1f}s")
print(f"Valid pixels: {len(lat_sel)}")

print("\n--- HDBSCAN-PCA on seasonal cycles ---")
t0 = time.time()
hdbscan_labels, hdbscan_probs, pca_obj, clusterer_obj = cluster_seasonal_hdbscan_pca(
    cycles,
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    n_components=N_PCA_COMPONENTS,
    cluster_selection_method="leaf",
)
print(f"HDBSCAN-PCA complete: {time.time() - t0:.1f}s")
n_hdbscan = len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
n_noise = np.sum(hdbscan_labels < 0)
print(f"HDBSCAN clusters found: {n_hdbscan}")
print(f"Noise pixels: {n_noise} ({100*n_noise/len(lat_sel):.1f}%)")
if pca_obj is not None:
    cumvar = np.cumsum(pca_obj.explained_variance_ratio_)
    print(f"PCA variance explained: {pca_obj.explained_variance_ratio_}")
    print(f"PCA cumulative: {cumvar}")

plot_seasonal_cluster_map(
    lon, lat, chl_clim,
    lat_sel, lon_sel, hdbscan_labels,
    title=f"HDBSCAN-PCA — Seasonal Cycles (P{PERCENTILE})",
    savepath=f"figures/09_seasonal_map_hdbscan_P{PERCENTILE}.png",
)
print(f"Saved: figures/09_seasonal_map_hdbscan_P{PERCENTILE}.png")

plot_seasonal_cycles_by_cluster(
    cycles, hdbscan_labels, periods,
    temporal_resolution="8day",
    savepath=f"figures/09_seasonal_cycles_hdbscan_P{PERCENTILE}.png",
)
print(f"Saved: figures/09_seasonal_cycles_hdbscan_P{PERCENTILE}.png")

print(f"\n--- K-means (k={K}) on seasonal cycles ---")
kmeans_labels = cluster_seasonal_kmeans(cycles, k=K, seed=SEED)
n_kmeans = len(np.unique(kmeans_labels[kmeans_labels >= 0]))
print(f"K-means clusters: {n_kmeans}")

plot_seasonal_cluster_map(
    lon, lat, chl_clim,
    lat_sel, lon_sel, kmeans_labels,
    title=f"K-means (k={K}) — Seasonal Cycles (P{PERCENTILE})",
    savepath=f"figures/09_seasonal_map_kmeans_P{PERCENTILE}.png",
)
print(f"Saved: figures/09_seasonal_map_kmeans_P{PERCENTILE}.png")

plot_seasonal_cycles_by_cluster(
    cycles, kmeans_labels, periods,
    temporal_resolution="8day",
    savepath=f"figures/09_seasonal_cycles_kmeans_P{PERCENTILE}.png",
)
print(f"Saved: figures/09_seasonal_cycles_kmeans_P{PERCENTILE}.png")


def compute_cluster_yearly_metrics(
    chl_da, lat_sel, lon_sel, cluster_labels, years, pixel_area_2d, percentile,
    ys_idx, xs_idx,
):
    """
    For each cluster and each year, compute background/peak metrics
    using the P99 threshold within that cluster's pixels.
    """
    n_years = len(years)
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters >= 0]

    records = []
    for cl in unique_clusters:
        mask_cl = cluster_labels == cl
        cl_ys = ys_idx[mask_cl]
        cl_xs = xs_idx[mask_cl]

        for yi, y in enumerate(years):
            yearly = chl_da.sel(time=str(y))
            vals = yearly.values
            if np.all(np.isnan(vals)):
                continue
            chl_ann = np.nanmean(vals, axis=0)

            cl_chl = chl_ann[cl_ys, cl_xs]
            cl_area = pixel_area_2d[cl_ys, cl_xs]

            valid = ~np.isnan(cl_chl)
            if valid.sum() < 3:
                continue

            chl_valid = cl_chl[valid]
            area_valid = cl_area[valid]

            threshold = float(np.percentile(chl_valid, percentile))

            bg_mask = chl_valid < threshold
            peak_mask = chl_valid >= threshold

            bg_mean = float(np.nanmean(chl_valid[bg_mask])) if bg_mask.any() else np.nan
            peak_mean = float(np.nanmean(chl_valid[peak_mask])) if peak_mask.any() else np.nan
            peak_area = float(np.nansum(area_valid[peak_mask])) if peak_mask.any() else 0.0
            integrated_chl = float(np.nansum(chl_valid[peak_mask] * area_valid[peak_mask])) if peak_mask.any() else 0.0

            records.append({
                "cluster": int(cl),
                "year": int(y),
                "threshold": threshold,
                "n_pixels": int(valid.sum()),
                "n_bg": int(bg_mask.sum()),
                "n_peak": int(peak_mask.sum()),
                "bg_mean": bg_mean,
                "peak_mean": peak_mean,
                "peak_area_km2": peak_area,
                "integrated_chl": integrated_chl,
            })

    return pd.DataFrame(records)


def compute_trends(df, method_name, cluster_labels, lat_sel, lon_sel):
    """Run Theil-Sen + MK on each cluster's metrics."""
    unique_clusters = sorted(df["cluster"].unique())
    trend_records = []

    for cl in unique_clusters:
        cl_df = df[df["cluster"] == cl].sort_values("year")
        if len(cl_df) < 4:
            continue

        cl_lat = lat_sel[cluster_labels == cl]
        cl_lon = lon_sel[cluster_labels == cl]

        for metric in ["bg_mean", "peak_mean", "integrated_chl"]:
            y = cl_df[metric]
            x = cl_df["year"].astype(float)

            ts = theil_sen_with_ci(y, x=x)
            mk = mann_kendall(y, x=x)

            trend_records.append({
                "method": method_name,
                "cluster": int(cl),
                "metric": metric,
                "n_years": int(len(cl_df)),
                "center_lat": float(np.mean(cl_lat)),
                "center_lon": float(np.mean(cl_lon)),
                "slope_per_decade": ts["slope_per_decade"],
                "slope_lo_per_decade": ts["slope_lo_per_decade"],
                "slope_hi_per_decade": ts["slope_hi_per_decade"],
                "slope_pct_per_decade": ts["slope_pct_per_decade"],
                "intercept": ts["intercept"],
                "intercept_at_x0": ts["intercept_at_x0"],
                "mk_tau": mk["tau"],
                "mk_p": mk["p_value"],
                "mk_trend": mk["trend"],
                "stars": p_value_to_stars(mk["p_value"]),
            })

    return pd.DataFrame(trend_records)


print("\n--- Computing per-cluster yearly metrics (HDBSCAN-PCA) ---")
t0 = time.time()
hdbscan_metrics = compute_cluster_yearly_metrics(
    chl, lat_sel, lon_sel, hdbscan_labels, years, pixel_area, PERCENTILE,
    ys_idx, xs_idx,
)
print(f"Done: {time.time() - t0:.1f}s")
print(f"Records: {len(hdbscan_metrics)}")
hdbscan_metrics.to_csv(f"results/09_cluster_metrics_hdbscan_P{PERCENTILE}.csv", index=False)
print(f"Saved: results/09_cluster_metrics_hdbscan_P{PERCENTILE}.csv")

print("\n--- Computing per-cluster yearly metrics (K-means) ---")
t0 = time.time()
kmeans_metrics = compute_cluster_yearly_metrics(
    chl, lat_sel, lon_sel, kmeans_labels, years, pixel_area, PERCENTILE,
    ys_idx, xs_idx,
)
print(f"Done: {time.time() - t0:.1f}s")
print(f"Records: {len(kmeans_metrics)}")
kmeans_metrics.to_csv(f"results/09_cluster_metrics_kmeans_P{PERCENTILE}.csv", index=False)
print(f"Saved: results/09_cluster_metrics_kmeans_P{PERCENTILE}.csv")

print("\n--- Trend analysis (HDBSCAN-PCA) ---")
hdbscan_trends = compute_trends(hdbscan_metrics, "HDBSCAN-PCA", hdbscan_labels, lat_sel, lon_sel)
hdbscan_trends.to_csv(f"results/09_cluster_trends_hdbscan_P{PERCENTILE}.csv", index=False)
print(f"Saved: results/09_cluster_trends_hdbscan_P{PERCENTILE}.csv")
print(hdbscan_trends[["cluster", "metric", "slope_per_decade", "mk_p", "stars"]].to_string(index=False))

print("\n--- Trend analysis (K-means) ---")
kmeans_trends = compute_trends(kmeans_metrics, "K-means", kmeans_labels, lat_sel, lon_sel)
kmeans_trends.to_csv(f"results/09_cluster_trends_kmeans_P{PERCENTILE}.csv", index=False)
print(f"Saved: results/09_cluster_trends_kmeans_P{PERCENTILE}.csv")
print(kmeans_trends[["cluster", "metric", "slope_per_decade", "mk_p", "stars"]].to_string(index=False))

print("\n--- Generating trend maps ---")
for method, labels, trends_df in [
    ("hdbscan", hdbscan_labels, hdbscan_trends),
    ("kmeans", kmeans_labels, kmeans_trends),
]:
    for metric in ["bg_mean", "peak_mean", "integrated_chl"]:
        metric_trends = trends_df[trends_df["metric"] == metric]
        if len(metric_trends) == 0:
            continue

        metric_labels = {
            "bg_mean": "Background Chl-a",
            "peak_mean": "Peak Chl-a",
            "integrated_chl": "Integrated Peak Chl-a",
        }
        plot_cluster_trend_map(
            lon, lat, chl_clim,
            lat_sel, lon_sel, labels,
            metric_trends,
            metric=metric,
            title=f"{method.upper()} — {metric_labels[metric]} Trend (P{PERCENTILE})",
            savepath=f"figures/09_trend_map_{method}_{metric}_P{PERCENTILE}.png",
        )
        print(f"Saved: figures/09_trend_map_{method}_{metric}_P{PERCENTILE}.png")

print("\n--- Per-cluster time series plots ---")
for method, metrics_df, trends_df in [
    ("hdbscan", hdbscan_metrics, hdbscan_trends),
    ("kmeans", kmeans_metrics, kmeans_trends),
]:
    for metric in ["bg_mean", "peak_mean", "integrated_chl"]:
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        cl_df = metrics_df[metrics_df["metric"] == metric] if "metric" in metrics_df.columns else metrics_df
        unique_cls = sorted(metrics_df["cluster"].unique())

        for cl in unique_cls:
            cl_data = metrics_df[metrics_df["cluster"] == cl].sort_values("year")
            color = CLUSTER_COLORS[int(cl) % len(CLUSTER_COLORS)]
            ax.plot(
                cl_data["year"], cl_data[metric],
                "o-", color=color, markersize=2.5, linewidth=0.5,
                markerfacecolor="white", markeredgewidth=0.4,
                markeredgecolor=color, label=f"C{cl}",
            )

            cl_trend = trends_df[(trends_df["cluster"] == cl) & (trends_df["metric"] == metric)]
            if not cl_trend.empty:
                tr = cl_trend.iloc[0]
                trend_line = tr["intercept"] + tr["slope_per_decade"] / 10.0 * cl_data["year"].values
                ax.plot(cl_data["year"], trend_line, "--", color=color,
                        linewidth=0.4, alpha=0.6)

        metric_labels = {
            "bg_mean": "Background Chl-a (mg m⁻³)",
            "peak_mean": "Peak Chl-a (mg m⁻³)",
            "integrated_chl": "Integrated Peak Chl-a (mg km² m⁻³)",
        }
        ax.set_ylabel(metric_labels[metric], fontsize=6)
        ax.set_xlabel("Year", fontsize=6)
        ax.set_title(f"{method.upper()} — P{PERCENTILE}", fontsize=7)
        ax.legend(frameon=False, fontsize=5)
        ax.tick_params(labelsize=5)
        plt.tight_layout()
        fig.savefig(f"figures/09_timeseries_{method}_{metric}_P{PERCENTILE}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: figures/09_timeseries_{method}_{metric}_P{PERCENTILE}.png")

print("\n=== SUMMARY ===")
print(f"\nHDBSCAN-PCA (P{PERCENTILE}):")
for _, row in hdbscan_trends.iterrows():
    print(f"  Cluster {int(row['cluster'])} — {row['metric']}: "
          f"{row['slope_per_decade']:.3g}/decade "
          f"(MK p={row['mk_p']:.3f}, {row['stars']})")

print(f"\nK-means (P{PERCENTILE}):")
for _, row in kmeans_trends.iterrows():
    print(f"  Cluster {int(row['cluster'])} — {row['metric']}: "
          f"{row['slope_per_decade']:.3g}/decade "
          f"(MK p={row['mk_p']:.3f}, {row['stars']})")

print("\nStep 9 complete.")
