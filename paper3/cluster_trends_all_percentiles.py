"""
Paper 3 — Per-Cluster Trend Analysis with Background/Peak Separation
=====================================================================
Reuses the seasonal-cycle clustering from notebook 08 (HDBSCAN and K-means)
for all percentiles (P85, P90, P95, P99). For each cluster, computes per-year
background and peak chlorophyll metrics using the percentile threshold method,
then runs Theil-Sen + Mann-Kendall trend tests.

Produces maps annotated with trend direction and significance, per-cluster
time series, and a consolidated significance table.
"""

import os
import time
import warnings
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    cluster_seasonal_hdbscan,
    cluster_seasonal_kmeans,
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

PERCENTILES = [85, 90, 95, 99]
K = 3
MIN_CLUSTER_SIZE = 100
MIN_SAMPLES = 50
SEED = 0

dlat_rad = np.deg2rad(np.abs(lat[1] - lat[0])) if len(lat) > 1 else np.deg2rad(0.04166)
dlon_rad = dlat_rad
lat_rad = np.deg2rad(lat_2d)
pixel_area = R_EARTH_KM**2 * np.cos(lat_rad) * dlat_rad * dlon_rad

years = np.unique(chl.time.dt.year.values)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    chl_clim = np.nanmean(chl.values, axis=0)

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

print(f"Data shape: {chl.shape}")
print(f"Percentiles: {PERCENTILES}")
print(f"HDBSCAN: min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}")
print(f"K-means k: {K}")
print(f"Years: {years[0]}–{years[-1]}")
print(f"Output: {OUT_DIR}")


def compute_cluster_yearly_metrics(
    chl_da, lat_sel, lon_sel, cluster_labels, years, pixel_area_2d, percentile,
    ys_idx, xs_idx,
):
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


def compute_trends(df, method_name, percentile, cluster_labels, lat_sel, lon_sel):
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
                "percentile": percentile,
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


all_metrics = []
all_trends = []

for pct in PERCENTILES:
    print(f"\n{'='*60}")
    print(f"   Percentile P{pct}")
    print(f"{'='*60}")

    print("\n--- Computing seasonal cycle vectors ---")
    t0 = time.time()
    cycles, lat_sel, lon_sel, periods, ys_idx, xs_idx = compute_seasonal_cycles(
        chl, lat, lon,
        percentile=pct,
        temporal_resolution="8day",
    )
    print(f"Seasonal cycles computed: {time.time() - t0:.1f}s")
    print(f"Valid pixels: {len(lat_sel)}")

    if len(lat_sel) == 0:
        print("No valid pixels. Skipping.")
        continue

    print(f"\n--- HDBSCAN on seasonal cycles (P{pct}) ---")
    t0 = time.time()
    hdbscan_labels, hdbscan_probs = cluster_seasonal_hdbscan(
        cycles,
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
    )
    print(f"HDBSCAN complete: {time.time() - t0:.1f}s")
    n_hdbscan = len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
    n_noise = np.sum(hdbscan_labels < 0)
    print(f"HDBSCAN clusters found: {n_hdbscan}")
    print(f"Noise pixels: {n_noise} ({100*n_noise/len(lat_sel):.1f}%)")

    plot_seasonal_cluster_map(
        lon, lat, chl_clim,
        lat_sel, lon_sel, hdbscan_labels,
        title=f"HDBSCAN-PCA — Seasonal Cycles (P{pct})",
        savepath=str(FIG_DIR / f"seasonal_map_hdbscan_P{pct}.png"),
    )
    print(f"Saved: {FIG_DIR / f'seasonal_map_hdbscan_P{pct}.png'}")

    plot_seasonal_cycles_by_cluster(
        cycles, hdbscan_labels, periods,
        temporal_resolution="8day",
        savepath=str(FIG_DIR / f"seasonal_cycles_hdbscan_P{pct}.png"),
    )
    print(f"Saved: {FIG_DIR / f'seasonal_cycles_hdbscan_P{pct}.png'}")

    print(f"\n--- K-means (k={K}) on seasonal cycles (P{pct}) ---")
    kmeans_labels = cluster_seasonal_kmeans(cycles, k=K, seed=SEED)
    n_kmeans = len(np.unique(kmeans_labels[kmeans_labels >= 0]))
    print(f"K-means clusters: {n_kmeans}")

    plot_seasonal_cluster_map(
        lon, lat, chl_clim,
        lat_sel, lon_sel, kmeans_labels,
        title=f"K-means (k={K}) — Seasonal Cycles (P{pct})",
        savepath=str(FIG_DIR / f"seasonal_map_kmeans_P{pct}.png"),
    )
    print(f"Saved: {FIG_DIR / f'seasonal_map_kmeans_P{pct}.png'}")

    plot_seasonal_cycles_by_cluster(
        cycles, kmeans_labels, periods,
        temporal_resolution="8day",
        savepath=str(FIG_DIR / f"seasonal_cycles_kmeans_P{pct}.png"),
    )
    print(f"Saved: {FIG_DIR / f'seasonal_cycles_kmeans_P{pct}.png'}")

    print(f"\n--- Per-cluster yearly metrics (HDBSCAN, P{pct}) ---")
    t0 = time.time()
    hdbscan_metrics = compute_cluster_yearly_metrics(
        chl, lat_sel, lon_sel, hdbscan_labels, years, pixel_area, pct,
        ys_idx, xs_idx,
    )
    hdbscan_metrics["percentile"] = pct
    hdbscan_metrics["method"] = "HDBSCAN"
    print(f"Done: {time.time() - t0:.1f}s, Records: {len(hdbscan_metrics)}")
    all_metrics.append(hdbscan_metrics)

    print(f"\n--- Per-cluster yearly metrics (K-means, P{pct}) ---")
    t0 = time.time()
    kmeans_metrics = compute_cluster_yearly_metrics(
        chl, lat_sel, lon_sel, kmeans_labels, years, pixel_area, pct,
        ys_idx, xs_idx,
    )
    kmeans_metrics["percentile"] = pct
    kmeans_metrics["method"] = "K-means"
    print(f"Done: {time.time() - t0:.1f}s, Records: {len(kmeans_metrics)}")
    all_metrics.append(kmeans_metrics)

    print(f"\n--- Trend analysis (HDBSCAN, P{pct}) ---")
    hdbscan_trends = compute_trends(
        hdbscan_metrics, "HDBSCAN", pct, hdbscan_labels, lat_sel, lon_sel,
    )
    all_trends.append(hdbscan_trends)
    if len(hdbscan_trends) > 0:
        print(hdbscan_trends[["cluster", "metric", "slope_per_decade", "mk_p", "stars"]].to_string(index=False))

    print(f"\n--- Trend analysis (K-means, P{pct}) ---")
    kmeans_trends = compute_trends(
        kmeans_metrics, "K-means", pct, kmeans_labels, lat_sel, lon_sel,
    )
    all_trends.append(kmeans_trends)
    if len(kmeans_trends) > 0:
        print(kmeans_trends[["cluster", "metric", "slope_per_decade", "mk_p", "stars"]].to_string(index=False))

    print(f"\n--- Generating trend maps (P{pct}) ---")
    for method, labels, trends_df in [
        ("hdbscan", hdbscan_labels, hdbscan_trends),
        ("kmeans", kmeans_labels, kmeans_trends),
    ]:
        if len(trends_df) == 0:
            continue
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
                title=f"{method.upper()} — {metric_labels[metric]} Trend (P{pct})",
                savepath=str(FIG_DIR / f"trend_map_{method}_{metric}_P{pct}.png"),
            )
            print(f"Saved: {FIG_DIR / f'trend_map_{method}_{metric}_P{pct}.png'}")

    print(f"\n--- Per-cluster time series plots (P{pct}) ---")
    for method, metrics_df, trends_df in [
        ("hdbscan", hdbscan_metrics, hdbscan_trends),
        ("kmeans", kmeans_metrics, kmeans_trends),
    ]:
        if len(metrics_df) == 0:
            continue
        for metric in ["bg_mean", "peak_mean", "integrated_chl"]:
            fig, ax = plt.subplots(figsize=(3.5, 2.0))
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
                "bg_mean": "Background Chl-a (mg m\u207b\u00b3)",
                "peak_mean": "Peak Chl-a (mg m\u207b\u00b3)",
                "integrated_chl": "Integrated Peak Chl-a (mg km\u00b2 m\u207b\u00b3)",
            }
            ax.set_ylabel(metric_labels[metric], fontsize=6)
            ax.set_xlabel("Year", fontsize=6)
            ax.set_title(f"{method.upper()} — P{pct}", fontsize=7)
            ax.legend(frameon=False, fontsize=5)
            ax.tick_params(labelsize=5)
            plt.tight_layout()
            fig.savefig(str(FIG_DIR / f"timeseries_{method}_{metric}_P{pct}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {FIG_DIR / f'timeseries_{method}_{metric}_P{pct}.png'}")

    print(f"\nP{pct} complete.")


if len(all_metrics) > 0:
    metrics_all = pd.concat(all_metrics, ignore_index=True)
    metrics_all.to_csv(str(RES_DIR / "cluster_metrics_all.csv"), index=False)
    print(f"\nSaved: {RES_DIR / 'cluster_metrics_all.csv'}")

if len(all_trends) > 0:
    trends_all = pd.concat(all_trends, ignore_index=True)
    trends_all.to_csv(str(RES_DIR / "cluster_trends_all.csv"), index=False)
    print(f"Saved: {RES_DIR / 'cluster_trends_all.csv'}")


print(f"\n{'='*60}")
print("   CONSOLIDATED SIGNIFICANCE TABLE")
print(f"{'='*60}")
if len(all_trends) > 0:
    trends_all = pd.concat(all_trends, ignore_index=True)
    display = trends_all[["percentile", "method", "cluster", "metric",
                          "slope_per_decade", "slope_pct_per_decade",
                          "mk_tau", "mk_p", "mk_trend", "stars"]].copy()
    print(display.to_string(index=False))

    sig = display[display["stars"] != "n.s."]
    if len(sig) > 0:
        print(f"\n--- Significant trends only ({len(sig)} entries) ---")
        print(sig.to_string(index=False))
    else:
        print("\nNo significant trends found.")

print(f"\n{'='*60}")
print("Paper 3 analysis complete.")
print(f"Figures: {FIG_DIR}")
print(f"Results: {RES_DIR}")
print(f"{'='*60}")
