"""
# 07 — Cluster Analysis: Locating Upwelling Centers
Spatial pixel clustering of high-chlorophyll regions to identify the
two major upwelling centers of the Canary Current (Cape Blanc ~21°N,
Cape Bojador ~26°N). Runs HDBSCAN and K-means per-year, then aggregates
cluster centers across years.
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

from src.utils.config import DATA_DIR, MODISA_FILE
from src.preprocessing.ingestion import load_chlorophyll_dataset
from src.models.clustering import (
    extract_high_chl_pixels,
    cluster_pixels_hdbscan,
    cluster_pixels_kmeans,
    compute_cluster_centers,
    select_optimal_k,
    run_yearly_clustering,
)
from src.visualization.cluster_plots import (
    plot_cluster_map,
    plot_cluster_centers_timeseries,
    plot_silhouette,
    plot_method_comparison,
)

FILEPATH = DATA_DIR / MODISA_FILE
ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]
lat = ds.lat.values
lon = ds.lon.values
lat_2d = lat[:, np.newaxis] * np.ones_like(lon)

PERCENTILE = 97
K = 2
MIN_CLUSTER_SIZE = 300
MIN_SAMPLES = 50
SEED = 0

print(f"Data shape: {chl.shape}")
print(f"Percentile: P{PERCENTILE}")
print(f"K-means k: {K}")
print(f"HDBSCAN: min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    chl_clim = np.nanmean(chl.values, axis=0)
lat_sel, lon_sel, chl_sel, threshold = extract_high_chl_pixels(
    chl_clim, lat, lon, PERCENTILE
)
print(f"\nClimatological high-Chl pixels (P{PERCENTILE}): {len(lat_sel)}")
print(f"Threshold: {threshold:.3f} mg/m³")

print("\n--- Silhouette analysis for optimal k ---")
k_df = select_optimal_k(lat_sel, lon_sel, k_range=range(2, 6), seed=SEED)
print(k_df.to_string(index=False))
plot_silhouette(k_df, savepath="figures/07_silhouette.png")
print("Saved: figures/07_silhouette.png")

print("\n--- HDBSCAN on climatological mean ---")
hdbscan_labels, hdbscan_probs = cluster_pixels_hdbscan(
    lat_sel, lon_sel,
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
)
n_hdbscan = len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
print(f"HDBSCAN clusters found: {n_hdbscan}")
print(f"Noise pixels: {np.sum(hdbscan_labels < 0)}")

hdbscan_centers_clim = compute_cluster_centers(hdbscan_labels, lat_sel, lon_sel, chl_sel)
print(hdbscan_centers_clim.to_string(index=False))

plot_cluster_map(
    lon, lat, chl_clim,
    lat_sel, lon_sel, hdbscan_labels,
    aggregated_centers=None,
    title=f"HDBSCAN — P{PERCENTILE} pixels (climatology)",
    savepath="figures/07_cluster_map_hdbscan.png",
)
print("Saved: figures/07_cluster_map_hdbscan.png")

print("\n--- K-means (k=2) on climatological mean ---")
kmeans_labels = cluster_pixels_kmeans(lat_sel, lon_sel, k=K, seed=SEED)
n_kmeans = len(np.unique(kmeans_labels[kmeans_labels >= 0]))
print(f"K-means clusters: {n_kmeans}")

kmeans_centers_clim = compute_cluster_centers(kmeans_labels, lat_sel, lon_sel, chl_sel)
print(kmeans_centers_clim.to_string(index=False))

plot_cluster_map(
    lon, lat, chl_clim,
    lat_sel, lon_sel, kmeans_labels,
    aggregated_centers=None,
    title=f"K-means (k={K}) — P{PERCENTILE} pixels (climatology)",
    savepath="figures/07_cluster_map_kmeans.png",
)
print("Saved: figures/07_cluster_map_kmeans.png")

print("\n--- Per-year HDBSCAN clustering ---")
t0 = time.time()
hdbscan_ref = hdbscan_centers_clim[["center_lat", "center_lon"]].values if len(hdbscan_centers_clim) > 0 else None
hdbscan_yearly, hdbscan_agg = run_yearly_clustering(
    chl, lat, lon,
    percentile=PERCENTILE,
    method="hdbscan",
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    seed=SEED,
    reference_centers=hdbscan_ref,
)
print(f"HDBSCAN yearly: {time.time() - t0:.1f}s")
print(f"Years processed: {len(hdbscan_yearly)}")
if len(hdbscan_agg) > 0:
    print("\nAggregated HDBSCAN centers:")
    print(hdbscan_agg.to_string(index=False))

print("\n--- Per-year K-means clustering ---")
t0 = time.time()
kmeans_ref = kmeans_centers_clim[["center_lat", "center_lon"]].values if len(kmeans_centers_clim) > 0 else None
kmeans_yearly, kmeans_agg = run_yearly_clustering(
    chl, lat, lon,
    percentile=PERCENTILE,
    method="kmeans",
    k=K,
    seed=SEED,
    reference_centers=kmeans_ref,
)
print(f"K-means yearly: {time.time() - t0:.1f}s")
print(f"Years processed: {len(kmeans_yearly)}")
if len(kmeans_agg) > 0:
    print("\nAggregated K-means centers:")
    print(kmeans_agg.to_string(index=False))

hdbscan_yearly.to_csv("results/cluster_centers_yearly_hdbscan.csv", index=False)
kmeans_yearly.to_csv("results/cluster_centers_yearly_kmeans.csv", index=False)
if len(hdbscan_agg) > 0:
    hdbscan_agg.to_csv("results/cluster_centers_aggregated_hdbscan.csv", index=False)
if len(kmeans_agg) > 0:
    kmeans_agg.to_csv("results/cluster_centers_aggregated_kmeans.csv", index=False)
print("\nSaved: results/cluster_centers_yearly_*.csv")
print("Saved: results/cluster_centers_aggregated_*.csv")

hdbscan_centers_all = pd.read_csv("results/cluster_centers_yearly_hdbscan.csv")
kmeans_centers_all = pd.read_csv("results/cluster_centers_yearly_kmeans.csv")

if "center_lat" in hdbscan_centers_all.columns and len(hdbscan_centers_all) > 0:
    plot_cluster_centers_timeseries(
        hdbscan_centers_all,
        savepath="figures/07_cluster_centers_ts_hdbscan.png",
    )
    print("Saved: figures/07_cluster_centers_ts_hdbscan.png")

if "center_lat" in kmeans_centers_all.columns and len(kmeans_centers_all) > 0:
    plot_cluster_centers_timeseries(
        kmeans_centers_all,
        savepath="figures/07_cluster_centers_ts_kmeans.png",
    )
    print("Saved: figures/07_cluster_centers_ts_kmeans.png")

if len(hdbscan_agg) > 0 and len(kmeans_agg) > 0:
    plot_method_comparison(
        hdbscan_agg, kmeans_agg,
        savepath="figures/07_method_comparison.png",
    )
    print("Saved: figures/07_method_comparison.png")

if len(hdbscan_agg) > 0:
    plot_cluster_map(
        lon, lat, chl_clim,
        lat_sel, lon_sel, hdbscan_labels,
        aggregated_centers=hdbscan_agg,
        title="HDBSCAN — Aggregated Upwelling Centers",
        savepath="figures/07_upwelling_centers_hdbscan.png",
    )
    print("Saved: figures/07_upwelling_centers_hdbscan.png")

if len(kmeans_agg) > 0:
    plot_cluster_map(
        lon, lat, chl_clim,
        lat_sel, lon_sel, kmeans_labels,
        aggregated_centers=kmeans_agg,
        title="K-means — Aggregated Upwelling Centers",
        savepath="figures/07_upwelling_centers_kmeans.png",
    )
    print("Saved: figures/07_upwelling_centers_kmeans.png")

print("\n=== SUMMARY ===")
if len(kmeans_agg) > 0:
    print(f"K-means (k={K}) aggregated upwelling centers:")
    for _, row in kmeans_agg.iterrows():
        print(f"  Cluster {int(row['cluster'])}: "
              f"{row['mean_center_lat']:.1f}°N, "
              f"{row['mean_center_lon']:.1f}°W "
              f"(±{row['std_center_lat']:.1f}° lat, "
              f"±{row['std_center_lon']:.1f}° lon) "
              f"over {int(row['n_years'])} years")

if len(hdbscan_agg) > 0:
    print(f"\nHDBSCAN aggregated upwelling centers:")
    for _, row in hdbscan_agg.iterrows():
        print(f"  Cluster {int(row['cluster'])}: "
              f"{row['mean_center_lat']:.1f}°N, "
              f"{row['mean_center_lon']:.1f}°W "
              f"(±{row['std_center_lat']:.1f}° lat, "
              f"±{row['std_center_lon']:.1f}° lon) "
              f"over {int(row['n_years'])} years")

print("\nStep 7 complete.")
