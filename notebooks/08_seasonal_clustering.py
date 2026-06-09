"""
# 08 — Seasonal Cycle Clustering (Multi-Percentile)
Cluster pixels by similarity of their climatological seasonal cycles
rather than geographic proximity. Runs for P85, P90, P95, P99 with
both HDBSCAN and K-means (k=3).
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
    compute_seasonal_cycles,
    cluster_seasonal_hdbscan,
    cluster_seasonal_kmeans,
    select_optimal_k_seasonal,
)
from src.visualization.cluster_plots import (
    plot_seasonal_cluster_map,
    plot_seasonal_cycles_by_cluster,
    plot_silhouette,
)

FILEPATH = DATA_DIR / MODISA_FILE
ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]
lat = ds.lat.values
lon = ds.lon.values

TEMPORAL_RESOLUTION = "8day"
K = 3
MIN_CLUSTER_SIZE = 200
MIN_SAMPLES = 100
SEED = 0
PERCENTILES = [85, 90, 95, 99]

print(f"Data shape: {chl.shape}")
print(f"Temporal resolution: {TEMPORAL_RESOLUTION}")
print(f"K-means k: {K}")
print(f"Percentiles: {PERCENTILES}")
print(f"HDBSCAN: min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    chl_clim = np.nanmean(chl.values, axis=0)

all_summaries = []

for pct in PERCENTILES:
    print(f"\n{'='*60}")
    print(f"   Percentile P{pct}")
    print(f"{'='*60}")

    print("\n--- Computing seasonal cycle vectors ---")
    t0 = time.time()
    cycles, lat_sel, lon_sel, periods = compute_seasonal_cycles(
        chl, lat, lon,
        percentile=pct,
        temporal_resolution=TEMPORAL_RESOLUTION,
    )
    print(f"Seasonal cycles computed: {time.time() - t0:.1f}s")
    print(f"Valid pixels: {len(lat_sel)}")
    print(f"Periods: {len(periods)}")

    if len(lat_sel) == 0:
        print("No valid pixels found. Skipping.")
        continue

    print("\n--- Silhouette analysis for optimal k ---")
    k_df = select_optimal_k_seasonal(cycles, k_range=range(2, 7), seed=SEED)
    print(k_df.to_string(index=False))
    plot_silhouette(k_df, savepath=f"figures/08_silhouette_P{pct}.png")
    print(f"Saved: figures/08_silhouette_P{pct}.png")

    print("\n--- HDBSCAN on seasonal cycles ---")
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
        title=f"HDBSCAN — Seasonal Cycles (P{pct})",
        savepath=f"figures/08_seasonal_map_hdbscan_P{pct}.png",
    )
    print(f"Saved: figures/08_seasonal_map_hdbscan_P{pct}.png")

    plot_seasonal_cycles_by_cluster(
        cycles, hdbscan_labels, periods,
        temporal_resolution=TEMPORAL_RESOLUTION,
        savepath=f"figures/08_seasonal_cycles_hdbscan_P{pct}.png",
    )
    print(f"Saved: figures/08_seasonal_cycles_hdbscan_P{pct}.png")

    print(f"\n--- K-means (k={K}) on seasonal cycles ---")
    kmeans_labels = cluster_seasonal_kmeans(cycles, k=K, seed=SEED)
    n_kmeans = len(np.unique(kmeans_labels[kmeans_labels >= 0]))
    print(f"K-means clusters: {n_kmeans}")

    plot_seasonal_cluster_map(
        lon, lat, chl_clim,
        lat_sel, lon_sel, kmeans_labels,
        title=f"K-means (k={K}) — Seasonal Cycles (P{pct})",
        savepath=f"figures/08_seasonal_map_kmeans_P{pct}.png",
    )
    print(f"Saved: figures/08_seasonal_map_kmeans_P{pct}.png")

    plot_seasonal_cycles_by_cluster(
        cycles, kmeans_labels, periods,
        temporal_resolution=TEMPORAL_RESOLUTION,
        savepath=f"figures/08_seasonal_cycles_kmeans_P{pct}.png",
    )
    print(f"Saved: figures/08_seasonal_cycles_kmeans_P{pct}.png")

    print(f"\n--- Cluster statistics P{pct} ---")
    for method, labels in [("HDBSCAN", hdbscan_labels), ("K-means", kmeans_labels)]:
        print(f"\n  {method}:")
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]
        for cl in unique_labels:
            mask = labels == cl
            cl_lat = lat_sel[mask]
            cl_lon = lon_sel[mask]
            cl_cycles = cycles[mask]

            with np.errstate(all='ignore'):
                mean_cycle = np.nanmean(cl_cycles, axis=0)
                peak_period = int(periods[np.nanargmax(mean_cycle)])
                peak_chl = float(np.nanmax(mean_cycle))

            print(f"    Cluster {cl}: n={mask.sum()}, "
                  f"lat={np.mean(cl_lat):.1f}°N (±{np.std(cl_lat):.1f}°), "
                  f"lon={np.mean(cl_lon):.1f}°W (±{np.std(cl_lon):.1f}°), "
                  f"peak at period {peak_period} ({peak_chl:.2f} mg/m³)")

            all_summaries.append({
                "percentile": pct,
                "method": method,
                "cluster": cl,
                "n_pixels": mask.sum(),
                "center_lat": round(np.mean(cl_lat), 1),
                "center_lon": round(np.mean(cl_lon), 1),
                "lat_std": round(np.std(cl_lat), 1),
                "lon_std": round(np.std(cl_lon), 1),
                "peak_period": peak_period,
                "peak_chl": round(peak_chl, 2),
            })

    print(f"\n  P{pct} done.")

summary_df = pd.DataFrame(all_summaries)
summary_path = "results/cluster_summary_seasonal.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved: {summary_path}")

print(f"\n{'='*60}")
print("   COMPLETE SUMMARY")
print(f"{'='*60}")
print(summary_df.to_string(index=False))
