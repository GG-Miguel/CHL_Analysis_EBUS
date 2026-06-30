"""
# 08b — HDBSCAN Parameter Search for Seasonal Cycle Clustering
Diagnostic notebook to find optimal HDBSCAN parameters for clustering
pixels by seasonal cycle similarity. Grid searches over min_cluster_size,
min_samples, PCA components, and cluster selection method. Evaluates
using DBCV (Density-Based Clustering Validation).
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
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from src.utils.config import DATA_DIR, MODISA_FILE
from src.preprocessing.ingestion import load_chlorophyll_dataset
from src.models.clustering import (
    compute_seasonal_cycles,
    cluster_seasonal_hdbscan_pca,
    cluster_seasonal_kmeans,
    search_hdbscan_parameters,
    _zscore_cycles,
)
from src.visualization.cluster_plots import (
    plot_seasonal_cluster_map,
    plot_seasonal_cycles_by_cluster,
)

FILEPATH = DATA_DIR / MODISA_FILE
ds = load_chlorophyll_dataset(FILEPATH)
chl = ds["chl"]
lat = ds.lat.values
lon = ds.lon.values

SUPP_DIR = Path("figures/Supplementary")
SUPP_DIR.mkdir(parents=True, exist_ok=True)

TEMPORAL_RESOLUTION = "8day"
K = 3
SEED = 0
PCT = 85

MIN_CLUSTER_SIZES = [20, 50, 100, 150, 200, 300, 500]
MIN_SAMPLES_LIST = [5, 10, 20, 50, 100]
PCA_COMPONENTS = [None, 10, 5, 3]
SELECTION_METHODS = ["eom", "leaf"]

print(f"Data shape: {chl.shape}")
print(f"Percentile: P{PCT}")
print(f"Temporal resolution: {TEMPORAL_RESOLUTION}")
print(f"K-means k: {K}")
print(f"\nParameter grid:")
print(f"  min_cluster_size: {MIN_CLUSTER_SIZES}")
print(f"  min_samples: {MIN_SAMPLES_LIST}")
print(f"  PCA components: {PCA_COMPONENTS}")
print(f"  selection methods: {SELECTION_METHODS}")
total_combos = (
    len(MIN_CLUSTER_SIZES) * len(MIN_SAMPLES_LIST)
    * len(PCA_COMPONENTS) * len(SELECTION_METHODS)
)
print(f"  Total combinations: {total_combos}")

print("\n--- Computing seasonal cycle vectors ---")
t0 = time.time()
cycles, lat_sel, lon_sel, periods = compute_seasonal_cycles(
    chl, lat, lon,
    percentile=PCT,
    temporal_resolution=TEMPORAL_RESOLUTION,
)
print(f"Done: {time.time() - t0:.1f}s")
print(f"Valid pixels: {len(lat_sel)}")
print(f"Periods: {len(periods)}")

if len(lat_sel) == 0:
    print("No valid pixels found. Exiting.")
    sys.exit(1)

SUBSAMPLE_SIZE = 10000
if len(lat_sel) > SUBSAMPLE_SIZE:
    print(f"\n--- Subsampling for parameter search ---")
    rng = np.random.default_rng(SEED)
    subsample_idx = rng.choice(len(lat_sel), size=SUBSAMPLE_SIZE, replace=False)
    cycles_search = cycles[subsample_idx]
    lat_sel_search = lat_sel[subsample_idx]
    lon_sel_search = lon_sel[subsample_idx]
    print(f"Subsampled to {SUBSAMPLE_SIZE} pixels for grid search")
else:
    cycles_search = cycles
    lat_sel_search = lat_sel
    lon_sel_search = lon_sel
    print(f"\nUsing all {len(lat_sel)} pixels for grid search")

print("\n--- PCA explained variance analysis ---")
cycles_z = _zscore_cycles(cycles_search)
pca_full = PCA()
pca_full.fit(cycles_z)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_90 = int(np.searchsorted(cumvar, 0.90) + 1)
n_95 = int(np.searchsorted(cumvar, 0.95) + 1)
print(f"Components for 90% variance: {n_90}")
print(f"Components for 95% variance: {n_95}")
print(f"Total components: {len(pca_full.explained_variance_ratio_)}")

fig, ax = plt.subplots(figsize=(3.5, 2.0))
ax.plot(
    np.arange(1, len(cumvar) + 1), cumvar,
    "o-", color="#3b6992", markersize=2, linewidth=0.8,
)
ax.axhline(0.90, color="#d55e00", linestyle="--", linewidth=0.5, alpha=0.7)
ax.axhline(0.95, color="#009e73", linestyle="--", linewidth=0.5, alpha=0.7)
ax.axvline(n_90, color="#d55e00", linestyle=":", linewidth=0.5, alpha=0.7)
ax.axvline(n_95, color="#009e73", linestyle=":", linewidth=0.5, alpha=0.7)
ax.set_xlabel("Number of components", fontsize=6)
ax.set_ylabel("Cumulative explained variance", fontsize=6)
ax.set_title(f"PCA: {n_90} comp → 90%, {n_95} comp → 95%", fontsize=7)
ax.tick_params(labelsize=5)
plt.tight_layout()
fig.savefig(str(SUPP_DIR / "08b_pca_variance.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {SUPP_DIR}/08b_pca_variance.png")

print("\n--- Running HDBSCAN parameter grid search ---")
t0 = time.time()
results = search_hdbscan_parameters(
    cycles_search,
    min_cluster_sizes=MIN_CLUSTER_SIZES,
    min_samples_list=MIN_SAMPLES_LIST,
    pca_components=PCA_COMPONENTS,
    selection_methods=SELECTION_METHODS,
)
print(f"Grid search complete: {time.time() - t0:.1f}s")
print(f"Combinations evaluated: {len(results)}")

print("\n--- DBCV score heatmaps ---")
for n_comp_label in PCA_COMPONENTS:
    if n_comp_label is None:
        mask = results["pca_requested"].isna()
        label = "auto"
    else:
        mask = results["pca_requested"] == n_comp_label
        label = str(n_comp_label)
    sub = results[mask]

    if len(sub) == 0:
        continue

    for csm in SELECTION_METHODS:
        sub_csm = sub[sub["cluster_selection_method"] == csm]
        if len(sub_csm) == 0:
            continue

        pivot = sub_csm.pivot_table(
            values="dbcv",
            index="min_cluster_size",
            columns="min_samples",
            aggfunc="first",
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        im = ax.imshow(
            pivot.values, cmap="RdYlGn", aspect="auto",
            vmin=-1, vmax=1,
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=5)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=5)
        ax.set_xlabel("min_samples", fontsize=6)
        ax.set_ylabel("min_cluster_size", fontsize=6)
        ax.set_title(
            f"DBCV — PCA={label}, {csm}", fontsize=7,
        )
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        fig.savefig(
            str(SUPP_DIR / f"08b_dbcv_heatmap_pca{label}_{csm}.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)
        print(
            f"Saved: {SUPP_DIR}/08b_dbcv_heatmap_pca{label}_{csm}.png"
        )

print("\n--- Noise ratio heatmaps ---")
for n_comp_label in PCA_COMPONENTS:
    if n_comp_label is None:
        mask = results["pca_requested"].isna()
        label = "auto"
    else:
        mask = results["pca_requested"] == n_comp_label
        label = str(n_comp_label)
    sub = results[mask]

    if len(sub) == 0:
        continue

    for csm in SELECTION_METHODS:
        sub_csm = sub[sub["cluster_selection_method"] == csm]
        if len(sub_csm) == 0:
            continue

        pivot = sub_csm.pivot_table(
            values="noise_ratio",
            index="min_cluster_size",
            columns="min_samples",
            aggfunc="first",
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        im = ax.imshow(
            pivot.values, cmap="YlOrRd", aspect="auto",
            vmin=0, vmax=1,
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=5)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=5)
        ax.set_xlabel("min_samples", fontsize=6)
        ax.set_ylabel("min_cluster_size", fontsize=6)
        ax.set_title(
            f"Noise ratio — PCA={label}, {csm}", fontsize=7,
        )
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        fig.savefig(
            str(SUPP_DIR / f"08b_noise_heatmap_pca{label}_{csm}.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)
        print(
            f"Saved: {SUPP_DIR}/08b_noise_heatmap_pca{label}_{csm}.png"
        )

print("\n--- Top 10 parameter combinations (by DBCV) ---")
valid_results = results[results["dbcv"].notna()].copy()
top10 = valid_results.head(10)
print(top10.to_string(index=False))

print("\n--- Current parameters (baseline) ---")
current = results[
    (results["min_cluster_size"] == 200)
    & (results["min_samples"] == 100)
    & (results["cluster_selection_method"] == "eom")
    & (results["pca_requested"].isna())
]
if len(current) > 0:
    print(current.to_string(index=False))
else:
    print("Current parameters not found in grid (may have been skipped)")

print("\n--- Running best HDBSCAN vs K-means comparison ---")
if len(top10) > 0:
    best = top10.iloc[0]
    print(f"\nBest parameters (from subsample):")
    print(f"  min_cluster_size: {int(best['min_cluster_size'])}")
    print(f"  min_samples: {int(best['min_samples'])}")
    print(f"  PCA components: {int(best['n_pca_components'])}")
    print(f"  selection method: {best['cluster_selection_method']}")
    print(f"  DBCV: {best['dbcv']:.4f}")

    print(f"\nApplying best parameters to full dataset ({len(lat_sel)} pixels)...")
    hdbscan_labels, hdbscan_probs, hdbscan_obj = cluster_seasonal_hdbscan_pca(
        cycles,
        min_cluster_size=int(best["min_cluster_size"]),
        min_samples=int(best["min_samples"]),
        n_components=int(best["n_pca_components"]),
        cluster_selection_method=best["cluster_selection_method"],
    )

    kmeans_labels = cluster_seasonal_kmeans(cycles, k=K, seed=SEED)

    hdbscan_valid = hdbscan_labels >= 0
    kmeans_valid = kmeans_labels >= 0
    both_valid = hdbscan_valid & kmeans_valid
    if both_valid.sum() > 0:
        ari = adjusted_rand_score(
            hdbscan_labels[both_valid], kmeans_labels[both_valid],
        )
    else:
        ari = np.nan

    n_hdbscan = len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
    n_kmeans = len(np.unique(kmeans_labels[kmeans_labels >= 0]))
    n_noise = np.sum(hdbscan_labels < 0)

    print(f"\nBest HDBSCAN: {n_hdbscan} clusters, "
          f"{n_noise} noise ({100*n_noise/len(lat_sel):.1f}%)")
    print(f"K-means (k={K}): {n_kmeans} clusters")
    print(f"ARI (HDBSCAN vs K-means): {ari:.4f}")

    plot_seasonal_cluster_map(
        lon, lat, np.nanmean(chl.values, axis=0),
        lat_sel, lon_sel, hdbscan_labels,
        title=(
            f"HDBSCAN-PCA — Best (P{PCT})\n"
            f"mcs={int(best['min_cluster_size'])}, "
            f"ms={int(best['min_samples'])}, "
            f"PCA={int(best['n_pca_components'])}, "
            f"{best['cluster_selection_method']}"
        ),
        savepath=f"figures/08b_best_hdbscan_map_P{PCT}.png",
    )
    print(f"Saved: figures/08b_best_hdbscan_map_P{PCT}.png")

    plot_seasonal_cycles_by_cluster(
        cycles, hdbscan_labels, periods,
        temporal_resolution=TEMPORAL_RESOLUTION,
        savepath=f"figures/08b_best_hdbscan_cycles_P{PCT}.png",
    )
    print(f"Saved: figures/08b_best_hdbscan_cycles_P{PCT}.png")

    plot_seasonal_cluster_map(
        lon, lat, np.nanmean(chl.values, axis=0),
        lat_sel, lon_sel, kmeans_labels,
        title=f"K-means (k={K}) — Seasonal Cycles (P{PCT})",
        savepath=f"figures/08b_kmeans_map_P{PCT}.png",
    )
    print(f"Saved: figures/08b_kmeans_map_P{PCT}.png")

    plot_seasonal_cycles_by_cluster(
        cycles, kmeans_labels, periods,
        temporal_resolution=TEMPORAL_RESOLUTION,
        savepath=f"figures/08b_kmeans_cycles_P{PCT}.png",
    )
    print(f"Saved: figures/08b_kmeans_cycles_P{PCT}.png")

    print("\n--- Cluster statistics comparison ---")
    for method, labels in [
        ("Best HDBSCAN", hdbscan_labels), ("K-means", kmeans_labels),
    ]:
        print(f"\n  {method}:")
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]
        for cl in unique_labels:
            mask = labels == cl
            cl_lat = lat_sel[mask]
            cl_lon = lon_sel[mask]
            cl_cycles = cycles[mask]

            with np.errstate(all="ignore"):
                mean_cycle = np.nanmean(cl_cycles, axis=0)
                peak_period = int(periods[np.nanargmax(mean_cycle)])
                peak_chl = float(np.nanmax(mean_cycle))

            print(
                f"    Cluster {cl}: n={mask.sum()}, "
                f"lat={np.mean(cl_lat):.1f}°N (±{np.std(cl_lat):.1f}°), "
                f"lon={np.mean(cl_lon):.1f}°W (±{np.std(cl_lon):.1f}°), "
                f"peak at period {peak_period} ({peak_chl:.2f} mg/m³)"
            )

print("\n--- Saving results ---")
results.to_csv("results/hdbscan_parameter_search.csv", index=False)
print("Saved: results/hdbscan_parameter_search.csv")

print(f"\n{'='*60}")
print("   SUMMARY")
print(f"{'='*60}")
if len(top10) > 0:
    print("\nTop 5 parameter sets:")
    for i, row in top10.head(5).iterrows():
        print(
            f"  {i+1}. DBCV={row['dbcv']:.3f} | "
            f"mcs={int(row['min_cluster_size'])}, "
            f"ms={int(row['min_samples'])}, "
            f"PCA={int(row['n_pca_components'])}, "
            f"{row['cluster_selection_method']} | "
            f"{int(row['n_clusters'])} clusters, "
            f"{row['noise_ratio']:.1%} noise"
        )
print(f"\nStep 08b complete.")
