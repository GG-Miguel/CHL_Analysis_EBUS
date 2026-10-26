"""
Paper 3 — Per-Cluster Trend Analysis with Background/Peak Separation
=====================================================================
Robustness version: clusters are computed once at a reference percentile
(P85 by default) and reused for all higher percentiles (P85, P90, P95,
P99). This fixes the cluster ID instability that occurs when each
percentile is clustered independently.

For each fixed cluster, per-year background and peak chlorophyll metrics
are computed using the percentile threshold method, then Theil-Sen +
Mann-Kendall trend tests are run. P-values are corrected for multiple
comparisons using Benjamini-Hochberg FDR across all cluster × percentile
× method × metric tests.

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
    cluster_seasonal_hdbscan_pca,
    cluster_seasonal_kmeans,
)
from src.statistics.significance import (
    theil_sen_with_ci, mann_kendall, p_value_to_stars, fdr_bh,
)
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
REFERENCE_PERCENTILE = 85
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

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    chl_clim = np.nanmean(chl.values, axis=0)

OUT_DIR = Path(__file__).resolve().parent
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

print(f"Data shape: {chl.shape}")
print(f"Reference percentile: P{REFERENCE_PERCENTILE}")
print(f"Analysis percentiles: {PERCENTILES}")
print(f"HDBSCAN-PCA: min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES}, "
      f"n_components={N_PCA_COMPONENTS}")
print(f"K-means k: {K}")
print(f"Years: {years[0]}–{years[-1]}")
print(f"Output: {OUT_DIR}")


# ---------------------------------------------------------------------------
# Reference cluster computation (fixed across percentiles)
# ---------------------------------------------------------------------------
def compute_reference_clusters(
    chl_da, lat, lon, percentile, k, min_cluster_size, min_samples, n_components, seed
):
    """Compute HDBSCAN-PCA and K-means seasonal-cycle clusters at the reference percentile."""
    print(f"\n--- Computing reference clusters at P{percentile} ---")
    t0 = time.time()
    cycles, lat_sel, lon_sel, periods, ys_idx, xs_idx = compute_seasonal_cycles(
        chl_da, lat, lon,
        percentile=percentile,
        temporal_resolution="8day",
    )
    print(f"Seasonal cycles computed: {time.time() - t0:.1f}s")
    print(f"Valid pixels: {len(lat_sel)}")

    if len(lat_sel) == 0:
        raise ValueError("No valid pixels at reference percentile.")

    print(f"\n--- HDBSCAN-PCA on seasonal cycles (reference P{percentile}) ---")
    t0 = time.time()
    hdbscan_labels, hdbscan_probs, _, _ = cluster_seasonal_hdbscan_pca(
        cycles,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        n_components=n_components,
        cluster_selection_method="leaf",
    )
    print(f"HDBSCAN-PCA complete: {time.time() - t0:.1f}s")
    n_hdbscan = len(np.unique(hdbscan_labels[hdbscan_labels >= 0]))
    n_noise = np.sum(hdbscan_labels < 0)
    print(f"HDBSCAN clusters found: {n_hdbscan}")
    print(f"Noise pixels: {n_noise} ({100*n_noise/len(lat_sel):.1f}%)")

    print(f"\n--- K-means (k={k}) on seasonal cycles (reference P{percentile}) ---")
    kmeans_labels = cluster_seasonal_kmeans(cycles, k=k, seed=seed)
    n_kmeans = len(np.unique(kmeans_labels[kmeans_labels >= 0]))
    print(f"K-means clusters: {n_kmeans}")

    ref = {
        "cycles": cycles,
        "lat_sel": lat_sel,
        "lon_sel": lon_sel,
        "periods": periods,
        "ys_idx": ys_idx,
        "xs_idx": xs_idx,
        "hdbscan_labels": hdbscan_labels,
        "kmeans_labels": kmeans_labels,
    }
    return ref


def map_to_reference_labels(ys_idx, xs_idx, ref_ys_idx, ref_xs_idx, ref_labels):
    """Map each (y, x) in the current selection to its reference cluster label."""
    ref_map = {
        (int(y), int(x)): int(lab)
        for y, x, lab in zip(ref_ys_idx, ref_xs_idx, ref_labels)
    }
    mapped = np.array(
        [ref_map.get((int(y), int(x)), -1) for y, x in zip(ys_idx, xs_idx)],
        dtype=int,
    )
    return mapped


def compute_cluster_centers(lat_sel, lon_sel, labels):
    """Return a dict cluster -> (center_lat, center_lon) from fixed labels."""
    centers = {}
    for cl in np.unique(labels):
        if cl < 0:
            continue
        mask = labels == cl
        centers[int(cl)] = (
            float(np.mean(lat_sel[mask])),
            float(np.mean(lon_sel[mask])),
        )
    return centers


# Compute reference clusters once
ref = compute_reference_clusters(
    chl, lat, lon,
    percentile=REFERENCE_PERCENTILE,
    k=K,
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    n_components=N_PCA_COMPONENTS,
    seed=SEED,
)

hdbscan_centers = compute_cluster_centers(ref["lat_sel"], ref["lon_sel"], ref["hdbscan_labels"])
kmeans_centers = compute_cluster_centers(ref["lat_sel"], ref["lon_sel"], ref["kmeans_labels"])

# Save reference cluster maps (once, not per percentile)
plot_seasonal_cluster_map(
    lon, lat, chl_clim,
    ref["lat_sel"], ref["lon_sel"], ref["hdbscan_labels"],
    title=f"HDBSCAN-PCA — Seasonal Cycles (reference P{REFERENCE_PERCENTILE})",
    savepath=str(FIG_DIR / f"seasonal_map_hdbscan_ref_P{REFERENCE_PERCENTILE}.png"),
)
print(f"Saved: {FIG_DIR / f'seasonal_map_hdbscan_ref_P{REFERENCE_PERCENTILE}.png'}")

plot_seasonal_cycles_by_cluster(
    ref["cycles"], ref["hdbscan_labels"], ref["periods"],
    temporal_resolution="8day",
    savepath=str(FIG_DIR / f"seasonal_cycles_hdbscan_ref_P{REFERENCE_PERCENTILE}.png"),
)
print(f"Saved: {FIG_DIR / f'seasonal_cycles_hdbscan_ref_P{REFERENCE_PERCENTILE}.png'}")

plot_seasonal_cluster_map(
    lon, lat, chl_clim,
    ref["lat_sel"], ref["lon_sel"], ref["kmeans_labels"],
    title=f"K-means (k={K}) — Seasonal Cycles (reference P{REFERENCE_PERCENTILE})",
    savepath=str(FIG_DIR / f"seasonal_map_kmeans_ref_P{REFERENCE_PERCENTILE}.png"),
)
print(f"Saved: {FIG_DIR / f'seasonal_map_kmeans_ref_P{REFERENCE_PERCENTILE}.png'}")

plot_seasonal_cycles_by_cluster(
    ref["cycles"], ref["kmeans_labels"], ref["periods"],
    temporal_resolution="8day",
    savepath=str(FIG_DIR / f"seasonal_cycles_kmeans_ref_P{REFERENCE_PERCENTILE}.png"),
)
print(f"Saved: {FIG_DIR / f'seasonal_cycles_kmeans_ref_P{REFERENCE_PERCENTILE}.png'}")


# ---------------------------------------------------------------------------
# Per-cluster yearly metrics and trends
# ---------------------------------------------------------------------------
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


def compute_trends(df, method_name, percentile, cluster_centers):
    unique_clusters = sorted(df["cluster"].unique())
    trend_records = []

    for cl in unique_clusters:
        cl_df = df[df["cluster"] == cl].sort_values("year")
        if len(cl_df) < 4:
            continue

        center_lat, center_lon = cluster_centers.get(int(cl), (np.nan, np.nan))

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
                "center_lat": float(center_lat),
                "center_lon": float(center_lon),
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

    # Map selected pixels to the fixed reference labels
    hdbscan_mapped = map_to_reference_labels(
        ys_idx, xs_idx, ref["ys_idx"], ref["xs_idx"], ref["hdbscan_labels"]
    )
    kmeans_mapped = map_to_reference_labels(
        ys_idx, xs_idx, ref["ys_idx"], ref["xs_idx"], ref["kmeans_labels"]
    )

    # Filter out noise / unassigned pixels (should only affect pct < reference)
    hdbscan_valid = hdbscan_mapped >= 0
    kmeans_valid = kmeans_mapped >= 0
    print(f"Pixels mapped to HDBSCAN reference clusters: {hdbscan_valid.sum()}/{len(hdbscan_mapped)}")
    print(f"Pixels mapped to K-means reference clusters: {kmeans_valid.sum()}/{len(kmeans_mapped)}")

    if hdbscan_valid.sum() > 0:
        hdbscan_cycles_f = cycles[hdbscan_valid]
        hdbscan_lat_f = lat_sel[hdbscan_valid]
        hdbscan_lon_f = lon_sel[hdbscan_valid]
        hdbscan_ys_f = ys_idx[hdbscan_valid]
        hdbscan_xs_f = xs_idx[hdbscan_valid]
        hdbscan_labels_f = hdbscan_mapped[hdbscan_valid]

        plot_seasonal_cluster_map(
            lon, lat, chl_clim,
            hdbscan_lat_f, hdbscan_lon_f, hdbscan_labels_f,
            title=f"HDBSCAN-PCA — P{pct} pixels in reference clusters",
            savepath=str(FIG_DIR / f"seasonal_map_hdbscan_P{pct}.png"),
        )
        print(f"Saved: {FIG_DIR / f'seasonal_map_hdbscan_P{pct}.png'}")

        plot_seasonal_cycles_by_cluster(
            hdbscan_cycles_f, hdbscan_labels_f, periods,
            temporal_resolution="8day",
            savepath=str(FIG_DIR / f"seasonal_cycles_hdbscan_P{pct}.png"),
        )
        print(f"Saved: {FIG_DIR / f'seasonal_cycles_hdbscan_P{pct}.png'}")

    if kmeans_valid.sum() > 0:
        kmeans_cycles_f = cycles[kmeans_valid]
        kmeans_lat_f = lat_sel[kmeans_valid]
        kmeans_lon_f = lon_sel[kmeans_valid]
        kmeans_ys_f = ys_idx[kmeans_valid]
        kmeans_xs_f = xs_idx[kmeans_valid]
        kmeans_labels_f = kmeans_mapped[kmeans_valid]

        plot_seasonal_cluster_map(
            lon, lat, chl_clim,
            kmeans_lat_f, kmeans_lon_f, kmeans_labels_f,
            title=f"K-means (k={K}) — P{pct} pixels in reference clusters",
            savepath=str(FIG_DIR / f"seasonal_map_kmeans_P{pct}.png"),
        )
        print(f"Saved: {FIG_DIR / f'seasonal_map_kmeans_P{pct}.png'}")

        plot_seasonal_cycles_by_cluster(
            kmeans_cycles_f, kmeans_labels_f, periods,
            temporal_resolution="8day",
            savepath=str(FIG_DIR / f"seasonal_cycles_kmeans_P{pct}.png"),
        )
        print(f"Saved: {FIG_DIR / f'seasonal_cycles_kmeans_P{pct}.png'}")

    print(f"\n--- Per-cluster yearly metrics (HDBSCAN, P{pct}) ---")
    if hdbscan_valid.sum() > 0:
        t0 = time.time()
        hdbscan_metrics = compute_cluster_yearly_metrics(
            chl, hdbscan_lat_f, hdbscan_lon_f, hdbscan_labels_f, years, pixel_area, pct,
            hdbscan_ys_f, hdbscan_xs_f,
        )
        hdbscan_metrics["percentile"] = pct
        hdbscan_metrics["method"] = "HDBSCAN"
        print(f"Done: {time.time() - t0:.1f}s, Records: {len(hdbscan_metrics)}")
        all_metrics.append(hdbscan_metrics)
    else:
        hdbscan_metrics = pd.DataFrame()

    print(f"\n--- Per-cluster yearly metrics (K-means, P{pct}) ---")
    if kmeans_valid.sum() > 0:
        t0 = time.time()
        kmeans_metrics = compute_cluster_yearly_metrics(
            chl, kmeans_lat_f, kmeans_lon_f, kmeans_labels_f, years, pixel_area, pct,
            kmeans_ys_f, kmeans_xs_f,
        )
        kmeans_metrics["percentile"] = pct
        kmeans_metrics["method"] = "K-means"
        print(f"Done: {time.time() - t0:.1f}s, Records: {len(kmeans_metrics)}")
        all_metrics.append(kmeans_metrics)
    else:
        kmeans_metrics = pd.DataFrame()

    print(f"\n--- Trend analysis (HDBSCAN, P{pct}) ---")
    if len(hdbscan_metrics) > 0:
        hdbscan_trends = compute_trends(
            hdbscan_metrics, "HDBSCAN", pct, hdbscan_centers,
        )
        all_trends.append(hdbscan_trends)
        if len(hdbscan_trends) > 0:
            print(hdbscan_trends[["cluster", "metric", "slope_per_decade", "mk_p", "stars"]].to_string(index=False))
    else:
        hdbscan_trends = pd.DataFrame()

    print(f"\n--- Trend analysis (K-means, P{pct}) ---")
    if len(kmeans_metrics) > 0:
        kmeans_trends = compute_trends(
            kmeans_metrics, "K-means", pct, kmeans_centers,
        )
        all_trends.append(kmeans_trends)
        if len(kmeans_trends) > 0:
            print(kmeans_trends[["cluster", "metric", "slope_per_decade", "mk_p", "stars"]].to_string(index=False))
    else:
        kmeans_trends = pd.DataFrame()

    print(f"\n--- Generating trend maps (P{pct}) ---")
    for method, labels, lat_f, lon_f, trends_df in [
        ("hdbscan", hdbscan_labels_f if hdbscan_valid.sum() > 0 else np.array([]),
         hdbscan_lat_f if hdbscan_valid.sum() > 0 else np.array([]),
         hdbscan_lon_f if hdbscan_valid.sum() > 0 else np.array([]),
         hdbscan_trends),
        ("kmeans", kmeans_labels_f if kmeans_valid.sum() > 0 else np.array([]),
         kmeans_lat_f if kmeans_valid.sum() > 0 else np.array([]),
         kmeans_lon_f if kmeans_valid.sum() > 0 else np.array([]),
         kmeans_trends),
    ]:
        if len(labels) == 0 or len(trends_df) == 0:
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
                lat_f, lon_f, labels,
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
        if len(metrics_df) == 0 or len(trends_df) == 0:
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
    plt.close("all")


# ---------------------------------------------------------------------------
# Consolidate outputs and apply FDR correction
# ---------------------------------------------------------------------------
if len(all_metrics) > 0:
    metrics_all = pd.concat(all_metrics, ignore_index=True)
    metrics_all.to_csv(str(RES_DIR / "cluster_metrics_all.csv"), index=False)
    print(f"\nSaved: {RES_DIR / 'cluster_metrics_all.csv'}")

if len(all_trends) > 0:
    trends_all = pd.concat(all_trends, ignore_index=True)

    # FDR correction across all valid p-values
    trends_all["mk_q"] = fdr_bh(trends_all["mk_p"].values)
    trends_all["fdr_stars"] = trends_all["mk_q"].apply(p_value_to_stars)

    trends_all.to_csv(str(RES_DIR / "cluster_trends_all.csv"), index=False)
    print(f"Saved: {RES_DIR / 'cluster_trends_all.csv'}")


print(f"\n{'='*60}")
print("   CONSOLIDATED SIGNIFICANCE TABLE")
print(f"{'='*60}")
if len(all_trends) > 0:
    display = trends_all[[
        "percentile", "method", "cluster", "metric",
        "slope_per_decade", "slope_pct_per_decade",
        "mk_tau", "mk_p", "mk_q", "mk_trend", "stars", "fdr_stars",
    ]].copy()
    print(display.to_string(index=False))

    sig_uncorrected = display[display["stars"] != "n.s."]
    sig_fdr = display[display["fdr_stars"] != "n.s."]

    print(f"\n--- Uncorrected significant trends ({len(sig_uncorrected)} entries) ---")
    if len(sig_uncorrected) > 0:
        print(sig_uncorrected.to_string(index=False))
    else:
        print("None.")

    print(f"\n--- FDR-corrected significant trends ({len(sig_fdr)} entries) ---")
    if len(sig_fdr) > 0:
        print(sig_fdr.to_string(index=False))
    else:
        print("None.")

print(f"\n{'='*60}")
print("Paper 3 analysis complete.")
print(f"Figures: {FIG_DIR}")
print(f"Results: {RES_DIR}")
print(f"{'='*60}")
