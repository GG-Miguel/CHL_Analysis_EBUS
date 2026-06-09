import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm

from src.visualization.style import (
    set_nature_style,
    save_nature,
    nature_fig,
    SINGLE_COL_WIDTH,
    DOUBLE_COL_WIDTH,
)
from src.visualization.maps import _prepare_grid, _add_map_basics, _add_colorbar

CLUSTER_COLORS = [
    "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def plot_cluster_map(
    lon: np.ndarray,
    lat: np.ndarray,
    chl_mean: np.ndarray,
    cluster_lat: np.ndarray,
    cluster_lon: np.ndarray,
    cluster_labels: np.ndarray,
    aggregated_centers=None,
    title: str = "",
    savepath: str | None = None,
    extent: list | None = None,
):
    if lat.ndim == 1 and lon.ndim == 2:
        lat_2d = lat[:, np.newaxis] * np.ones_like(lon)
    elif lat.ndim == 1 and lon.ndim == 1:
        lat_2d, _ = np.meshgrid(lat, lon, indexing="ij")
    else:
        lat_2d = lat

    lon_filled, chl_mean = _prepare_grid(lon, lat_2d, chl_mean)

    fig, ax = nature_fig(
        width=SINGLE_COL_WIDTH, height=5.5,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _add_map_basics(ax, extent=extent or [-34, -4, 8, 46])

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    pcm = ax.pcolormesh(
        lon_filled, lat_2d, chl_mean,
        norm=LogNorm(vmin=0.01, vmax=10),
        cmap=cmap, shading="auto", transform=ccrs.PlateCarree(),
        linewidth=0, rasterized=True, alpha=0.6,
    )
    _add_colorbar(fig, ax, pcm).set_label("Chl-a (mg m⁻³)", fontsize=6)

    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels >= 0]

    for cl in unique_labels:
        mask = cluster_labels == cl
        color = CLUSTER_COLORS[int(cl) % len(CLUSTER_COLORS)]
        ax.scatter(
            cluster_lon[mask], cluster_lat[mask],
            c=color, s=0.3, alpha=0.15,
            transform=ccrs.PlateCarree(), rasterized=True,
        )

    if aggregated_centers is not None and len(aggregated_centers) > 0:
        for _, row in aggregated_centers.iterrows():
            cl = int(row["cluster"])
            color = CLUSTER_COLORS[cl % len(CLUSTER_COLORS)]
            ax.plot(
                row["mean_center_lon"], row["mean_center_lat"],
                "X", color=color, markersize=8, markeredgewidth=0.8,
                markeredgecolor="white",
                transform=ccrs.PlateCarree(), zorder=5,
            )
            ax.annotate(
                f"{row['mean_center_lat']:.1f}°N",
                xy=(row["mean_center_lon"], row["mean_center_lat"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=5, color=color, fontweight="bold",
                transform=ccrs.PlateCarree(),
            )

    if title:
        ax.set_title(title, fontsize=7, pad=4)

    if savepath:
        save_nature(fig, savepath)
    return fig, ax


def plot_cluster_centers_timeseries(
    centers_yearly: "pd.DataFrame",
    savepath: str | None = None,
):
    import pandas as pd

    clusters = sorted(centers_yearly["cluster"].unique())
    n = len(clusters)
    if n == 0:
        return None, None

    fig, axes = nature_fig(width=SINGLE_COL_WIDTH, height=1.5, nrows=2, ncols=1, sharex=True)

    for cl in clusters:
        cl_data = centers_yearly[centers_yearly["cluster"] == cl].sort_values("year")
        color = CLUSTER_COLORS[int(cl) % len(CLUSTER_COLORS)]
        axes[0].plot(
            cl_data["year"], cl_data["center_lat"],
            "o-", color=color, markersize=2.5, linewidth=0.5,
            markerfacecolor="white", markeredgewidth=0.4,
            markeredgecolor=color, label=f"Cluster {cl}",
        )
        axes[1].plot(
            cl_data["year"], cl_data["center_lon"],
            "o-", color=color, markersize=2.5, linewidth=0.5,
            markerfacecolor="white", markeredgewidth=0.4,
            markeredgecolor=color,
        )

    axes[0].set_ylabel("Center latitude (°N)", fontsize=6)
    axes[1].set_ylabel("Center longitude (°W)", fontsize=6)
    axes[1].set_xlabel("Year", fontsize=6)
    axes[0].legend(frameon=False, fontsize=5)
    for ax in axes:
        ax.tick_params(labelsize=5)
        ax.locator_params(axis="y", nbins=4)

    fig.suptitle("Upwelling Center Positions Over Time", fontsize=7)
    plt.tight_layout()
    if savepath:
        save_nature(fig, savepath)
    return fig, axes


def plot_silhouette(
    k_df: "pd.DataFrame",
    savepath: str | None = None,
):
    fig, ax = nature_fig(width=SINGLE_COL_WIDTH, height=2.0)
    ax.plot(k_df["k"], k_df["silhouette"], "o-", color="#3b6992",
            markersize=4, linewidth=0.8)
    best_k = k_df.loc[k_df["silhouette"].idxmax(), "k"]
    ax.axvline(best_k, color="#d55e00", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Number of clusters (k)", fontsize=6)
    ax.set_ylabel("Silhouette score", fontsize=6)
    ax.set_title(f"Optimal k = {int(best_k)}", fontsize=7)
    ax.tick_params(labelsize=5)
    ax.set_xticks(k_df["k"].values)
    plt.tight_layout()
    if savepath:
        save_nature(fig, savepath)
    return fig, ax


def plot_method_comparison(
    hdbscan_agg: "pd.DataFrame",
    kmeans_agg: "pd.DataFrame",
    savepath: str | None = None,
):
    fig, axes = nature_fig(width=DOUBLE_COL_WIDTH, height=2.0, nrows=1, ncols=2)

    for ax, df, title in zip(
        axes,
        [hdbscan_agg, kmeans_agg],
        ["HDBSCAN", "K-means (k=2)"],
    ):
        if len(df) == 0:
            ax.text(0.5, 0.5, "No clusters found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=6)
            ax.set_title(title, fontsize=7)
            continue

        for _, row in df.iterrows():
            cl = int(row["cluster"])
            color = CLUSTER_COLORS[cl % len(CLUSTER_COLORS)]
            ax.errorbar(
                row["mean_center_lon"], row["mean_center_lat"],
                xerr=row["std_center_lon"], yerr=row["std_center_lat"],
                fmt="X", color=color, markersize=10,
                markeredgewidth=0.8, markeredgecolor="white",
                capsize=3, capthick=0.5, elinewidth=0.5,
            )
            ax.annotate(
                f"C{cl}: {row['mean_center_lat']:.1f}°N\n"
                f"σ={row['std_center_lat']:.1f}°",
                xy=(row["mean_center_lon"], row["mean_center_lat"]),
                xytext=(8, -8), textcoords="offset points",
                fontsize=5, color=color,
            )

        ax.set_xlabel("Longitude (°W)", fontsize=6)
        ax.set_ylabel("Latitude (°N)", fontsize=6)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=5)
        ax.invert_xaxis()

    plt.tight_layout()
    if savepath:
        save_nature(fig, savepath)
    return fig, axes


def plot_seasonal_cluster_map(
    lon: np.ndarray,
    lat: np.ndarray,
    chl_mean: np.ndarray,
    cluster_lat: np.ndarray,
    cluster_lon: np.ndarray,
    cluster_labels: np.ndarray,
    title: str = "",
    savepath: str | None = None,
    extent: list | None = None,
):
    """Plot map colored by seasonal cycle cluster membership."""
    if lat.ndim == 1 and lon.ndim == 2:
        lat_2d = lat[:, np.newaxis] * np.ones_like(lon)
    elif lat.ndim == 1 and lon.ndim == 1:
        lat_2d, _ = np.meshgrid(lat, lon, indexing="ij")
    else:
        lat_2d = lat

    lon_filled, chl_mean = _prepare_grid(lon, lat_2d, chl_mean)

    fig, ax = nature_fig(
        width=SINGLE_COL_WIDTH, height=5.5,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _add_map_basics(ax, extent=extent or [-34, -4, 8, 46])

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    ax.pcolormesh(
        lon_filled, lat_2d, chl_mean,
        norm=LogNorm(vmin=0.01, vmax=10),
        cmap=cmap, shading="auto", transform=ccrs.PlateCarree(),
        linewidth=0, rasterized=True, alpha=0.4,
    )

    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels >= 0]

    for cl in unique_labels:
        mask = cluster_labels == cl
        color = CLUSTER_COLORS[int(cl) % len(CLUSTER_COLORS)]
        ax.scatter(
            cluster_lon[mask], cluster_lat[mask],
            c=color, s=0.5, alpha=0.3,
            transform=ccrs.PlateCarree(), rasterized=True,
        )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=CLUSTER_COLORS[int(cl) % len(CLUSTER_COLORS)],
              label=f"Cluster {cl}")
        for cl in unique_labels
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=6)

    if title:
        ax.set_title(title, fontsize=7, pad=4)

    if savepath:
        save_nature(fig, savepath)
    return fig, ax


def plot_seasonal_cycles_by_cluster(
    cycles: np.ndarray,
    labels: np.ndarray,
    periods: np.ndarray,
    temporal_resolution: str = "8day",
    savepath: str | None = None,
):
    """Plot mean seasonal cycle for each cluster with shaded std."""
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    if len(unique_labels) == 0:
        return None, None

    fig, ax = nature_fig(width=SINGLE_COL_WIDTH, height=2.5)

    x_labels = None
    if temporal_resolution == "monthly":
        x_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        x_ticks = periods
    else:
        x_ticks = np.arange(0, len(periods), 4)
        x_labels = [f"{int(p*8)}" for p in periods[x_ticks]]

    for cl in unique_labels:
        mask = labels == cl
        cl_cycles = cycles[mask]
        color = CLUSTER_COLORS[int(cl) % len(CLUSTER_COLORS)]

        with np.errstate(all='ignore'):
            mean_cycle = np.nanmean(cl_cycles, axis=0)
            std_cycle = np.nanstd(cl_cycles, axis=0)

        ax.plot(periods, mean_cycle, color=color, linewidth=1.0,
                label=f"Cluster {cl} (n={mask.sum()})")
        ax.fill_between(periods, mean_cycle - std_cycle, mean_cycle + std_cycle,
                        color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Day of year" if temporal_resolution == "8day" else "Month",
                  fontsize=6)
    ax.set_ylabel("Chl-a (mg m⁻³)", fontsize=6)
    ax.set_title("Seasonal Cycles by Cluster", fontsize=7)
    ax.legend(frameon=False, fontsize=5)
    ax.tick_params(labelsize=5)

    if x_labels is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=5)

    plt.tight_layout()
    if savepath:
        save_nature(fig, savepath)
    return fig, ax
