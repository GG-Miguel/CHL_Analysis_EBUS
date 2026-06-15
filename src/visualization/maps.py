import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm, ListedColormap
from src.visualization.style import set_nature_style, save_nature, nature_fig, SINGLE_COL_WIDTH, DOUBLE_COL_WIDTH


PATCH_CMAP = ListedColormap(["#ffffff00", "#e8c848", "#c8a830", "#e88a2a", "#d62728"])


set_nature_style()

LAND_COLOR = "#e0e0e0"
COAST_COLOR = "#888888"
COAST_WIDTH = 0.3
GRID_COLOR = "#cccccc"
GRID_WIDTH = 0.2


def _prepare_grid(lon, lat, data=None):
    """
    Fill NaN in lon via linear interpolation along each row (pcolormesh
    requires finite X/Y). Returns filled lon and masked data (if given).
    """
    lon_filled = lon.copy()
    for i in range(lon.shape[0]):
        row = lon[i]
        mask = np.isnan(row)
        if mask.any():
            valid = np.where(~mask)[0]
            if len(valid) > 0:
                lon_filled[i, mask] = np.interp(np.where(mask)[0], valid, row[valid])
    if data is not None:
        data = np.ma.masked_where(np.isnan(lon), data)
    return lon_filled, data


def _add_map_basics(ax, extent=[-30, -7, 10, 45]):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, color=LAND_COLOR, zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=COAST_WIDTH, edgecolor=COAST_COLOR, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=GRID_WIDTH, color=GRID_COLOR, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 6, "color": "#666666"}
    gl.ylabel_style = {"size": 6, "color": "#666666"}
    return gl


def _add_colorbar(fig, ax, pcm):
    cb = fig.colorbar(pcm, ax=ax, shrink=0.6, pad=0.04)
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(size=2, width=0.3)
    return cb


def plot_mean_chl_map(lon, lat, chl_mean, savepath=None):
    lon_filled, chl_mean = _prepare_grid(lon, lat, chl_mean)
    fig, ax = nature_fig(width=SINGLE_COL_WIDTH, height=4.2, subplot_kw={"projection": ccrs.PlateCarree()})
    _add_map_basics(ax)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    pcm = ax.pcolormesh(
        lon_filled, lat, chl_mean,
        norm=LogNorm(vmin=0.01, vmax=10),
        cmap=cmap, shading="auto", transform=ccrs.PlateCarree(),
        linewidth=0, rasterized=True,
    )
    cb = _add_colorbar(fig, ax, pcm)
    cb.set_label("Chlorophyll-a (mg m⁻³)", fontsize=6)
    if savepath:
        save_nature(fig, savepath)
    return fig, ax


def plot_patch_overlay(lon, lat, chl_mean, masks, year, savepath=None):
    """
    masks : list of (mask_2d, hex_color, label) tuples, ordered lowest→highest threshold.
    Non-overlapping bands via integer-coded class map + ListedColormap.
    """
    lon_filled, chl_mean = _prepare_grid(lon, lat, chl_mean)
    fig, ax = nature_fig(width=SINGLE_COL_WIDTH, height=4.2, subplot_kw={"projection": ccrs.PlateCarree()})
    _add_map_basics(ax)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    ax.pcolormesh(
        lon_filled, lat, chl_mean,
        norm=LogNorm(vmin=0.01, vmax=10),
        cmap=cmap, shading="auto", transform=ccrs.PlateCarree(),
        linewidth=0, rasterized=True, alpha=0.7,
    )

    n = len(masks)
    classified = np.zeros(lon.shape, dtype=np.uint8)
    for i in range(n):
        mask = masks[i][0]
        classified[mask] = i + 1
    classified_masked = np.ma.masked_where(classified == 0, classified)

    ax.pcolormesh(
        lon_filled, lat, classified_masked,
        cmap=PATCH_CMAP, vmin=0, vmax=n,
        shading="auto", transform=ccrs.PlateCarree(),
        linewidth=0, rasterized=True,
    )

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=masks[i][1], label=masks[i][2]) for i in range(n)]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=6)
    if savepath:
        save_nature(fig, savepath)
    return fig, ax


def plot_metrics_timeseries(df, metrics, savepath=None):
    n = len(metrics)
    fig, axes = nature_fig(width=DOUBLE_COL_WIDTH, height=1.5, nrows=n, ncols=1, sharex=True)
    if n == 1:
        axes = [axes]
    labels = {
        "area_km2": "Area above P85 (km²)",
        "mean_chl": "Mean Chl-a in patch (mg m⁻³)",
        "integrated_chl": "Integrated Chl-a (mg km² m⁻³)",
        "n_patches": "No. of patches",
        "largest_patch_area": "Largest patch area (km²)",
    }
    for ax, m in zip(axes, metrics):
        ax.plot(df["year"], df[m], "o-", markersize=2.5, linewidth=0.5, color="steelblue",
                markerfacecolor="white", markeredgewidth=0.4, markeredgecolor="steelblue")
        ax.set_ylabel(labels.get(m, m), fontsize=6)
        ax.tick_params(labelsize=5)
        ax.locator_params(axis="y", nbins=4)
    axes[-1].set_xlabel("Year", fontsize=6)
    axes[-1].tick_params(labelsize=5)
    if savepath:
        save_nature(fig, savepath)
    return fig, axes


def plot_trend_bar(df_trends, metrics, savepath=None):
    fig, ax = nature_fig(width=SINGLE_COL_WIDTH, height=2.5)
    labels_dict = {
        "area_km2": "Area above P85",
        "mean_chl": "Mean Chl-a",
        "n_patches": "No. of patches",
        "largest_patch_area": "Largest patch",
        "integrated_chl": "Integrated Chl-a",
    }
    short = [m for m in metrics if m in df_trends.metric.values]
    xpos = np.arange(len(short))
    slopes = [df_trends[df_trends.metric == m].slope.values[0] for m in short]
    colors = ["#3b6992" if s > 0 else "#d55e00" for s in slopes]
    ax.bar(xpos, slopes, color=colors, width=0.5, linewidth=0.3, edgecolor="white")
    ax.axhline(0, color="#666666", linewidth=0.4)
    ax.set_xticks(xpos)
    ax.set_xticklabels([labels_dict.get(m, m) for m in short], fontsize=5, rotation=25, ha="right")
    ax.set_ylabel("Theil–Sen slope (yr⁻¹)", fontsize=6)
    ax.tick_params(labelsize=5)
    if savepath:
        save_nature(fig, savepath)
    return fig, ax
