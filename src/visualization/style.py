import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

NATURE_COLORS = [
    "#3b6992", "#d55e00", "#cdb241", "#009e73",
    "#cc79a7", "#56b4e9", "#f0e442", "#0072b2",
]

SINGLE_COL_WIDTH = 3.5
ONE_POINT_FIVE_COL_WIDTH = 4.75
DOUBLE_COL_WIDTH = 7.2

def set_nature_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.title_fontsize": 6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "lines.linewidth": 0.6,
            "lines.markersize": 3,
            "patch.linewidth": 0.3,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.grid": False,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.width": 0.3,
            "ytick.minor.width": 0.3,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": False,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.4,
            "axes.prop_cycle": cycler(color=NATURE_COLORS),
            "image.cmap": "viridis",
        }
    )


def nature_fig(width=SINGLE_COL_WIDTH, height=None, nrows=1, ncols=1, **kwargs):
    if height is None:
        height = width * 0.65
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width * ncols, height * nrows),
        **kwargs,
    )
    return fig, axes


def save_nature(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
