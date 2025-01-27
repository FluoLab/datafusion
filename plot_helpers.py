from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.optimize import curve_fit

from utils import mono_exponential_decay_numpy as decay

def add_z_text(ax, z_index, pos=(0.95, 0.95), font_size=12):
    ax.text(
        pos[0], pos[1], f"z={z_index} µm",
        transform=ax.transAxes,
        fontsize=font_size,
        fontweight="bold",
        va="top",
        ha="right",
        c="w"
    )


def add_scalebar(ax, pixel_size, length_micrometers=30, font_size=12, size_vertical=1):
    scalebar_length_pixels = length_micrometers / pixel_size
    scalebar = AnchoredSizeBar(
        ax.transData,
        scalebar_length_pixels,
        f"{length_micrometers} µm",
        "upper left",
        pad=0.3,
        color="white",
        frameon=False,
        size_vertical=size_vertical,
        fontproperties={"size": font_size, "weight": "bold"},
    )
    ax.add_artist(scalebar)


def add_letter(ax, letter, pos=(0.05, 0.05), font_size=16, color="w"):
    ax.text(
        pos[0], pos[1], f"({letter})",
        transform=ax.transAxes,
        fontsize=font_size,
        fontweight="bold",
        va="bottom",
        c=color
    )


def exp_fit(x_data, y_data):
    return curve_fit(
        decay, x_data, y_data,
        bounds=([0.0, 1e-6, -0.1], [1, 6.0, 0.1]),
        p0=(0.5, 2.0, 0.000001),
        maxfev=5000,
    )[0]
