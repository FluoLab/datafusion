import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from utils import R, G, B, WAVELENGTHS, spectral_volume_to_srgb, time_image_to_lifetime, FIGURES_PATH


def plot_zoom_results(
        x, t, dt, lam,
        ground_truth_spc,
        z_index,
        zoom_slice=slice(48, 80),
        x_continuous=True,
        save_name=None,
):
    spectral_colors = [(r, g, b) for r, g, b in zip(R, G, B)]
    spectral_cmap = LinearSegmentedColormap.from_list("spectrum", spectral_colors, N=100)

    fused_intensity = x.mean(axis=(0, 1))[z_index]
    fused_zoomed_spectral = spectral_volume_to_srgb(
        lam, x.mean(axis=0)[:, z_index:z_index + 1, zoom_slice, zoom_slice])[0]

    if x_continuous:
        fused_zoomed_tau = x.mean(axis=1)[1, 5, zoom_slice, zoom_slice]
    else:
        time_img = x.mean(axis=1)[:, z_index, zoom_slice, zoom_slice]
        fused_zoomed_tau = time_image_to_lifetime(t, dt, time_img)

    ground_truth_spc_spectral = spectral_volume_to_srgb(lam, ground_truth_spc.mean(axis=0)[:, np.newaxis])[0]
    time_img = ground_truth_spc.mean(axis=1)
    time_img = np.where(time_img < 0, 0, time_img)
    ground_truth_spc_tau = time_image_to_lifetime(t, dt, time_img)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.7, 1, 1, 1], height_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[:, 0])
    fused_0 = ax0.imshow(fused_intensity, cmap='gray')
    ax0.set_title(f"Fused Intensity Image ({fused_intensity.shape[-1]}x{fused_intensity.shape[-1]})")
    ax0.add_patch(patches.Rectangle(
        (zoom_slice.start, zoom_slice.start),
        zoom_slice.stop - zoom_slice.start,
        zoom_slice.stop - zoom_slice.start,
        linewidth=2, edgecolor='red', facecolor='none')
    )
    ax0.axis("off")

    # Spectral comparison
    ax01 = fig.add_subplot(gs[0, 1])
    zoom01 = ax01.imshow(fused_zoomed_spectral, cmap=spectral_cmap)
    ax01.set_title(f"Fused Spectrum {fused_zoomed_spectral.shape[0]}x{fused_zoomed_spectral.shape[0]}")
    ax01.axis("off")

    ax02 = fig.add_subplot(gs[0, 2])
    zoom02 = ax02.imshow(ground_truth_spc_spectral, cmap=spectral_cmap)
    ax02.set_title(f"GT Spectrum {ground_truth_spc_spectral.shape[0]}x{ground_truth_spc_spectral.shape[0]}")
    ax02.axis("off")

    ax03 = fig.add_subplot(gs[0, 3])
    tmp = x.mean(axis=(0, 3, 4))[:, z_index]
    ax03.plot(lam, tmp / tmp.max(), label="Fused")
    tmp = ground_truth_spc.mean(axis=(0, 2, 3))
    ax03.plot(lam, tmp / tmp.max(), label="GT")
    ax03.set_title("Global Slice Spectrum")
    ax03.set_xlabel("Wavelength [nm]")
    ax03.set_ylabel("Intensity [a.u.]")
    ax03.legend()
    ax03.grid()

    # Lifetime comparison
    ax11 = fig.add_subplot(gs[1, 1])
    zoom11 = ax11.imshow(fused_zoomed_tau, cmap='rainbow')
    ax11.set_title(f"Fused Lifetime {fused_zoomed_tau.shape[-1]}x{fused_zoomed_tau.shape[-1]}")
    ax11.axis("off")

    ax12 = fig.add_subplot(gs[1, 2])
    zoom12 = ax12.imshow(ground_truth_spc_tau, cmap='rainbow')
    ax12.set_title(f"GT Lifetime {ground_truth_spc_tau.shape[-1]}x{ground_truth_spc_tau.shape[-1]}")
    ax12.axis("off")

    ax13 = fig.add_subplot(gs[1, 3])
    tmp = x.mean(axis=(1, 3, 4))[:, z_index]
    ax13.plot(t, tmp / tmp.max(), label="Fused")
    tmp = ground_truth_spc.mean(axis=(1, 2, 3))
    ax13.plot(t, tmp / tmp.max(), label="GT")
    ax13.set_title("Global Slice Lifetime")
    ax13.set_xlabel("Time [ns]")
    ax13.set_ylabel("Intensity [a.u.]")
    ax13.legend()
    ax13.grid()

    fig.colorbar(fused_0, ax=ax0, fraction=0.175, pad=0.02, orientation="horizontal", label="Intensity [a.u.]")
    cbar = fig.colorbar(zoom01, ax=ax01, fraction=0.046, pad=0.02, orientation="horizontal", label="Wavelength [nm]")
    cbar.set_ticks(np.linspace(0, 1, len(WAVELENGTHS[::3])))
    cbar.set_ticklabels([f"{w:.0f}" for w in WAVELENGTHS[::3]])

    cbar = fig.colorbar(zoom02, ax=ax02, fraction=0.046, pad=0.02, orientation="horizontal", label="Wavelength [nm]")
    cbar.set_ticks(np.linspace(0, 1, len(WAVELENGTHS[::3])))
    cbar.set_ticklabels([f"{w:.0f}" for w in WAVELENGTHS[::3]])

    fig.colorbar(zoom11, ax=ax11, fraction=0.046, pad=0.02, orientation="horizontal", label="Lifetime [ns]")
    fig.colorbar(zoom12, ax=ax12, fraction=0.046, pad=0.02, orientation="horizontal", label="Lifetime [ns]")

    ax0.annotate(
        "", xy=(-0.01, 0.5), xytext=(zoom_slice.stop, 54),
        xycoords=ax01.transAxes, textcoords="data",
        arrowprops=dict(arrowstyle="->", color="red")
    )

    ax0.annotate(
        "", xy=(-0.01, 0.5), xytext=(zoom_slice.stop, 74),
        xycoords=ax11.transAxes, textcoords="data",
        arrowprops=dict(arrowstyle="->", color="red")
    )

    plt.tight_layout()

    if save_name:
        plt.savefig(FIGURES_PATH / save_name, dpi=300)

    plt.show()
