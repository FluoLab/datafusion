from pathlib import Path
from copy import deepcopy
from joblib import Parallel, delayed

import h5py
import pywt
import torch
import numpy as np
import scipy as sp
from tqdm import tqdm
from scipy.optimize import nnls
from scipy.optimize import curve_fit

FILE_PATH = Path(__file__)
RESOURCES_PATH = FILE_PATH.parent / "resources"
FIGURES_PATH = FILE_PATH.parent / "figures"
TVAL3_PATH = FILE_PATH.parent / "vendored" / "TVAL3"

# L16 at 610nm: wavelengths and sRGB values (for old colouring scheme)
# WAVELENGTHS = np.array([547.36, 556.56, 565.76, 574.97, 584.17, 593.37, 602.57, 611.78, 620.98, 630.18, 639.38, 648.59, 657.79, 666.99, 676.19, 685.4])
# R = np.array([154, 184, 212, 240, 255, 255, 255, 255, 255, 255, 255, 251, 241, 231, 221, 211]) / 255
# G = np.array([255, 255, 255, 255, 241, 212, 181, 149, 114, 78, 35, 30, 30, 30, 30, 30]) / 255
# B = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / 255


def calibrate_spc(spc: np.ndarray, eff_path: Path, off_path: Path):
    """
    Calibrates the SPC data using efficiency and offset calibration.
    :param spc: SPC data of shape (n_times, n_spectra, n_measurements).
    :param eff_path: Path to the efficiency calibration file.
    :param off_path: Path to the offset calibration file.
    :return: Calibrated SPC data of shape (n_times, n_spectra, n_measurements).
    """
    spc = deepcopy(spc)
    eff = sp.io.loadmat(str(eff_path))["efficiency_L16"].flatten()
    off = sp.io.loadmat(str(off_path))["time_delay_shift"].flatten()

    for spectral_index in range(spc.shape[1]):
        spc[:, spectral_index, :] *= eff[spectral_index]
        spc[:, spectral_index, :] = np.roll(
            spc[:, spectral_index, :], off[spectral_index], axis=0
        )

    return spc


def cut_spc(spc: np.ndarray, t: np.ndarray, max_times: int = 2048):
    """
    Cuts the SPC data from its peak to a maximum number of times.
    :param spc: SPC data of shape (n_times, n_spectra, n_measurements)
    :param t: Time axis of shape (n_times,).
    :param max_times: Maximum number of times to keep.
    :return: Cut SPC data of shape (max_times, n_spectra, n_measurements),
             and the cut time axis of shape (max_times,).
    """
    spc, t = deepcopy(spc), deepcopy(t)
    curve = np.sum(spc, axis=(1, 2))
    curve_argmax = np.argmax(curve)
    spc = spc[curve_argmax : curve_argmax + max_times]
    t = t[curve_argmax : curve_argmax + max_times]
    t = t - t[0]
    return spc, t


def bin_spc(spc: np.ndarray, t: np.ndarray, n_bins: int = 64):
    """
    Bins the SPC data.
    :param spc: SPC data of shape (n_times, n_spectra, n_measurements)
    :param t: Time axis of shape (n_times,).
    :param n_bins: Number of bins to use.
    :return: Binned SPC data of shape (n_bins, n_spectra, n_measurements),
             binned time axis of shape (n_bins,),
             and the binned time step.
    """
    spc, t = deepcopy(spc), deepcopy(t)
    bin_length = int(len(t) // n_bins)

    binned_spc = np.empty((n_bins, spc.shape[1], spc.shape[2]))
    binned_t = t.reshape(-1, bin_length).mean(axis=1)
    binned_dt = binned_t[1] - binned_t[0]

    for li in range(spc.shape[1]):
        for mi in range(spc.shape[2]):
            binned_spc[:, li, mi] = spc[:, li, mi].reshape(-1, bin_length).sum(axis=1)

    return binned_spc, binned_t, binned_dt


def reconstruct_spc(
    spc: np.ndarray, forward_matrix: np.ndarray, algo: callable = nnls, n_jobs: int = 8
):
    """
    Reconstructs the SPC from the forward matrix and the SPC.
    :param spc: SPC  to reconstruct of shape (n_times, n_spectra, n_measurements).
    :param forward_matrix: Forward matrix of shape (n_measurements, pattern_size).
    :param algo: Algorithm to use for the reconstruction.
    :param n_jobs: Number of jobs to use for parallelization.
    :return: Image reconstruction of shape (n_times, n_spectra, img_dim, img_dim).
             Where img_dim is the square root of the number of measurements.
    """
    n_times, n_spectra, n_measurements = spc.shape[0], spc.shape[1], spc.shape[2]
    img_dim = int(np.sqrt(n_measurements))

    recon = np.empty((n_times, n_spectra, n_measurements), dtype=np.float64)
    for s in tqdm(range(spc.shape[1])):
        s_recon = Parallel(n_jobs=n_jobs)(
            delayed(algo)(forward_matrix, spc[t, s]) for t in range(n_times)
        )
        recon[:, s] = np.array([x[0] for x in s_recon])
    return recon.reshape(spc.shape[0], spc.shape[1], img_dim, img_dim)


def load_raw_spc(spc_path: Path):
    """
    Loads the raw SPC data.
    :param spc_path: Path to the raw SPC data.
    :return: Raw SPC data of shape (n_times, n_spectra, n_measurements).
    """
    with h5py.File(spc_path, "r") as f:
        spc = np.array(f["spc"])[1:1026]
        spc[545] = spc[0] + spc[1]
        spc = np.delete(spc, 1, axis=0)
        spc = spc.astype(np.float64)
        spc = np.swapaxes(spc, 0, 2)
    return spc


def linear_to_srgb(channel):
    channel = channel.clip(0, 1)
    return np.where(
        channel <= 0.0031308,
        12.92 * channel,
        1.055 * (channel ** (1 / 2.4)) - 0.055,
    )


def wavelength_to_srgb(lambdas, method):
    cmf_table = np.loadtxt(RESOURCES_PATH / f"srgb_cmf_{method}.csv", delimiter=",")
    wavelengths = cmf_table[:, 0].flatten()
    srgb_cmf = cmf_table[:, 1:].T
    return np.array([np.interp(lambdas, wavelengths, channel) for channel in srgb_cmf])


def spectral_volume_to_color(lambdas, spectral_volume, method="basic"):
    if lambdas[0] < 380 or lambdas[-1] > 780:
        raise ValueError("Wavelength range out of visible range")

    if spectral_volume.ndim != 4:
        raise ValueError(
            "The spectral_volume should have 4 dimensions: (num_lambda, depth, height, width)"
        )

    if lambdas.shape[0] != spectral_volume.shape[0]:
        raise ValueError(
            "The number of lambda values should match the number of lambda values in the tensor"
        )

    # Values that are less than zero give problems in the spectral visualization
    spectral_volume[spectral_volume < 0] = 0
    intensity_volume = spectral_volume.sum(axis=0)

    srgb_cmf = wavelength_to_srgb(lambdas, method)
    rgb_volume = np.apply_along_axis(lambda s: srgb_cmf @ s, 0, spectral_volume)

    # Discard (visual) intensity and scale to original (relative) photon intensity
    rgb_volume /= rgb_volume.max(axis=0)
    srgb_volume = linear_to_srgb(rgb_volume)
    srgb_volume *= intensity_volume / intensity_volume.max()

    srgb_volume = np.moveaxis(srgb_volume, 0, -1)
    return srgb_volume


def wavelet_denoising(x, wavelet="db2", threshold=0.1):
    coeffs = pywt.wavedec(x, wavelet)
    coeffs_thresholded = [pywt.threshold(c, threshold) for c in coeffs]
    return pywt.waverec(coeffs_thresholded, wavelet)


def time_volume_to_lifetime(t, dt, time_volume):
    raise NotImplementedError("This function is not implemented yet.")


def bin_data(data, t, dt):
    # TODO: Fix naming
    bin_size = round(len(t) / (dt / (t[1] - t[0])))

    if bin_size < len(t):
        N = data.shape[0]
        K = np.arange(1, N + 1)
        D = K[N % K == 0]
        p = np.argmin(np.abs(bin_size - D))
        bins = D[p]
        bin_length = int(N / bins)

        binned = np.zeros((bins, data.shape[1], data.shape[2], data.shape[3]))
        for li in range(data.shape[1]):
            for xi in range(data.shape[2]):
                for yi in range(data.shape[3]):
                    binned[:, li, xi, yi] = (
                        data[:, li, xi, yi].reshape(-1, bin_length).sum(axis=1)
                    )
        t = t.reshape(-1, bin_length).mean(axis=1)
    else:
        raise ValueError("The bin size is larger than the data.")

    if abs((t[1] - t[0]) - dt) > (dt / 2):
        print("Some problems determining the desired bin size.")

    dt = t[1] - t[0]

    return t, binned, dt


def mono_exponential_decay_numpy(t, I, tau, c):
    return I * np.exp(-t / tau) + c


def mono_exponential_decay_torch(I, tau, c, t):
    return (I.unsqueeze(1) * torch.exp(-t / tau.unsqueeze(1)) + c.unsqueeze(1)).T


def get_discrete_time_decay(tensor, t):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor.astype(np.float32))

    tensor = tensor.reshape(tensor.shape[0], -1)
    tensor = mono_exponential_decay_torch(tensor[0], tensor[1], tensor[2], t)
    tensor = tensor.reshape(
        len(t),
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
        tensor.shape[4],
    )
    return tensor
