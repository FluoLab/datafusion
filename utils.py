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

# Alberto's wavelengths and RGB values
# WAVELENGTHS = np.array([380, 420, 440, 490, 510, 580, 645, 780])
# R = np.array([97, 106, 0, 0, 0, 255, 255, 97]) / 255
# G = np.array([0, 0, 0, 255, 255, 255, 0, 0]) / 255
# B = np.array([97, 255, 255, 255, 0, 0, 0, 0]) / 255

# TODO: Add wavelengths and RGB values from a file.
WAVELENGTHS = np.array(
    [
        547.35972343,
        556.56210764,
        565.76449186,
        574.96687608,
        584.1692603,
        593.37164452,
        602.57402874,
        611.77641296,
        620.97879718,
        630.18118139,
        639.38356561,
        648.58594983,
        657.78833405,
        666.99071827,
        676.19310249,
        685.39548671,
    ]
)

# sRGB values
R = (
    np.array(
        [
            154,
            184,
            212,
            240,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            251,
            241,
            231,
            221,
            211,
        ]
    )
    / 255
)
G = (
    np.array(
        [
            255,
            255,
            255,
            255,
            241,
            212,
            181,
            149,
            114,
            78,
            35,
            30,
            30,
            30,
            30,
            30,
        ]
    )
    / 255
)
B = (
    np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    / 255
)


def srgb_to_linear(channel):
    return np.where(
        channel <= 0.04045,
        channel / 12.92,
        ((channel + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(channel):
    return np.where(
        channel <= 0.0031308,
        12.92 * channel,
        1.055 * (channel ** (1 / 2.4)) - 0.055,
    ).clip(0, 1)


def wavelength_to_srgb(lambdas):
    return (
        np.interp(lambdas, WAVELENGTHS, R),
        np.interp(lambdas, WAVELENGTHS, G),
        np.interp(lambdas, WAVELENGTHS, B),
    )


def wavelength_to_linear_rgb(lambdas):
    return (
        np.interp(lambdas, WAVELENGTHS, srgb_to_linear(R)),
        np.interp(lambdas, WAVELENGTHS, srgb_to_linear(G)),
        np.interp(lambdas, WAVELENGTHS, srgb_to_linear(B)),
    )


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


def spectral_volume_to_srgb(spectrum, spectral_volume):
    if spectrum[0] < 380 or spectrum[-1] > 780:
        raise ValueError("Wavelength range out of visible range")

    if spectral_volume.ndim != 4:
        raise ValueError(
            "The spectral_volume should have 4 dimensions: (num_lambda, depth, height, width)"
        )

    if spectrum.shape[0] != spectral_volume.shape[0]:
        raise ValueError(
            "The number of lambda values should match the number of lambda values in the tensor"
        )

    num_lambda, depth, height, width = spectral_volume.shape

    # Values that are less than zero give problems in the spectral visualization
    spectral_volume[spectral_volume < 0] = 0

    r, g, b = wavelength_to_srgb(spectrum)

    images_r = np.zeros((num_lambda, depth, height, width))
    images_g = np.zeros((num_lambda, depth, height, width))
    images_b = np.zeros((num_lambda, depth, height, width))

    for li in range(num_lambda):
        for z in range(depth):
            images_r[li, z] = r[li] * spectral_volume[li, z]
            images_g[li, z] = g[li] * spectral_volume[li, z]
            images_b[li, z] = b[li] * spectral_volume[li, z]

    images_r = np.sum(images_r, axis=0)
    images_g = np.sum(images_g, axis=0)
    images_b = np.sum(images_b, axis=0)

    srgb_volume = np.stack((images_r, images_g, images_b), axis=-1)
    srgb_volume = srgb_volume / np.max(srgb_volume)
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
