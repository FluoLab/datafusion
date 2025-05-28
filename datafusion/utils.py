from pathlib import Path
from copy import deepcopy
from joblib import Parallel, delayed

import h5py
import torch
import numpy as np
import scipy as sp
from tqdm.autonotebook import tqdm
from scipy.linalg import lstsq
from scipy.optimize import curve_fit
from matplotlib.colors import hsv_to_rgb

FILE_PATH = Path(__file__)
PROJECT_PATH = FILE_PATH.parent.parent
RESOURCES_PATH = PROJECT_PATH / "resources"
FIGURES_PATH = PROJECT_PATH / "figures"
DF_PATH = PROJECT_PATH / "datafusion"


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
    spc: np.ndarray,
    forward_matrix: np.ndarray,
    algo: callable = lstsq,
    n_jobs: int = 8,
    img_dim: int = 32,
):
    """
    Reconstructs the SPC from the forward matrix and the SPC.
    :param spc: SPC  to reconstruct of shape (n_times, n_spectra, n_measurements).
    :param forward_matrix: Forward matrix of shape (n_measurements, pattern_size).
    :param algo: Algorithm to use for the reconstruction.
    :param n_jobs: Number of jobs to use for parallelization.
    :param img_dim: Image dimension.
    :return: Image reconstruction of shape (n_times, n_spectra, img_dim, img_dim).
             Where img_dim is the square root of the number of measurements.
    """
    n_times, n_spectra, n_measurements = spc.shape[0], spc.shape[1], int(img_dim**2)

    recon = np.empty((n_times, n_spectra, n_measurements), dtype=np.float32)
    for s in tqdm(range(spc.shape[1])):
        s_recon = Parallel(n_jobs=n_jobs)(
            delayed(algo)(forward_matrix, spc[t, s]) for t in range(n_times)
        )
        recon[:, s] = np.array([x[0] for x in s_recon])
    return recon.reshape(spc.shape[0], spc.shape[1], img_dim, img_dim)


def load_raw_spc(
    spc_path: Path, n_measurements: int = 1024, dtype: np.dtype = np.float64
):
    """
    Loads the raw SPC data. The data is expected to be in the shape (n_measurements, n_spectra, n_times).
    :param spc_path: Path to the additional SPC data.
    :param n_measurements: Number of measurements used.
    :param dtype: Data type to use.
    :return: Raw SPC data of shape (n_times, n_spectra, n_measurements).
    """
    # TODO: This only works for Pos One Neg type of measurement, add other options.
    with h5py.File(spc_path, "r") as f:
        spc = np.array(f["spc"], order="C")[1 : n_measurements + 2]
        spc[545] = spc[0] + spc[1]
        spc = np.delete(spc, 1, axis=0)
        spc = spc.astype(dtype)
        spc = np.swapaxes(spc, 0, 2)
    return spc


def preprocess_raw_spc(
    raw_spc_path: Path,
    reconstruction_save_path: Path,
    forward_matrix_path: Path,
    efficiency_calib_path: Path,
    offset_calib_path: Path,
    temporal_axis_path: Path,
    n_measurements: int,
    max_times: int = 2048,
    n_bins: int = 32,
    n_jobs: int = 8,
    algo: callable = lstsq,
    compression: float | None = 1,
    dtype: np.dtype = np.float64,
):
    """
    Preprocesses the raw SPC data and saves the reconstruction.
    The preprocessing steps it takes are:
    1. Calibrate the SPC data.
    2. Cut the SPC data.
    3. Bin the SPC data.
    4. Reconstruct the SPC data.
    :param raw_spc_path: The path to the raw SPC data.
    :param reconstruction_save_path:  The path to save the reconstruction.
    :param forward_matrix_path: Path to the forward matrix.
    :param efficiency_calib_path: Path to the efficiency calibration.
    :param offset_calib_path: Path to the offset calibration.
    :param temporal_axis_path: Path to the temporal axis.
    :param n_measurements: Number of measurements.
    :param max_times: Maximum number of times to keep.
    :param n_bins: Number of bins to use.
    :param n_jobs: Number of jobs to use for parallelization.
    :param dtype: Data type to use.
    :param algo: Algorithm to use for the reconstruction.
    :param compression: Compression factor to use.
    :return: The reconstructed SPC data, the binned time axis, and the binned time step.
    """
    spc = load_raw_spc(raw_spc_path, n_measurements=n_measurements, dtype=dtype)
    # (n_times, n_spectra, n_measurements)

    forward_matrix = sp.io.loadmat(str(forward_matrix_path))["M"][::2]
    forward_matrix = np.array(forward_matrix, dtype=dtype, order="C")
    # (n_measurements, pattern_size)

    if compression is not None and compression < 1:
        n_patterns = int((1 - compression) * forward_matrix.shape[0])
        spc = spc[:, :, :n_patterns]
        forward_matrix = forward_matrix[:n_patterns]

    t = np.load(temporal_axis_path).flatten().astype(dtype)
    # (n_times,)

    spc_calib = calibrate_spc(spc, efficiency_calib_path, offset_calib_path)
    spc_calib_cut, t_cut = cut_spc(spc_calib, t, max_times=max_times)
    spc_calib_cut_binned, t_cut_binned, dt_cut_binned = bin_spc(
        spc_calib_cut, t_cut, n_bins=n_bins
    )

    spc_recon = reconstruct_spc(
        spc_calib_cut_binned,
        forward_matrix,
        algo=algo,  # accepts nnls, lstsq, and TVAL3(),
        n_jobs=n_jobs,
    )  # (n_times, n_spectra, img_dim, img_dim)

    np.savez_compressed(
        reconstruction_save_path,
        spc_recon=spc_recon,
        t_cut_binned=t_cut_binned,
        dt_cut_binned=dt_cut_binned,
    )
    return spc_recon, t_cut_binned, dt_cut_binned


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
            "The spectral_volume should have 4 dimensions: (n_lambdas, depth, height, width)"
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
    rgb_volume /= rgb_volume.max(axis=0) + 1e-10
    srgb_volume = linear_to_srgb(rgb_volume)
    srgb_volume *= intensity_volume / intensity_volume.max()

    srgb_volume = np.moveaxis(srgb_volume, 0, -1)
    return srgb_volume


def time_volume_to_lifetime(t, tensor, tau_clip=None, max_tau=6.0, noise_thr=0.1):
    lifetime_volume = np.zeros(
        (tensor.shape[1], tensor.shape[2], tensor.shape[3], 3), dtype=np.float32
    )
    a_out = np.zeros(tensor.shape[1:], dtype=np.float32)
    tau_out = np.zeros(tensor.shape[1:], dtype=np.float32)
    # c_out = np.zeros(tensor.shape[1:], dtype=np.float32)

    intensity = tensor.sum(axis=0)
    intensity /= intensity.max()

    for zi in tqdm(range(tensor.shape[1])):
        for xi in range(tensor.shape[2]):
            for yi in range(tensor.shape[3]):

                if intensity[zi, xi, yi] < noise_thr:
                    a_out[zi, xi, yi] = 0
                    tau_out[zi, xi, yi] = 0
                    # c_out[zi, xi, yi] = 0

                else:
                    max_voxel = tensor[:, zi, xi, yi].max()
                    params, covariance = curve_fit(
                        mono_exponential_decay_numpy,
                        t,
                        tensor[:, zi, xi, yi] / max_voxel,
                        bounds=([0.0, 1e-6, -0.1], [1.0, max_tau, 0.1]),
                        p0=(0.5, 2.0, 1e-4),
                        maxfev=5000,
                    )
                    a_out[zi, xi, yi] = params[0] * max_voxel
                    tau_out[zi, xi, yi] = params[1]
                    # c_out[zi, xi, yi] = params[2]

    a_out /= a_out.max()
    if tau_clip is not None:
        tau_out = np.clip(tau_out, tau_clip[0], tau_clip[1])
    tau_min = tau_out.min()
    tau_max = tau_out.max()
    for zi in range(tensor.shape[1]):
        # h = (260 / 360) * (1 - (tau_out[zi] - tau_min) / (tau_max - tau_min))
        h = (260 / 360) * (
            1 - (tau_out[zi] - tau_clip[0]) / (tau_clip[1] - tau_clip[0])
        )
        lifetime_volume[zi] = hsv_to_rgb(
            np.stack([h, np.ones_like(tau_out[zi]), a_out[zi]], axis=-1)
        )
    return lifetime_volume, tau_min, tau_max


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


def sam(vec1, vec2):
    return np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def rmse(vec1, vec2):
    return np.sqrt(np.mean((vec1 - vec2) ** 2))
