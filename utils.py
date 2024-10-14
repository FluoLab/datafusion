from pathlib import Path

import pywt
import torch
import numpy as np
from scipy.optimize import curve_fit

FILE_PATH = Path(__file__)
RESOURCES_PATH = FILE_PATH.parent / "resources"
FIGURES_PATH = FILE_PATH.parent / "figures"

# Alberto's space
# SPECTRUM_CHANNELS = np.array([380, 420, 440, 490, 510, 580, 645, 780])
# R = np.array([97, 106, 0, 0, 0, 255, 255, 97]) / 255
# G = np.array([0, 0, 0, 255, 255, 255, 0, 0]) / 255
# B = np.array([97, 255, 255, 255, 0, 0, 0, 0]) / 255


WAVELENGTHS = np.array([547.35972343, 556.56210764, 565.76449186, 574.96687608, 584.1692603, 593.37164452,
                        602.57402874, 611.77641296, 620.97879718, 630.18118139, 639.38356561, 648.58594983,
                        657.78833405, 666.99071827, 676.19310249, 685.39548671])

# sRGB values
R = np.array([154, 184, 212, 240, 255, 255, 255, 255, 255, 255, 255, 251, 241, 231, 221, 211]) / 255
G = np.array([255, 255, 255, 255, 241, 212, 181, 149, 114, 78, 35, 30, 30, 30, 30, 30]) / 255
B = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / 255


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


def spectral_volume_to_srgb(spectrum, spectral_volume):
    if spectrum[0] < 380 or spectrum[-1] > 780:
        raise ValueError("Wavelength range out of visible range")

    if spectral_volume.ndim != 4:
        raise ValueError("The spectral_volume should have 4 dimensions: (num_lambda, depth, height, width)")

    if spectrum.shape[0] != spectral_volume.shape[0]:
        raise ValueError("The number of lambda values should match the number of lambda values in the tensor")

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


def mono_exponential_decay_model(t, I0, tau, c):
    return I0 * np.exp(-t / tau) + c


def fit_decay(x, y, lower_bounds=(0.1, 0.01, 0.001), upper_bounds=(1.0, 6.0, 0.01)):
    params, covariance = curve_fit(
        mono_exponential_decay_model,
        x,
        y,
        bounds=(lower_bounds, upper_bounds),
        p0=(0.5, 2.0, 0.005),
        maxfev=5000,
    )
    return params


def wavelet_denoising(x, wavelet="db2", threshold=0.1):
    coeffs = pywt.wavedec(x, wavelet)
    coeffs_thresholded = [pywt.threshold(c, threshold) for c in coeffs]
    return pywt.waverec(coeffs_thresholded, wavelet)


def time_image_to_lifetime(t, dt, img, denoise=wavelet_denoising):
    img_out = np.zeros((img.shape[1], img.shape[2]))
    time_coords = np.arange(0.0, dt * len(t), dt)
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            img_out[i, j] = fit_decay(time_coords, denoise(img[:, i, j]))[1]

    return img_out


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


def decay_model(I, tau, c, t):
    # TODO: Find ways to make this more efficient
    return torch.cat([(I[i] * torch.exp(-t / tau[i]) + c[i]).unsqueeze(0) for i in range(len(I))]).sum(dim=0)


def mono_exponential_decay(I, tau, c, t):
    return (I.unsqueeze(1) * torch.exp(-t / tau.unsqueeze(1)) + c.unsqueeze(1)).T


def get_discrete_time_decay(tensor, t, cmos_mask):
    discrete_tensor_shape = (len(t), tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.shape[4])
    indices = cmos_mask.nonzero(as_tuple=True)

    tensor = tensor.reshape(tensor.shape[0], -1)
    return mono_exponential_decay(tensor[0], tensor[1], tensor[2], t).reshape(discrete_tensor_shape)

    # for l in tqdm(range(tensor.shape[1])):
    #     # Use the mask to avoid computing the decay for pixels that are not in the mask
    #     for z, x, y in zip(*indices):
    #         discrete_tensor[:, l, z, x, y] = decay_model(
    #             tensor[:, l, z, x, y][::3],
    #             tensor[:, l, z, x, y][1::3],
    #             tensor[:, l, z, x, y][2::3],
    #             t
    #         )