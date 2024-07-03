from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

FILE_PATH = Path(__file__)
RESOURCES_PATH = FILE_PATH.parent / "resources"


def wavelength_to_color(a):
    lambda_vals = np.array([380, 420, 440, 490, 510, 580, 645, 780])
    r = np.array([97, 106, 0, 0, 0, 255, 255, 97]) / 255
    g = np.array([0, 0, 0, 255, 255, 255, 0, 0]) / 255
    b = np.array([97, 255, 255, 255, 0, 0, 0, 0]) / 255

    interp_r = interp1d(lambda_vals, r, kind='linear', bounds_error=False, fill_value=0)
    interp_g = interp1d(lambda_vals, g, kind='linear', bounds_error=False, fill_value=0)
    interp_b = interp1d(lambda_vals, b, kind='linear', bounds_error=False, fill_value=0)

    x = interp_r(a)
    y = interp_g(a)
    z = interp_b(a)

    return x, y, z


def hyperspectral2RGB(lambda_vals, im):
    if lambda_vals[0] < 380 or lambda_vals[-1] > 780:
        raise ValueError("Wavelength range out of visible range")

    # Values that are less than zero give problems in the spectral visualization
    im[im < 0] = 0

    r, g, b = wavelength_to_color(lambda_vals)

    S_r = np.zeros((im.shape[-2], im.shape[-1], im.shape[-3]))
    S_g = np.zeros((im.shape[-2], im.shape[-1], im.shape[-3]))
    S_b = np.zeros((im.shape[-2], im.shape[-1], im.shape[-3]))

    if im.ndim == 4:
        for li in range(im.shape[-3]):
            I = np.sum(im[:, li, :, :], axis=0)
            S_r[:, :, li] = r[li] * I
            S_g[:, :, li] = g[li] * I
            S_b[:, :, li] = b[li] * I
    else:
        for li in range(im.shape[-3]):
            I = im[li, :, :]
            S_r[:, :, li] = r[li] * I
            S_g[:, :, li] = g[li] * I
            S_b[:, :, li] = b[li] * I

    S_r = np.sum(S_r, axis=2)
    S_g = np.sum(S_g, axis=2)
    S_b = np.sum(S_b, axis=2)

    max_val = np.max([S_r.max(), S_g.max(), S_b.max()])
    S_r_n = 255 * S_r / max_val
    S_g_n = 255 * S_g / max_val
    S_b_n = 255 * S_b / max_val

    S = np.stack((S_r_n, S_g_n, S_b_n), axis=-1).astype(np.uint8)
    return S


def hyperspectral2RGBvolume(lambda_vals, im):
    if lambda_vals[0] < 380 or lambda_vals[-1] > 780:
        raise ValueError("Wavelength range out of visible range")

    r, g, b = wavelength_to_color(lambda_vals)

    num_layers = im.shape[1]
    height, width = im.shape[2], im.shape[3]
    num_lambda = len(lambda_vals)

    S_r = np.zeros((num_lambda, num_layers, height, width))
    S_g = np.zeros((num_lambda, num_layers, height, width))
    S_b = np.zeros((num_lambda, num_layers, height, width))

    for li in range(num_lambda):
        for z in range(num_layers):
            I = im[li, z, :, :]
            S_r[li, z, :, :] += r[li] * I
            S_g[li, z, :, :] += g[li] * I
            S_b[li, z, :, :] += b[li] * I

    S_r = np.sum(S_r, axis=0)
    S_g = np.sum(S_g, axis=0)
    S_b = np.sum(S_b, axis=0)

    max_val = np.max([S_r.max(), S_g.max(), S_b.max()])
    S_r_n = 255 * S_r / max_val
    S_g_n = 255 * S_g / max_val
    S_b_n = 255 * S_b / max_val

    S = np.stack((S_r_n, S_g_n, S_b_n), axis=-1).astype(np.uint8)

    return S


def bin_data(data_nobin, t_nobin, dt):
    data = data_nobin
    t = t_nobin
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
        data = binned
        t = t_nobin.reshape(-1, bin_length).mean(axis=1)

    if abs((t[1] - t[0]) - dt) > (dt / 2):
        print("Some problems determining the desired bin size.")

    dt = t[1] - t[0]

    return t, data, dt
