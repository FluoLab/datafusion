from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# from scipy.interpolate import interp1d

FILE_PATH = Path(__file__)
RESOURCES_PATH = FILE_PATH.parent / "resources"


def decay_model(I, tau, t):
    return torch.sum(torch.cat([(I[i] * torch.exp(-t / tau[i])).unsqueeze(0) for i in range(len(I))]), dim=0)


def get_discrete_time_decay(tensor, t, cmos_mask):
    discrete_tensor = torch.zeros(len(t), tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.shape[4])
    indices = cmos_mask.nonzero(as_tuple=True)

    for l in tqdm(range(tensor.shape[1])):
        # Use the mask to avoid computing the decay for pixels that are not in the mask
        for z, x, y in zip(*indices):
            discrete_tensor[:, l, z, x, y] = decay_model(
                tensor[:, l, z, x, y][::2],
                tensor[:, l, z, x, y][1::2],
                t
            )
        # for z in range(tensor.shape[2]):
        #     for x in range(tensor.shape[3]):
        #         for y in range(tensor.shape[4]):
        #             discrete_tensor[:, l, z, x, y] = decay_model(
        #                 tensor[:, l, z, x, y][::3],
        #                 tensor[:, l, z, x, y][1::3],
        #                 t
        #             )

    return discrete_tensor


def wavelength_to_color(a):
    # SCALA ALBERTO
    # lambda_vals = np.array([380, 420, 440, 490, 510, 580, 645, 780])
    # r = np.array([97, 106, 0, 0, 0, 255, 255, 97]) / 255
    # g = np.array([0, 0, 0, 255, 255, 255, 0, 0]) / 255
    # b = np.array([97, 255, 255, 255, 0, 0, 0, 0]) / 255

    # SCALA PER OGNI CANALE
    lambda_vals = np.array([547.35972343, 556.56210764, 565.76449186, 574.96687608, 584.1692603,
                            593.37164452, 602.57402874, 611.77641296, 620.97879718, 630.18118139,
                            639.38356561, 648.58594983, 657.78833405, 666.99071827, 676.19310249,
                            685.39548671])
    r = np.array([154, 184, 212, 240, 255, 255, 255, 255, 255, 255, 255, 251, 241, 231, 221, 211]) / 255
    g = np.array([255, 255, 255, 255, 241, 212, 181, 149, 114, 78, 35, 30, 30, 30, 30, 30]) / 255
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / 255

    r_l = RGB_to_lin(r)
    g_l = RGB_to_lin(g)
    b_l = RGB_to_lin(b)

    x = np.interp(a, lambda_vals, r_l)
    y = np.interp(a, lambda_vals, g_l)
    z = np.interp(a, lambda_vals, b_l)

    return x, y, z


def hyperspectral2RGB(lambda_vals, im):
    if lambda_vals[0] < 380 or lambda_vals[-1] > 780:
        raise ValueError("Wavelength range out of visible range")

    # Values that are less than zero give problems in the spectral visualization
    im[im < 0] = 0

    r, g, b = wavelength_to_color(lambda_vals)

    S_r = np.zeros((im.shape[1], im.shape[2], im.shape[0]))
    S_g = np.zeros((im.shape[1], im.shape[2], im.shape[0]))
    S_b = np.zeros((im.shape[1], im.shape[2], im.shape[0]))

    for li in range(im.shape[0]):
        I = im[li, :, :]
        S_r[:, :, li] = r[li] * I
        S_g[:, :, li] = g[li] * I
        S_b[:, :, li] = b[li] * I

    S_r = np.sum(S_r, axis=2)
    S_g = np.sum(S_g, axis=2)
    S_b = np.sum(S_b, axis=2)

    max_val = np.max([S_r.max(), S_g.max(), S_b.max()])
    S_r_n = S_r / max_val
    S_g_n = S_g / max_val
    S_b_n = S_b / max_val

    S = np.stack((S_r_n, S_g_n, S_b_n), axis=-1)  # .astype(np.uint8)

    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            S[i, j] = lin_to_RGB(S[i, j])

    S = (S * 255).astype(np.uint8)

    return S


def hyperspectral2RGBvolume(lambda_vals, im):
    if lambda_vals[0] < 380 or lambda_vals[-1] > 780:
        raise ValueError("Wavelength range out of visible range")

    # Values that are less than zero give problems in the spectral visualization
    im[im < 0] = 0

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


def RGB_to_lin(colors):
    for i in range(len(colors)):
        if (colors[i] <= 0.04045):
            colors[i] = colors[i] / 12.92
        else:
            colors[i] = np.power((colors[i] + 0.055) / 1.055, 2.4)
    return colors


def lin_to_RGB(colors):
    for color in colors:
        if (color > 0.0031308):
            color = 1.055 * (np.power(color, 1 / 2.4)) - 0.055
        else:
            color = 12.92 * color
    return colors
