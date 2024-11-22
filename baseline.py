import torch
import numpy as np
from torchvision.transforms import Resize, InterpolationMode

from tqdm import tqdm


def baseline(cmos, spc, device, return_numpy=True):
    # Idea: Use bilinear interpolation for SPC and then multiply by normalized CMOS for each z.

    if isinstance(spc, np.ndarray):
        spc = torch.from_numpy(spc.astype(np.float32))  # (time,lambda,x,y)

    if isinstance(cmos, np.ndarray):
        cmos = torch.from_numpy(cmos.astype(np.float32))  # (z,x,y)

    spc = spc.to(device)
    cmos = cmos.to(device)

    n_times = spc.shape[0]
    n_lambdas = spc.shape[1]
    xy_dim = cmos.shape[1]
    z_dim = cmos.shape[0]

    x = torch.zeros(
        (n_times, n_lambdas, z_dim, xy_dim, xy_dim), requires_grad=False
    ).to(device)

    upsampler = Resize(
        size=(cmos.shape[-2], cmos.shape[-1]),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).to(device)

    cmos = cmos / cmos.sum(dim=0, keepdim=True)

    for time in tqdm(range(spc.shape[0])):
        for z in range(cmos.shape[0]):
            x[time, :, z, :, :] = upsampler(spc[time, :, :, :]) * cmos[z, :, :]

    return x.cpu().numpy() if return_numpy else x
