import torch
import numpy as np
from torchvision.transforms import Resize, InterpolationMode

from tqdm import tqdm


def baseline(cmos, spc, device):
    # Idea: Use bilinear interpolation for SPC and then multiply by CMOS for each z.

    # for each time:
    #   for each lambda:
    #       for each z in cmos:
    #           interpolate spc to z
    #           multiply spc by cmos[z]

    spc = torch.from_numpy(spc.astype(np.float32)).to(device)  # (time,lambda,x,y)
    cmos = torch.from_numpy(cmos.astype(np.float32)).to(device)  # (z,x,y)

    n_times = spc.shape[0]
    n_lambdas = spc.shape[1]
    xy_dim = cmos.shape[1]
    z_dim = cmos.shape[0]

    x = torch.zeros((n_times, n_lambdas, z_dim, xy_dim, xy_dim), requires_grad=False).to(device)

    upsampler = Resize(
        size=(cmos.shape[-2], cmos.shape[-1]),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).to(device)

    for time in tqdm(range(spc.shape[0])):
        for z in range(cmos.shape[0]):
            x[time, :, z, :, :] = upsampler(spc[time, :, :, :]) * cmos[z, :, :]

    return x.cpu().detach().numpy()
