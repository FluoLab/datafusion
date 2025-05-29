import torch
import numpy as np
from torchvision.transforms import Resize, InterpolationMode

from tqdm.autonotebook import tqdm


def baseline(
    cmos: np.ndarray | torch.Tensor,
    spc: np.ndarray | torch.Tensor,
    device: str,
    return_numpy: bool = True,
) -> np.ndarray | torch.Tensor:
    """
    Baseline function to compute a good starting point for data fusion algorithms.
    Uses bilinear interpolation for SPC and then multiplies by normalized CMOS for each z.
    :param cmos: CMOS data as a numpy array or torch tensor of shape (z, x, y).
    :param spc: SPC data as a numpy array or torch tensor of shape (time, lambda, x, y).
    :param device: Device to run the computation on (e.g., 'cuda' or 'cpu').
    :param return_numpy: If True, returns the result as a numpy array; otherwise, returns a torch tensor.
    :return: A numpy array or torch tensor of shape (time, lambda, z, x, y) containing the fused data.
    """

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

    x = torch.zeros((n_times, n_lambdas, z_dim, xy_dim, xy_dim), requires_grad=False).to(device)

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
