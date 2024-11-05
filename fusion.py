import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.functional import mse_loss
from torchvision.transforms import Resize, InterpolationMode
from torchvision.transforms.functional import resize

from baseline import baseline


def _initialize(spc, cmos, x_shape, init_type, device, seed):
    torch.manual_seed(seed)
    if init_type == "random":
        x = (cmos.min() - cmos.max()) * torch.rand(x_shape, device=device) + cmos.max()
    elif init_type == "zeros":
        x = torch.zeros(x_shape).to(device)
    elif init_type == "baseline":
        x = baseline(cmos, spc, device, return_numpy=False)
    elif init_type == "upsampled_spc":
        upsampler = Resize(
            size=(cmos.shape[-2], cmos.shape[-1]),
            interpolation=InterpolationMode.NEAREST,
        ).to(device)
        x = upsampler(spc).unsqueeze(2).repeat(1, 1, cmos.shape[0], 1, 1)
    else:
        raise ValueError("Invalid initialization type.")
    return x


def _mask_gradients(x, cmos_mask):
    x.grad[:, :, ~cmos_mask] = 0.0


def _get_masks(spc, cmos):
    cmos_mask = cmos > (0.05 * cmos.max())
    spc_mask = resize(
        cmos_mask.any(dim=0).unsqueeze(0).float(),
        size=[spc.shape[-2], spc.shape[-1]],
        interpolation=InterpolationMode.BILINEAR,
        antialias=False,
    ).bool()
    return spc_mask, cmos_mask


def normalize_energy(tensor, total_energy=1):
    return total_energy * tensor / tensor.sum()


def fuse(
    spc,
    cmos,
    weights,
    iterations=100,
    lr=0.001,
    init_type="random",
    mask_initializations=False,
    mask_gradients=False,
    non_neg=True,
    device="cpu",
    seed=42,
    total_energy=1000,
    return_numpy=True,
):
    # torch.set_default_dtype(torch.float64)

    # Convert numpy arrays to torch tensors
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
    x_shape = (n_times, n_lambdas, z_dim, xy_dim, xy_dim)
    spatial_increase = cmos.shape[-1] // spc.shape[-1]

    spatial_loss_history = []
    lambda_time_loss_history = []
    global_loss_history = []

    # Get the masks for the spatial dimensions
    spc_mask, cmos_mask = _get_masks(spc, cmos)

    # Mask
    if mask_initializations:
        spc = spc * spc_mask.float()
        cmos = cmos * cmos_mask.float()

    # Normalize the energy of the input data
    spc = normalize_energy(spc, total_energy)
    cmos = normalize_energy(cmos, total_energy)

    # Initialize the parameters
    x = _initialize(spc, cmos, x_shape, init_type, device, seed)
    x = normalize_energy(x, total_energy)
    if mask_initializations:
        x[:, :, ~cmos_mask] = 0.0
    x = torch.nn.Parameter(x, requires_grad=True)

    spc_mask = spc_mask.squeeze(0)
    optimizer = torch.optim.Adam([x], lr=lr, amsgrad=True)

    down_sampler = torch.nn.LPPool2d(1, spatial_increase, spatial_increase).to(device)

    for _ in (progress_bar := tqdm(range(iterations))):
        optimizer.zero_grad()
        resized_x = torch.cat([down_sampler(xi.sum(dim=1)).unsqueeze(0) for xi in x])

        # Computing losses
        if mask_gradients:
            spatial_loss = weights["spatial"] * mse_loss(
                cmos[cmos_mask], x.sum(dim=(0, 1))[cmos_mask]
            )
            lambda_time_loss = weights["lambda_time"] * mse_loss(
                spc[:, :, spc_mask], resized_x[:, :, spc_mask]
            )

        else:
            spatial_loss = weights["spatial"] * mse_loss(
                cmos.flatten(), x.sum(dim=(0, 1)).flatten()
            )
            lambda_time_loss = weights["lambda_time"] * mse_loss(
                spc.flatten(), resized_x.flatten()
            )

        # Global has no spatial dimension, so no need to mask.
        global_loss = weights["global"] * mse_loss(
            spc.sum(dim=(2, 3)).flatten(),
            x.sum(dim=(2, 3, 4)).flatten(),
        )

        loss = spatial_loss + lambda_time_loss + global_loss
        if weights["l2_regularization"] > 0:
            loss = loss + weights["l2_regularization"] * x.norm(2)
        loss.backward()

        log = (
            f"Spatial: {spatial_loss.item():.2E} | "
            f"Lambda Time: {lambda_time_loss.item():.2E} | "
            f"Global: {global_loss.item():.2E} | "
            # f"Grad Norm: {x.grad.data.norm(2).item():.2E}"
        )
        spatial_loss_history.append(spatial_loss.item())
        lambda_time_loss_history.append(lambda_time_loss.item())
        global_loss_history.append(global_loss.item())

        if mask_gradients:
            _mask_gradients(x, cmos_mask)
        optimizer.step()

        # Clamping the values to be non-negative
        if non_neg:
            with torch.no_grad():
                x.copy_(x.data.clamp(min=0))

        progress_bar.set_description(log)

    _, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(spatial_loss_history)
    ax[0].set_title("Spatial Loss")
    ax[0].set_yscale("log")
    ax[1].plot(lambda_time_loss_history)
    ax[1].set_title("Lambda Time Loss")
    ax[1].set_yscale("log")
    ax[2].plot(global_loss_history)
    ax[2].set_title("Global Loss")
    ax[2].set_yscale("log")
    plt.show()

    if return_numpy:
        return (
            x.detach().cpu().numpy(),
            spc.detach().cpu().numpy(),
            cmos.detach().cpu().numpy(),
        )
    else:
        return x, spc, cmos
