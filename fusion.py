import torch
import numpy as np
import torchvision.transforms.functional as f

from tqdm import tqdm
from torchvision.transforms import Resize, InterpolationMode

from losses import CosineLoss, MatrixCosineLoss, DecayLoss


def _initialize(spc, cmos, x_shape, init_type, device, seed):
    torch.manual_seed(seed)
    if init_type == "random":
        x = torch.rand(x_shape).to(device)
    elif init_type == "normal":
        x = torch.empty(x_shape).normal_(0.1, 0.05).to(device)
    elif init_type == "zeros":
        x = torch.zeros(x_shape).to(device)
    elif init_type == "baseline":
        from baseline import baseline
        x = baseline(cmos, spc, device, return_numpy=False)
    else:
        raise ValueError("Invalid initialization type.")
    return x


def _mask_initializations(x, spc, cmos, spc_mask, cmos_mask):
    spc = spc * spc_mask.float()
    cmos = cmos * cmos_mask.float()
    x[:, :, ~cmos_mask] = 0.0
    return spc, cmos, x


def _mask_gradients(x, cmos_mask):
    x.grad[:, :, ~cmos_mask] = 0.0


def _get_masks(spc, cmos):
    cmos_mask = cmos > (0.05 * cmos.max())
    spc_mask = f.resize(
        cmos_mask.any(dim=0).unsqueeze(0).float(),
        size=[spc.shape[-2], spc.shape[-1]],
        interpolation=InterpolationMode.BILINEAR,
        antialias=False,
    ).bool()
    return spc_mask, cmos_mask


def optimize(
        spc,
        cmos,
        weights,
        iterations=30,
        lr=0.1,
        init_type="random",
        mask_initializations=False,
        mask_gradients=False,
        non_neg=False,
        device="cpu",
        seed=42,
        return_numpy=True,
):
    """
    Parameters
    ----------
    spc : np.ndarray
        The spectral cube to be used for the optimization. (time,lambda,x,y)
    cmos : np.ndarray
        The CMOS data to be used for the optimization. (z,x,y)
    iterations : int
        The number of iterations to run the optimization.
    lr : float
        The learning rate to be used for the optimization.
    weights : dict
        The weights to be used for the loss function.
    init_type : str
        The initialization type to be used for the parameters to be optimized.
    mask_initializations : bool
        Whether to mask the initializations of the parameters.
    mask_gradients : bool
        Whether to mask the gradients of the parameters.
    non_neg : bool
        Whether to enforce non-negativity on the parameters.
    device : str
        The device to be used for the optimization.
    seed : int
        The seed to be used for the random initialization of the data.
    return_numpy : bool
        Whether to return the optimized data as a numpy array or as a tensor.

    Returns
    -------
    torch.Tensor
        The optimized spectral cube.
    """

    # TODO: clean the logic of the code
    # TODO: fix readability of the code

    spc = torch.from_numpy(spc.astype(np.float32)).to(device)
    spc = torch.swapaxes(spc, 0, 1)  # (lambda,time,x,y)
    cmos = torch.from_numpy(cmos.astype(np.float32)).to(device)  # (z,x,y)

    n_lambdas = spc.shape[0]
    n_times = spc.shape[1]
    xy_dim = cmos.shape[1]
    z_dim = cmos.shape[0]
    x_shape = (n_lambdas, n_times, z_dim, xy_dim, xy_dim)

    # Initialization of the parameters
    x = _initialize(spc, cmos, x_shape, init_type, device, seed)
    spc_mask, cmos_mask = _get_masks(spc, cmos)
    if mask_initializations:
        spc, cmos, x = _mask_initializations(x, spc, cmos, spc_mask, cmos_mask)
    x = torch.nn.Parameter(x, requires_grad=True)

    optimizer = torch.optim.Adam([x], lr=lr)

    # Initialization of losses
    mse_spatial = torch.nn.MSELoss().to(device)
    if weights["spectral_time"] > 0:
        cosine_spectral_time = MatrixCosineLoss().to(device)
    else:
        cosine_spectral = CosineLoss().to(device)
        cosine_time = CosineLoss().to(device)

    # Down-sampler to resize the cmos xy dimensions to the xy dimensions of the spc
    down_sampler = Resize(
        size=(spc.shape[-2], spc.shape[-1]),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).to(device)

    for _ in (progress_bar := tqdm(range(iterations))):
        # TODO: add termination condition based on convergence
        resized = torch.cat([down_sampler(torch.mean(xi, dim=1)).unsqueeze(0) for xi in x])

        # Computing losses
        spatial_loss = weights["spatial"] * mse_spatial(cmos.flatten(), torch.mean(x, dim=(0, 1)).flatten())

        if weights["spectral_time"] > 0:
            spectral_time_loss = weights["spectral_time"] * cosine_spectral_time(
                pred=torch.swapdims(spc.reshape(n_lambdas, n_times, -1), 0, -1),
                target=torch.swapdims(resized.reshape(n_lambdas, n_times, -1), 0, -1),
            )
            loss = spectral_time_loss + spatial_loss
            log = (
                f"Spatial: {spatial_loss.item():.4F} | "
                f"SpectralTime: {spectral_time_loss.item():.4F} | "
            )

        else:
            spectral_loss = weights["spectral"] * cosine_spectral(
                pred=torch.mean(spc, dim=1).view(n_lambdas, -1).T,
                target=torch.mean(resized, dim=1).view(n_lambdas, -1).T,
            )

            time_loss = weights["time"] * cosine_time(
                pred=torch.mean(spc, dim=0).view(n_times, -1).T,
                target=torch.mean(resized, dim=0).view(n_times, -1).T,
            )

            loss = spatial_loss + spectral_loss + time_loss
            log = (
                f"Spatial: {spatial_loss.item():.4F} | "
                f"Spectral: {spectral_loss.item():.4F} | "
                f"Time: {time_loss.item():.4F} | "
            )

        loss.backward()

        if mask_gradients:
            _mask_gradients(x, cmos_mask)
        optimizer.step()

        # Clamping the values to be non-negative
        if non_neg:
            with torch.no_grad():
                x.data.clamp_(min=0.0)

        optimizer.zero_grad()
        progress_bar.set_description(log)

    x = torch.swapaxes(x, 0, 1)
    return x.detach().cpu().numpy() if return_numpy else x.detach()


def optimize_with_continuous_time(
        spc,
        cmos,
        weights,
        t,
        n_decays=1,
        iterations=30,
        lr=0.1,
        init_type="random",
        mask_initializations=False,
        mask_gradients=False,
        non_neg=False,
        device="cpu",
        seed=42,
        return_numpy=True,
):
    # Idea:
    #       -- Use the optimization method to also find the decay of the exponential decay model.

    #       -- I_0 * e^(-t / tau_0) + I_1 * e^(-t / tau_1) +  ... + I_n * e^(-t / tau_n)

    #       -- The coefficients (I_0, tau_0, ...) are now the elements in the x tensor for the time dimension.

    #       -- To jointly optimize x and perform regression on the decay parameters we can change the time loss to MSE
    #          between the spc and the resized version of x.

    #       -- The decay parameters have to be carefully integrated now in a separate resize though.

    #       -- TODO: Think about better ways to integrate the decay parameters in the optimization for the resize.

    spc = torch.from_numpy(spc.astype(np.float32)).to(device)
    spc = torch.swapaxes(spc, 0, 1)  # (lambda,time,x,y)
    cmos = torch.from_numpy(cmos.astype(np.float32)).to(device)  # (z,x,y)

    n_lambdas = spc.shape[0]
    n_times = spc.shape[1]
    xy_dim = cmos.shape[1]
    z_dim = cmos.shape[0]
    x_shape = (n_lambdas, 2 * n_decays, z_dim, xy_dim, xy_dim)

    # Initialization of the parameters
    x = _initialize(spc, cmos, x_shape, init_type, device, seed)
    spc_mask, cmos_mask = _get_masks(spc, cmos)
    if mask_initializations:
        spc, cmos, x = _mask_initializations(x, spc, cmos, spc_mask, cmos_mask)
    x = torch.nn.Parameter(x, requires_grad=True)

    optimizer = torch.optim.Adam([x], lr=lr)

    # Initialization of losses
    cosine_spectral = CosineLoss().to(device)
    mse_spatial = torch.nn.MSELoss().to(device)
    mse_decay = DecayLoss(t).to(device)

    # Down-sampler to resize the cmos xy dimensions to the xy dimensions of the spc
    down_sampler = Resize(
        size=(spc.shape[-2], spc.shape[-1]),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).to(device)

    for _ in (progress_bar := tqdm(range(iterations))):
        resized = torch.cat([down_sampler(torch.mean(xi, dim=1)).unsqueeze(0) for xi in x])

        spatial_loss = weights["spatial"] * mse_spatial(cmos.flatten(), torch.mean(x, dim=(0, 1)).flatten())
        spectral_loss = weights["spectral"] * cosine_spectral(
            pred=torch.mean(resized, dim=1).view(n_lambdas, -1).T,
            target=torch.mean(spc, dim=1).view(n_lambdas, -1).T,
        )

        time_loss = weights["time"] * mse_decay(
            pred_coeffs=torch.mean(resized, dim=0).view(2 * n_decays, -1).T,
            target=torch.mean(spc, dim=0).view(n_times, -1).T,
        )

        loss = spectral_loss + time_loss + spatial_loss

        log = (
            f"Spatial: {spatial_loss.item():.4F} | "
            f"Spectral: {spectral_loss.item():.4F} | "
            f"Time: {time_loss.item():.4F} | "
        )

        loss.backward()

        if mask_gradients:
            x.grad[:, :, ~cmos_mask] = 0.0

        optimizer.step()

        if non_neg:
            with torch.no_grad():
                x.data.clamp_(min=0.0)

        optimizer.zero_grad()
        progress_bar.set_description(log)

    x = torch.swapaxes(x, 0, 1)
    return x.detach().cpu().numpy() if return_numpy else x.detach()

# Comments:

# - The global-map term is useful if we use the MSE for the time and spectrum to remove the offset. Using the cosine 
#   loss the global-map term is not needed. This term may be useful in the gradient descent method to keep the loss
#   convex, since the cosine loss is not convex.
#
# - To initialize the x consider also the following code snippet:
#       x = torch.rand(n_times, n_lambdas, z_dim, xy_dim, xy_dim)
#       up_sampler = transforms.Resize((xy_dim, xy_dim), interpolation=transforms.InterpolationMode.BILINEAR)
#       weights_z = torch.mean(cmos, dim=(1, 2))
#       weights_z /= weights_z.max()
#       # Starting point
#       for zi in range(x.shape[2]):
#           x[:, :, zi, :, :] = up_sampler(spc) * weights_z[zi]
#
