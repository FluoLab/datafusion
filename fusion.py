import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Resize, InterpolationMode
from torchvision.transforms.functional import resize


from losses import CosineLoss, DecayLoss


def _initialize(spc, cmos, x_shape, init_type, device, seed):
    torch.manual_seed(seed)
    if init_type == "random":
        x = (cmos.min() - cmos.max()) * torch.rand(x_shape) + cmos.max()
    elif init_type == "zeros":
        x = torch.zeros(x_shape).to(device)
    elif init_type == "baseline":
        from baseline import baseline

        x = baseline(cmos, spc, device, return_numpy=False)
    elif init_type == "upsampled_spc":
        upsampler = Resize(
            size=(cmos.shape[-2], cmos.shape[-1]),
            interpolation=InterpolationMode.NEAREST,
        ).to(device)
        x = upsampler(spc).unsqueeze(2).repeat(1, 1, cmos.shape[0], 1, 1)
        x = x / x.sum()
    else:
        raise ValueError("Invalid initialization type.")
    return x


def _initialize_continuous_time(x_shape, init_type, device, seed):
    torch.manual_seed(seed)
    if init_type == "random":
        x = torch.zeros(x_shape).to(device)
        x[:, 0, :, :, :] = (0.1 - 0.9) * torch.rand_like(x[:, 0, :, :, :]) + 0.9
        x[:, 1, :, :, :] = (0.1 - 5.0) * torch.rand_like(x[:, 1, :, :, :]) + 5.0
        x[:, 2, :, :, :] = (0.0001 - 0.01) * torch.rand_like(x[:, 1, :, :, :]) + 0.01
    elif init_type == "normal":
        x = torch.empty(x_shape).normal_(0.5, 0.01).to(device)
    elif init_type == "fixed":
        x = torch.zeros(x_shape, dtype=torch.float32).to(device)
        x[:, 0, :, :, :] = 0.5
        x[:, 1, :, :, :] = 2.5
        x[:, 2, :, :, :] = 0.0045
        # Add some noise to the initialization
        x[:, 0, :, :, :] += (
            torch.empty(x[:, 0, :, :, :].shape).normal_(0.0, 0.1).to(device)
        )
        x[:, 0, :, :, :] = x[:, 0, :, :, :].clamp(min=0.01, max=0.99)
        x[:, 1, :, :, :] += (
            torch.empty(x[:, 1, :, :, :].shape).normal_(0.0, 1.0).to(device)
        )
        x[:, 1, :, :, :] = x[:, 1, :, :, :].clamp(min=0.01, max=5.0)
        x[:, 2, :, :, :] += (
            torch.empty(x[:, 2, :, :, :].shape).normal_(0.0, 0.001).to(device)
        )
        x[:, 2, :, :, :] = x[:, 2, :, :, :].clamp(min=0.001, max=0.009)

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
    spc_mask = resize(
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
    torch.set_default_dtype(torch.DoubleTensor)

    spc = torch.from_numpy(spc.astype(np.float64)).to(device)
    spc = torch.swapaxes(spc, 0, 1)  # (lambda,time,x,y)
    cmos = torch.from_numpy(cmos.astype(np.float64)).to(device)  # (z,x,y)

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

    optimizer = torch.optim.NAdam([x], lr=lr)

    # Initialization of losses
    mse_spatial = torch.nn.MSELoss().to(device)
    mse_lambda_time = torch.nn.MSELoss().to(device)
    mse_global = torch.nn.MSELoss().to(device)

    down_sampler = torch.nn.LPPool2d(1, (4, 4), (4, 4))

    # down_sampler = Resize(
    #     size=(spc.shape[-2], spc.shape[-1]),
    #     interpolation=InterpolationMode.BILINEAR,
    #     antialias=True,
    # ).to(device)

    # up_sampler = Resize(
    #     size=(xy_dim, xy_dim),
    #     interpolation=InterpolationMode.BILINEAR,
    #     antialias=True,
    # ).to(device)
    #
    # up_sampled_spc = up_sampler(spc)

    for _ in (progress_bar := tqdm(range(iterations))):
        optimizer.zero_grad()
        # TODO: add termination condition based on convergence
        resized_x = torch.cat(
            [down_sampler(torch.sum(xi, dim=1)).unsqueeze(0) for xi in x]
        )

        # Computing losses
        spatial_loss = weights["spatial"] * mse_spatial(
            cmos.flatten(), torch.sum(x, dim=(0, 1)).flatten()
        )

        # spectral_loss = weights["spectral"] * cosine_spectral(
        #     pred=torch.mean(spc, dim=1).view(n_lambdas, -1).T,
        #     target=torch.mean(resized_x, dim=1).view(n_lambdas, -1).T,
        # )
        #
        # time_loss = weights["time"] * cosine_time(
        #     pred=torch.mean(spc, dim=0).view(n_times, -1).T,
        #     target=torch.mean(resized_x, dim=0).view(n_times, -1).T,
        # )

        lambda_time_loss = weights["lambda_time"] * mse_lambda_time(
            spc.flatten(),
            resized_x.flatten(),
        )

        global_loss = weights["global"] * mse_global(
            spc.sum(dim=(2, 3)).flatten(),
            resized_x.sum(dim=(2, 3)).flatten(),
        )

        loss = spatial_loss + lambda_time_loss + global_loss
        log = (
            f"Spatial: {spatial_loss.item()} | "
            f"Lambda Time: {lambda_time_loss.item()} | "
            f"Global: {global_loss.item()} | "
        )

        loss.backward()

        if mask_gradients:
            _mask_gradients(x, cmos_mask)
        optimizer.step()

        # Clamping the values to be non-negative
        if non_neg:
            with torch.no_grad():
                x.data.clamp_(min=0.0)

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
    spc = torch.from_numpy(spc.astype(np.float32)).to(device)
    spc = torch.swapaxes(spc, 0, 1)  # (lambda,time,x,y)
    cmos = torch.from_numpy(cmos.astype(np.float32)).to(device)  # (z,x,y)

    # spc_energy = spc.sum()
    # cmos_energy = cmos.sum()
    # spc = spc * (cmos_energy / spc_energy)

    n_lambdas = spc.shape[0]
    n_times = spc.shape[1]
    xy_dim = cmos.shape[1]
    z_dim = cmos.shape[0]
    x_shape = (n_lambdas, 3 * n_decays, z_dim, xy_dim, xy_dim)
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t.astype(np.float32)).to(device)

    # Initialization of the parameters
    x = _initialize_continuous_time(x_shape, init_type, device, seed)
    spc_mask, cmos_mask = _get_masks(spc, cmos)
    if mask_initializations:
        spc, cmos, x = _mask_initializations(x, spc, cmos, spc_mask, cmos_mask)
    x = torch.nn.Parameter(x, requires_grad=True)

    optimizer = torch.optim.Adam([x], lr=lr)

    # Initialization of losses
    cosine_spectral = CosineLoss().to(device)
    mse_spatial = torch.nn.MSELoss().to(device)
    mse_decay = DecayLoss(t).to(device)
    # mse_decay = torch.nn.MSELoss().to(device)

    # Down-sampler to resize the cmos xy dimensions to the xy dimensions of the spc
    down_sampler = Resize(
        size=(spc.shape[-2], spc.shape[-1]),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).to(device)

    # down_sampler = torch.nn.LPPool2d(1, (4, 4), (4, 4))

    # up_sampler = Resize(
    #     size=(xy_dim, xy_dim),
    #     interpolation=InterpolationMode.BILINEAR,
    #     antialias=True,
    # ).to(device)
    #
    # up_sampled_spc = up_sampler(spc)

    for _ in (progress_bar := tqdm(range(iterations))):
        optimizer.zero_grad()

        # discrete_x = x.swapaxes(0, 1).reshape(3, -1)
        # discrete_x = mono_exponential_decay_torch(discrete_x[0], discrete_x[1], discrete_x[2], t)
        # discrete_x = discrete_x.reshape(n_times, n_lambdas, z_dim, xy_dim, xy_dim).swapaxes(0, 1)
        # resized_x = torch.cat([down_sampler(torch.mean(xi, dim=1)).unsqueeze(0) for xi in discrete_x])
        resized_x = torch.cat(
            [down_sampler(torch.mean(xi, dim=1)).unsqueeze(0) for xi in x]
        )

        spatial_loss = weights["spatial"] * mse_spatial(
            cmos.flatten(), torch.mean(x, dim=(0, 1)).flatten()
        )

        spectral_loss = weights["spectral"] * cosine_spectral(
            pred=torch.mean(resized_x, dim=1).view(n_lambdas, -1).T,
            target=torch.mean(spc, dim=1).view(n_lambdas, -1).T,
        )

        # time_loss = weights["time"] * mse_decay(
        #     resized_x.mean(dim=0).flatten(),
        #     spc.mean(dim=0).flatten()
        # )

        time_loss = weights["time"] * mse_decay(
            pred_coeffs=torch.mean(resized_x, dim=0).view(3 * n_decays, -1).T,
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

        with torch.no_grad():
            # Clamp I
            x.data[:, 0, :, :, :].clamp_(max=1.1)
            # Clamp c
            x.data[:, 2, :, :, :].clamp_(max=0.01)

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
