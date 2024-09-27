import torch
import numpy as np
import torchvision.transforms.functional as f
from torchvision.transforms import Resize, InterpolationMode


class CosineLoss(torch.nn.Module):
    def __init__(self, dim=1, reduction="mean"):
        super(CosineLoss, self).__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        cos_sim = 1 - torch.nn.functional.cosine_similarity(pred, target, dim=self.dim)
        if self.reduction == "mean":
            return cos_sim.mean()
        else:
            return cos_sim.sum()


class MatrixCosineLoss(torch.nn.Module):
    def __init__(self, dim=0, reduction="mean"):
        super(MatrixCosineLoss, self).__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        # Element-wise product and then take the sum
        dot_product = torch.sum(pred * target, dim=(-2, -1))

        # Frobenius norms of both matrices
        norm_pred = torch.linalg.matrix_norm(pred, ord="fro")
        norm_target = torch.linalg.matrix_norm(target, ord="fro")

        # Prevent division by zero
        norm_pred = torch.where(norm_pred == 0, torch.ones_like(norm_pred), norm_pred)
        norm_target = torch.where(norm_target == 0, torch.ones_like(norm_target), norm_target)

        # Compute matrix-level cosine similarity
        cos_sim_matrix = dot_product / (norm_pred * norm_target)

        cos_sim_matrix = 1 - cos_sim_matrix
        if self.reduction == "mean":
            return cos_sim_matrix.mean(self.dim)
        else:
            return cos_sim_matrix.sum(self.dim)


def get_masks(cmos, spc):
    cmos_mask = cmos > (0.05 * cmos.max())
    spc_mask = f.resize(
        cmos_mask.any(dim=0).unsqueeze(0).float(),
        size=[spc.shape[-2], spc.shape[-1]],
        interpolation=InterpolationMode.BILINEAR,
        antialias=False,
    ).bool()
    return cmos_mask, spc_mask


def optimize(
        spc,
        cmos,
        iterations=30,
        lr=0.1,
        weights=(1, 1, 1),
        init_type="random",
        mask_noise=False,
        device="cpu",
        seed=42
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
    weights : tuple
        The weights to be used for the loss function. The order is:
        (spectral_loss, time_loss, spatial_loss, non_neg_loss, global_map_loss)
    init_type : str
        The initialization type to be used for the parameters to be optimized.
    mask_noise : bool
        Whether to apply a mask to void areas or not.
    device : str
        The device to be used for the optimization.
    seed : int
        The seed to be used for the random initialization of the data.

    Returns
    -------
    torch.Tensor
        The optimized spectral cube.
    """

    spc = torch.from_numpy(spc.astype(np.float32)).to(device)
    spc = torch.swapaxes(spc, 0, 1)  # (lambda,time,x,y)
    cmos = torch.from_numpy(cmos.astype(np.float32)).to(device)  # (z,x,y)

    n_lambdas = spc.shape[0]
    n_times = spc.shape[1]
    xy_dim = cmos.shape[1]
    z_dim = cmos.shape[0]
    x_shape = (n_lambdas, n_times, z_dim, xy_dim, xy_dim)

    if mask_noise:
        cmos_mask, spc_mask = get_masks(cmos, spc)
        cmos = cmos * cmos_mask.float()
        spc = spc * spc_mask.float()

    # Initialization
    torch.manual_seed(seed)
    if init_type == "random":
        x = torch.rand(x_shape).to(device)
    elif init_type == "zeros":
        x = torch.zeros(x_shape).to(device)
    else:
        raise ValueError("Invalid initialization type.")

    x = torch.nn.Parameter(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)

    cosine_spectral_time = MatrixCosineLoss().to(device)
    mse_spatial = torch.nn.MSELoss().to(device)
    mse_non_neg = torch.nn.MSELoss().to(device)

    down_sampler = Resize(
        size=(spc.shape[-2], spc.shape[-1]),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).to(device)

    for it in range(iterations):

        if mask_noise:
            with torch.no_grad():
                x[:, :, ~cmos_mask] = 0

        resized = torch.cat([down_sampler(torch.mean(torch.abs(xi), dim=1)).unsqueeze(0) for xi in x])

        spectral_time_loss = weights[0] * cosine_spectral_time(
            pred=torch.swapdims(spc.reshape(n_lambdas, n_times, -1), 0, -1),
            target=torch.swapdims(resized.reshape(n_lambdas, n_times, -1), 0, -1),
        )

        spatial_loss = weights[1] * mse_spatial(cmos.flatten(), torch.mean(torch.abs(x), dim=(0, 1)).flatten())
        non_neg_loss = weights[2] * mse_non_neg(x.flatten(), torch.nn.functional.relu(x.flatten()))

        loss = spectral_time_loss + spatial_loss + non_neg_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(
            f"Iteration {it + 1} | "
            f"SpectralTime: {spectral_time_loss.item():.4F} | "
            f"Spatial: {spatial_loss.item():.4F} | "
            f"Non Neg: {non_neg_loss.item():.4F} | "
        )

    return torch.swapaxes(x, 0, 1).detach().cpu().numpy()

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
# - To move back to summing one loss for the time and one for the spectrum, consider this code snippet:
#       spectral_loss = weights[0] * cosine_spectral(
#           pred=torch.mean(spc, dim=1).view(n_lambdas, -1).T,
#           target=torch.mean(resized, dim=1).view(n_lambdas, -1).T,
#       )
#
#       time_loss = weights[1] * cosine_time(
#           pred=torch.mean(spc, dim=0).view(n_times, -1).T,
#           target=torch.mean(resized, dim=0).view(n_times, -1).T,
#       )
