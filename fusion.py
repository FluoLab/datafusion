import torch
import numpy as np
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


def optimize(spc, cmos, iterations=30, lr=0.1, weights=(1, 1, 1, 1, 1, 1), device="cpu", seed=42):
    """
    Parameters
    ----------
    spc : np.ndarray
        The spectral cube to be used for the optimization.
    cmos : np.ndarray
        The CMOS data to be used for the optimization.
    iterations : int
        The number of iterations to run the optimization.
    lr : float
        The learning rate to be used for the optimization.
    weights : tuple
        The weights to be used for the loss function. The order is:
        (spectral_loss, time_loss, spatial_loss, non_neg_loss, global_map_loss)
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
    cmos = torch.from_numpy(cmos.astype(np.float32)).to(device)
    
    # maskcmos = cmos > (0.05 * torch.max(cmos))
    # cmos = cmos * maskcmos

    n_times = spc.shape[0]
    n_lambdas = spc.shape[1]
    xy_dim = cmos.shape[1]
    z_dim = cmos.shape[0]

    torch.manual_seed(seed)
    x = torch.rand(n_times, n_lambdas, z_dim, xy_dim, xy_dim).to(device)

    x = torch.swapaxes(x, 0, 1)  # (lambda,time,z,x,y)
    spc = torch.swapaxes(spc, 0, 1)  # (lambda,time,x,y)
    
    # mask = maskcmos.unsqueeze(0).unsqueeze(0).repeat(n_lambdas, n_times, 1, 1, 1).int().to(device)
    # mask.requires_grad(True)

    x = torch.nn.Parameter(x)
    x.data.requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    cosine_spectral = CosineLoss().to(device)
    cosine_time = CosineLoss().to(device)
    mse_spatial = torch.nn.MSELoss().to(device)
    mse_non_neg = torch.nn.MSELoss().to(device)
    mse_global_map = torch.nn.MSELoss().to(device)
    mse_intensity = torch.nn.MSELoss().to(device)

    down_sampler = Resize(
        size=(spc.shape[-2], spc.shape[-1]),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).to(device)

    spectrum_time = torch.mean(spc, dim=(2, 3))
    spectrum_time = spectrum_time / torch.sum(spectrum_time)

    for it in range(iterations):
        resized = torch.cat([down_sampler(torch.mean(xi, dim=1)).unsqueeze(0) for xi in x])

        spectral_loss = weights[0] * cosine_spectral(
            pred=torch.mean(torch.abs(spc), dim=1).view(n_lambdas, -1).T,
            target=torch.mean(torch.abs(resized), dim=1).view(n_lambdas, -1).T,
        )

        time_loss = weights[1] * cosine_time(
            pred=torch.mean(torch.abs(spc), dim=0).view(n_times, -1).T,
            target=torch.mean(torch.abs(resized), dim=0).view(n_times, -1).T,
        )

        spatial_loss = weights[2] * mse_spatial(torch.abs(cmos.flatten()), torch.mean(torch.abs(x), dim=(0, 1)).flatten())
        non_neg_loss = weights[3] * mse_non_neg(x.flatten(), torch.nn.functional.relu(x.flatten()))
        global_map_loss = weights[4] * mse_global_map(spectrum_time, torch.mean(x, dim=(2, 3, 4)))
        # spectral_loss = weights[0] * f.mse_loss(spc.flatten(), resized.flatten())
        
        intensity = weights[5] * mse_intensity(torch.mean(cmos, dim=(1,2)), torch.mean(x, dim=(0,1,3,4)))

        loss = spectral_loss + time_loss + spatial_loss + intensity + global_map_loss# + non_neg_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(
            f"Iteration {it + 1} | "
            f"Spectral: {spectral_loss.item():.4F} | "
            f"Time: {time_loss.item():.4F} | "
            f"Spatial: {spatial_loss.item():.4F} | "
            f"Non Neg: {non_neg_loss.item():.4F} | "
            f"Global: {global_map_loss.item():.4F} | "
            f"Intensity: {intensity.item():.4F} | "
        )

    return torch.swapaxes(x, 0, 1).detach().cpu().numpy()

# Comments:
# - The global-map term is useful if we use the MSE for the time and spectrum to remove the offset. Using the cosine 
#   loss the global-map term is not needed. This term may be useful in the gradient descent method to keep the loss
#   convex, since the cosine loss is not convex.
# - To initialize the x consider also the following code snippet:
#       x = torch.rand(n_times, n_lambdas, z_dim, xy_dim, xy_dim)
#       up_sampler = transforms.Resize((xy_dim, xy_dim), interpolation=transforms.InterpolationMode.BILINEAR)
#       weights_z = torch.mean(cmos, dim=(1, 2))
#       weights_z /= weights_z.max()
#       # Starting point
#       for zi in range(x.shape[2]):
#           x[:, :, zi, :, :] = up_sampler(spc) * weights_z[zi]
