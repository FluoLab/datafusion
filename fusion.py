import numpy as np

import torch
import torch.nn.functional as f


def cosine_loss(
        pred: torch.Tensor, 
        target: torch.Tensor, 
        dim: int=1, 
        reduction="mean",
) -> float:
    cos_sim = 1 - torch.nn.functional.cosine_similarity(pred, target, dim=dim)
    if reduction == "mean":
        return cos_sim.mean()
    else: 
        return cos_sim.sum()


def optimize(spc, cmos, iterations=30, lr=0.1, weights=(1,1,1,1), seed=42):
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
        (spectral_loss, time_loss, spatial_loss, non_neg_loss)
    seed : int
        The seed to be used for the random initialization of the data.

    Returns
    -------
    torch.Tensor
        The optimized spectral cube.
    """

    spc = torch.from_numpy(spc.astype(np.float32))
    cmos = torch.from_numpy(cmos.astype(np.float32))

    n_times = spc.shape[0]
    n_lambdas = spc.shape[1]
    xy_dim = cmos.shape[1]
    z_dim = cmos.shape[0]

    torch.manual_seed(seed)
    x = torch.rand(n_times, n_lambdas, z_dim, xy_dim, xy_dim)
    
    x = torch.swapaxes(x, 0, 1)      # (lambda,time,z,x,y)
    spc = torch.swapaxes(spc, 0, 1)  # (lambda,time,x,y)

    x = torch.nn.Parameter(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)

    down_sampler_kernel_size = int(xy_dim / spc.shape[-1])
    down_sampler = torch.nn.AvgPool2d(down_sampler_kernel_size, down_sampler_kernel_size)

    for it in range(iterations):
        optimizer.zero_grad()
        
        x_flat = x.flatten()
        resized = torch.cat([down_sampler(torch.mean(xi, dim=1)).unsqueeze(0) for xi in x])

        spectral_loss = weights[0] * cosine_loss(
                    pred=torch.mean(spc, dim=1).view(n_lambdas, -1).T, 
                    target=torch.mean(resized, dim=1).view(n_lambdas, -1).T,
                )
        
        time_loss = weights[1] * cosine_loss(
                    pred=torch.mean(spc, dim=0).view(n_times, -1).T, 
                    target=torch.mean(resized, dim=0).view(n_times, -1).T,
                )
        
        spatial_loss = weights[2] * f.mse_loss(cmos.flatten(), torch.mean(x, dim=(0, 1)).flatten())
        non_neg_loss = weights[3] * f.mse_loss(x_flat, torch.nn.functional.relu(x_flat))
        # global_lambda_loss =  weights[4] * f.mse_loss(torch.mean(spc, dim=(2, 3)), torch.mean(x, dim=(2,3,4)))
        
        loss = spectral_loss + time_loss + spatial_loss + non_neg_loss

        loss.backward()
        optimizer.step()

        print(
            f"Iteration {it + 1} | "
            f"Spectral: {spectral_loss.item():.4F} | "
            f"Time: {time_loss.item():.4F} | "
            f"Spatial: {spatial_loss.item():.4F} | "
            f"Non Neg: {non_neg_loss.item():.4F} | "
            # f"Global: {global_lambda_loss.item():.4F}"
        )

    return torch.swapaxes(x, 0, 1)


# Comments:
# - Provare a mettere il termine global map normalizzando i dati su ogni fetta. Quando normalizzo farlo sull'area
# - Per inizializzare i dati considerare: 
#       x = torch.rand(n_times, n_lambdas, z_dim, xy_dim, xy_dim)
#       up_sampler = transforms.Resize((xy_dim, xy_dim), interpolation=transforms.InterpolationMode.BILINEAR)
#       weights_z = torch.mean(cmos, dim=(1, 2))
#       weights_z /= weights_z.max()
#       # Starting point
#       for zi in range(x.shape[2]):
#           x[:, :, zi, :, :] = up_sampler(spc) * weights_z[zi]