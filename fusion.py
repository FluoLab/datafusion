import numpy as np

import torch
from torchvision import transforms


def optimize(spc, cmos, iterations=30):
    spc = torch.from_numpy(spc.astype(np.float32))
    cmos = torch.from_numpy(cmos.astype(np.float32))

    n_times = spc.shape[0]
    n_lambdas = spc.shape[1]
    xy_dim = cmos.shape[1]
    z_dim = cmos.shape[0]

    x = torch.zeros(n_times, n_lambdas, z_dim, xy_dim, xy_dim)

    x = torch.swapaxes(x, 0, 1)
    spc = torch.swapaxes(spc, 0, 1)

    down_sampler = torch.nn.AvgPool2d(4, 4)
    up_sampler = transforms.Resize((xy_dim, xy_dim), interpolation=transforms.InterpolationMode.BILINEAR)

    weights_z = torch.mean(cmos, dim=(1, 2))
    weights_z /= weights_z.max()

    # Starting point
    for zi in range(x.shape[2]):
        x[:, :, zi, :, :] = up_sampler(spc) * weights_z[zi]

    x = torch.nn.Parameter(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.1)

    spectral_fidelity = torch.nn.MSELoss(reduction="mean")
    spatial_fidelity = torch.nn.MSELoss(reduction="mean")
    non_neg_fidelity = torch.nn.MSELoss(reduction="mean")
    # spectral_slice_fidelity = torch.nn.MSELoss(reduction="mean")
    # intensity_fidelity = torch.nn.MSELoss(reduction="mean")
    # global_lambda_fidelity = torch.nn.MSELoss(reduction="mean")

    for it in range(iterations):
        optimizer.zero_grad()

        x_flat = x.flatten()
        resized = torch.cat([down_sampler(torch.mean(xi, dim=1)).unsqueeze(0) for xi in x])

        spectral_loss = spectral_fidelity(spc.flatten(), resized.flatten())
        spatial_loss = spatial_fidelity(cmos.flatten(), torch.mean(x, dim=(0, 1)).flatten())
        non_neg_loss = non_neg_fidelity(x_flat, torch.nn.functional.relu(x_flat))
        # global_lambda_loss =  global_lambda_fidelity(torch.mean(spc, dim=(1, 2)), torch.mean(x, dim=(1,2,3)))

        loss = spectral_loss + spatial_loss + non_neg_loss

        loss.backward()
        optimizer.step()

        print(
            f"Iteration {it + 1} | "
            f"Spectral: {spectral_loss.item():.4F} | "
            f"Spatial: {spatial_loss.item():.4F} | "
            f"Non Neg: {non_neg_loss.item():.4F} | "
            # f"Global: {global_lambda_loss.item():.4F}"
        )

    return torch.swapaxes(x, 0, 1)
