import numpy as np

import torch
from torchvision import transforms

class CosineDistanceLoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(CosineDistanceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, dim: int=1) -> float:
        if self.reduction == "mean":
            return (1 - torch.nn.functional.cosine_similarity(x1, x2, dim=dim)).mean()
        else: 
            return (1 - torch.nn.functional.cosine_similarity(x1, x2, dim=dim)).sum()

def optimize(spc, cmos, iterations=30, weights=(1,1,1,1), seed=42):
    spc = torch.from_numpy(spc.astype(np.float32))
    cmos = torch.from_numpy(cmos.astype(np.float32))

    n_times = spc.shape[0]
    n_lambdas = spc.shape[1]
    xy_dim = cmos.shape[1]
    z_dim = cmos.shape[0]

    # x = torch.zeros(n_times, n_lambdas, z_dim, xy_dim, xy_dim)
    torch.manual_seed(seed)
    x = torch.rand(n_times, n_lambdas, z_dim, xy_dim, xy_dim)
    
    x = torch.swapaxes(x, 0, 1)      # (lambda,time,z,x,y)
    spc = torch.swapaxes(spc, 0, 1)  # (lambda,time,x,y)

    down_sampler = torch.nn.AvgPool2d(4, 4)
    up_sampler = transforms.Resize((xy_dim, xy_dim), interpolation=transforms.InterpolationMode.BILINEAR)

    # weights_z = torch.mean(cmos, dim=(1, 2))
    # weights_z /= weights_z.max()

    # # Starting point
    # for zi in range(x.shape[2]):
    #     x[:, :, zi, :, :] = up_sampler(spc) * weights_z[zi]

    x = torch.nn.Parameter(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=0.1)

    spectral_fidelity = CosineDistanceLoss(reduction="mean")
    time_fidelity = CosineDistanceLoss(reduction="mean")
    spatial_fidelity = torch.nn.MSELoss(reduction="mean")
    non_neg_fidelity = torch.nn.MSELoss(reduction="mean")
    # global_lambda_fidelity = torch.nn.MSELoss(reduction="mean")

    for it in range(iterations):
        optimizer.zero_grad()

        x_flat = x.flatten()
        resized = torch.cat([down_sampler(torch.mean(xi, dim=1)).unsqueeze(0) for xi in x])

        spectral_loss = weights[0] * spectral_fidelity(
                    torch.mean(spc, dim=1).flatten(start_dim=1,end_dim=2).T, 
                    torch.mean(resized, dim=1).flatten(start_dim=1,end_dim=2).T
                )
        
        time_loss = weights[1] * time_fidelity(
                    torch.mean(spc, dim=0).flatten(start_dim=1,end_dim=2).T, 
                    torch.mean(resized, dim=0).flatten(start_dim=1,end_dim=2).T
                )
        
        spatial_loss = weights[2] * spatial_fidelity(cmos.flatten(), torch.mean(x, dim=(0, 1)).flatten())
        non_neg_loss = weights[3] * non_neg_fidelity(x_flat, torch.nn.functional.relu(x_flat))
        # global_lambda_loss =  weights[3] * global_lambda_fidelity(torch.mean(spc, dim=(2, 3)), torch.mean(x, dim=(2,3,4)))
        
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


# provare a mettere il termine global map normalizzando i dati su ogni fetta. Quando normalizzo farlo sull'area