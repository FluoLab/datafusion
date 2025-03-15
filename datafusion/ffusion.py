import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torch.nn.functional import conv2d, conv_transpose2d
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from datafusion.baseline import baseline
from datafusion.utils import time_volume_to_lifetime


def squared_l2(x):
    return torch.sum(x**2)


class IntegralOperator:
    def __init__(self, size=1, integral_dim=0):
        self.size = size
        self.integral_dim = integral_dim

    def A(self, x):
        self.size = x.shape[self.integral_dim]
        return x.sum(dim=self.integral_dim, keepdim=True)

    def A_adjoint(self, y):
        return y.repeat_interleave(self.size, dim=self.integral_dim) / self.size

    def __call__(self, *args):
        return self.A(*args)


class SumPoolOperator:
    def __init__(self, size=4, channels=16, device="cpu"):
        self.size = size
        self.kernel = torch.ones(channels, 1, size, size, device=device)
        self.channels = channels

    def A(self, x):
        x = x.squeeze(2)
        y = conv2d(x, self.kernel, stride=self.size, groups=self.channels, bias=None)
        y = y.unsqueeze(2)
        return y

    def A_adjoint(self, y):
        y = y.squeeze(2)
        x = conv_transpose2d(y, self.kernel, stride=self.size, groups=self.channels, bias=None)
        x = x.unsqueeze(2)
        return x / self.size**2

    def __call__(self, *args):
        return self.A(*args)


class MonoDecayOperator:
    def __init__(self, t: torch.Tensor):
        self.t = t.view(-1, 1, 1, 1, 1)
        self.eps = 1e-7

    def A(self, x):
        # x is assumed to be of shape (3, n_lambdas, z_dim, x_dim, y_dim)
        # x[0] is the amplitude
        # x[1] is the lifetime
        # x[2] is the background
        return x[0:1] * torch.exp(- self.t / (x[1:2] + self.eps)) + x[2:3]

    def __call__(self, *args):
        return self.A(*args)


class Fusion:
    def __init__(
        self,
        spc: np.ndarray | torch.Tensor,
        cmos: np.ndarray | torch.Tensor,
        time_axis: np.ndarray,
        weights: dict,
        init_type: str,
        tol: float | None = 1e-6,
        mask_noise: bool = False,
        total_energy: float = 1.0,
        device: str = "cpu",
        seed: int = 42,
    ):
        if isinstance(spc, np.ndarray):
            # spc: (time,lambda,x,y)
            self.spc = torch.from_numpy(spc.astype(np.float32)).to(device)
        if isinstance(cmos, np.ndarray):
            # cmos: (z,x,y)
            self.cmos = torch.from_numpy(cmos.astype(np.float32)).to(device)

        self.t = torch.from_numpy(time_axis.astype(np.float32)).to(device)
        self.weights = weights
        self.init_type = init_type
        self.tol = tol
        self.seed = seed
        self.mask_noise = mask_noise
        self.device = device
        self.n_times = spc.shape[0]
        self.n_lambdas = spc.shape[1]
        self.xy_dim = cmos.shape[1]
        self.z_dim = cmos.shape[0]
        self.x_shape = (
            3,
            spc.shape[1],
            cmos.shape[0],
            cmos.shape[1],
            cmos.shape[2],
        )
        self.spatial_increase = cmos.shape[-1] // spc.shape[-1]
        self.prev_x = None

        self.T = IntegralOperator(size=self.n_times, integral_dim=0)
        self.S = IntegralOperator(size=self.n_lambdas, integral_dim=1)
        self.D = IntegralOperator(size=self.z_dim, integral_dim=2)
        self.R = SumPoolOperator(
            size=self.spatial_increase, channels=self.n_lambdas, device=self.device
        )
        self.E = MonoDecayOperator(self.t)

        # Get the masks on input tensor to not optimize the background noise
        self.spc_mask, self.cmos_mask = self._get_masks()

        # Mask
        if self.mask_noise:
            self.spc = self.spc * self.spc_mask.float()
            self.cmos = self.cmos * self.cmos_mask.float()

        self.spc_mask = self.spc_mask.squeeze(0)

        # Normalize the energy of the input data (on the mask if necessary)
        # self.spc = self.spc + self.spc.min()
        self.spc = self.normalize_energy(self.spc, total_energy)
        self.cmos = self.normalize_energy(self.cmos, total_energy)

        # Initialize the parameters
        self.x = self._initialize()
        if self.mask_noise:
            self.x[:, :, ~self.cmos_mask] = 0.0
        self.x.requires_grad = True

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @staticmethod
    def normalize_energy(tensor, total_energy=1):
        return total_energy * tensor / tensor.sum()

    def loss(self):
        expanded_x = self.E(self.x)

        if self.mask_noise:
            x1 = self.cmos[self.cmos_mask]
            x2 = self.T(self.S(expanded_x))[0, 0, self.cmos_mask]
            spatial_loss = self.weights["spatial"] * squared_l2(x1 - x2)

            x1 = self.spc[:, :, self.spc_mask]
            x2 = self.R(self.D(expanded_x)).squeeze(2)[:, :, self.spc_mask]
            lambda_time_loss = self.weights["lambda_time"] * squared_l2(x1 - x2)

        else:
            x1 = self.cmos.flatten()
            x2 = self.T(self.S(expanded_x)).flatten()
            spatial_loss = self.weights["spatial"] * squared_l2(x1 - x2)

            x1 = self.spc.flatten()
            x2 = self.R(self.D(expanded_x)).squeeze(2).flatten()
            lambda_time_loss = self.weights["lambda_time"] * squared_l2(x1 - x2)

        return spatial_loss, lambda_time_loss

    def sensitivity(self):
        return torch.linalg.vector_norm(self.x.flatten() - self.prev_x.flatten())

    def _initialize(self):
        torch.manual_seed(self.seed)

        if self.init_type == "baseline":
            x = torch.zeros(self.x_shape)
            base = baseline(self.cmos, self.spc, device="cpu", return_numpy=True)
            for ch in range(base.shape[1]):
                a, tau, c = time_volume_to_lifetime(
                    self.t, base[:, ch, :, :], tau_clip=(0, 10), max_tau=6.0, noise_thr=0.2, return_all=True
                )
                x[0, ch, :, :] = torch.from_numpy(a)
                x[1, ch, :, :] = torch.from_numpy(tau)
                x[2, ch, :, :] = torch.from_numpy(c)
            x = x.to(self.device)

        elif self.init_type == "random":
            x = torch.rand(self.x_shape).to(self.device)

        elif self.init_type == "simple":
            x = torch.zeros(self.x_shape, dtype=torch.float32).to(self.device)
            x[0] = torch.normal(0.5, 0.1, size=self.x_shape[1:]).to(self.device)
            x[1] = torch.normal(2.0, 0.1, size=self.x_shape[1:]).to(self.device)
            x[2] = torch.normal(1e-4, 1e-6, size=self.x_shape[1:]).to(self.device)
            x[0] = torch.clamp(x[0], min=0)
            x[1] = torch.clamp(x[1], min=1e-4, max=10)
            x[2] = torch.clamp(x[2], min=0, max=0.2)

        elif self.init_type == "zeros":
            x = torch.zeros(self.x_shape, dtype=torch.float32).to(self.device)

        else:
            raise ValueError("Invalid initialization type.")

        return x

    def _mask_gradients(self):
        self.x.grad[:, :, ~self.cmos_mask] = 0.0

    def _get_masks(self):
        cmos_mask = self.cmos > (0.05 * self.cmos.max())
        spc_mask = resize(
            cmos_mask.any(dim=0).unsqueeze(0).float(),
            size=[self.spc.shape[-2], self.spc.shape[-1]],
            interpolation=InterpolationMode.BILINEAR,
            antialias=False,
        ).bool()
        return spc_mask, cmos_mask


class FusionAdam(Fusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        lr: float,
        max_iterations: int,
        non_neg: bool = False,
        return_numpy: bool = True,
    ):

        history = np.zeros((max_iterations, len(self.weights) + 1))

        self.x = torch.nn.Parameter(self.x, requires_grad=True)
        optimizer = torch.optim.Adam([self.x], lr=lr, amsgrad=False)

        for i in (progress_bar := tqdm(range(max_iterations))):
            self.prev_x = self.x.detach().clone() if self.tol is not None else None
            optimizer.zero_grad()

            spatial_loss, lambda_time_loss = self.loss()
            loss = spatial_loss + lambda_time_loss
            loss.backward()

            if self.mask_noise:
                self._mask_gradients()
            optimizer.step()

            if non_neg:
                with torch.no_grad():
                    self.x.copy_(self.x.data.clamp(min=0))

            sensitivity = self.sensitivity().item() if self.tol is not None else None

            progress_bar.set_description(
                f"Spatial: {spatial_loss.item():.2E} | "
                f"Lambda Time: {lambda_time_loss.item():.2E} | "
                f"Total: {loss.item():.2E} | "
                f"Sensitivity: {f'{sensitivity:.2E}' if sensitivity is not None else 'Not considered'} | "
                # f"Grad Norm: {self.x.grad.data.norm(2).item():.2E}"
            )

            if sensitivity is not None and sensitivity < self.tol:
                break

            history[i] = np.array(
                [
                    spatial_loss.item(),
                    lambda_time_loss.item(),
                    loss.item(),
                    # global_loss.item(),
                ]
            )

        _, ax = plt.subplots(1, history.shape[1], figsize=(4 * history.shape[1], 4))
        for i, title in enumerate(["Spatial", "Lambda Time", "Total"]):
            ax[i].scatter(np.arange(len(history[:, i])), history[:, i], marker=".")
            ax[i].set_title(title)
            ax[i].set_yscale("log")
        plt.tight_layout()
        plt.show()

        expanded = self.E(self.x)

        if return_numpy:
            return (
                expanded.detach().cpu().numpy(),
                self.spc.detach().cpu().numpy(),
                self.cmos.detach().cpu().numpy(),
                self.x.detach().cpu().numpy(),
            )
        else:
            return expanded, self.spc, self.cmos, self.x

