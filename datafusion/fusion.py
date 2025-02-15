import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torch.nn.functional import conv2d, conv_transpose2d
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from datafusion.baseline import baseline


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
        x = conv_transpose2d(
            y, self.kernel, stride=self.size, groups=self.channels, bias=None
        )
        x = x.unsqueeze(2)
        return x / self.size**2

    def __call__(self, *args):
        return self.A(*args)


class Fusion:
    def __init__(
        self,
        spc: np.ndarray | torch.Tensor,
        cmos: np.ndarray | torch.Tensor,
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
            spc.shape[0],
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

        # Get the masks on input tensor to not optimize the background noise
        self.spc_mask, self.cmos_mask = self._get_masks()

        # Mask
        if self.mask_noise:
            self.spc = self.spc * self.spc_mask.float()
            self.cmos = self.cmos * self.cmos_mask.float()

        self.spc_mask = self.spc_mask.squeeze(0)

        # Normalize the energy of the input data (on the mask if necessary)
        self.spc = self.normalize_energy(self.spc, total_energy)
        self.cmos = self.normalize_energy(self.cmos, total_energy)

        # Initialize the parameters
        self.x = self._initialize()
        if self.mask_noise:
            self.x[:, :, ~self.cmos_mask] = 0.0
        self.x = self.normalize_energy(self.x, total_energy)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    @staticmethod
    def normalize_energy(tensor, total_energy=1):
        return total_energy * tensor / tensor.sum()

    def loss(self):
        if self.mask_noise:
            x1 = self.cmos[self.cmos_mask]
            x2 = self.T(self.S(self.x))[0, 0, self.cmos_mask]
            spatial_loss = self.weights["spatial"] * squared_l2(x1 - x2)

            x1 = self.spc[:, :, self.spc_mask]
            x2 = self.R(self.D(self.x)).squeeze(2)[:, :, self.spc_mask]
            lambda_time_loss = self.weights["lambda_time"] * squared_l2(x1 - x2)

        else:
            x1 = self.cmos.flatten()
            x2 = self.T(self.S(self.x)).flatten()
            spatial_loss = self.weights["spatial"] * squared_l2(x1 - x2)

            x1 = self.spc.flatten()
            x2 = self.R(self.D(self.x)).squeeze(2).flatten()
            lambda_time_loss = self.weights["lambda_time"] * squared_l2(x1 - x2)

        # There is a possible global loss that can be added to the total loss.
        # We notice that adding it does not improve the results.
        # Global has no spatial dimension, so no need to mask.
        # x1 = self.spc.sum(dim=(2, 3)).flatten()
        # x2 = self.x.sum(dim=(2, 3, 4)).flatten()
        # global_loss = weights["global"] * squared_l2(x1 - x2)

        return spatial_loss, lambda_time_loss

    def sensitivity(self):
        return torch.linalg.vector_norm(self.x.flatten() - self.prev_x.flatten())

    def _initialize(self):
        torch.manual_seed(self.seed)
        if self.init_type == "random":
            x = (self.cmos.min() - self.cmos.max()) * torch.rand(
                self.x_shape, device=self.device
            ) + self.cmos.max()
        elif self.init_type == "zeros":
            x = torch.zeros(self.x_shape).to(self.device)
        elif self.init_type == "baseline":
            x = baseline(self.cmos, self.spc, device="cpu", return_numpy=False)
            x = x.to(self.device)
        elif self.init_type == "upscaled_spc":
            x = resize(
                self.spc,
                size=self.cmos.shape[-2:],
                interpolation=InterpolationMode.NEAREST,
            )
            x = x.unsqueeze(2).repeat(1, 1, self.cmos.shape[0], 1, 1)
            x = x.to(self.device)
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
            loss = spatial_loss + lambda_time_loss  # + global_loss

            # times = self.x.sum(dim=1).reshape(self.x.shape[0], -1)
            # regularization = 1e-3 * torch.linalg.svdvals(times).sum()
            # loss += regularization
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
                # f"Reg: {regularization.item():.2E} | "
                # f"Global: {global_loss.item():.2E} | "
                f"Total: {loss.item():.2E} | "
                f"Sensitivity: {f'{sensitivity:.2E}' if sensitivity is not None else 'Not considered'} | "
                # f"Grad Norm: {x.grad.data.norm(2).item():.2E}"
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

        if return_numpy:
            return (
                self.x.detach().cpu().numpy(),
                self.spc.detach().cpu().numpy(),
                self.cmos.detach().cpu().numpy(),
            )
        else:
            return self.x, self.spc, self.cmos


class FusionCG(Fusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        max_iterations: int,
        eps: float = 1e-10,
        return_numpy: bool = True,
    ):
        history = np.zeros((max_iterations, len(self.weights) + 2))

        A = lambda x: self.T.A_adjoint(
            self.S.A_adjoint(self.S(self.T(x)))
        ) + self.D.A_adjoint(self.R.A_adjoint(self.R(self.D(x))))
        b = self.T.A_adjoint(
            self.S.A_adjoint(self.cmos.unsqueeze(0).unsqueeze(0))
        ) + self.D.A_adjoint(self.R.A_adjoint(self.spc.unsqueeze(2)))

        # Slightly modified version of the conjugate gradient method from:
        # https://deepinv.github.io/deepinv/_modules/deepinv/optim/utils.html#conjugate_gradient

        r = b - A(self.x)
        p = r
        rsold = torch.dot(r.flatten(), r.flatten())

        for i in (progress_bar := tqdm(range(int(max_iterations)))):
            self.prev_x = self.x.clone() if self.tol is not None else None

            Ap = A(p)
            alpha = rsold / (torch.dot(p.flatten(), Ap.flatten()) + eps)
            self.x = self.x + p * alpha
            r = r - Ap * alpha
            rsnew = torch.dot(r.flatten(), r.flatten())
            assert rsnew.isfinite(), "Conjugate gradient diverged"
            # if rsnew < tol**2:
            #     break

            # We break based on sensitivity or max iterations
            sensitivity = self.sensitivity() if self.tol is not None else None
            if self.tol is not None and sensitivity < self.tol:
                break

            p = r + p * (rsnew / (rsold + eps))
            rsold = rsnew

            spatial_loss, lambda_time_loss = self.loss()
            loss = spatial_loss + lambda_time_loss
            progress_bar.set_description(
                f"Spatial: {spatial_loss.item():.2E} | "
                f"Lambda Time: {lambda_time_loss.item():.2E} | "
                f"Total: {loss.item():.2E} | "
                f"Sensitivity: {f'{sensitivity:.2E}' if sensitivity is not None else 'Not considered'} | "
                f"Residual: {rsnew.item():.2E}"
                # f"Global: {global_loss.item():.2E} | "
            )

            history[i] = np.array(
                [
                    spatial_loss.item(),
                    lambda_time_loss.item(),
                    loss.item(),
                    rsnew.item(),
                    # global_loss.item(),
                ]
            )

        _, ax = plt.subplots(1, history.shape[1], figsize=(4 * history.shape[1], 4))
        for i, title in enumerate(["Spatial", "Lambda Time", "Total", "Residual"]):
            ax[i].scatter(np.arange(len(history[:, i])), history[:, i], marker=".")
            ax[i].set_title(title)
            ax[i].set_yscale("log")
        plt.tight_layout()
        plt.show()

        if return_numpy:
            return (
                self.x.cpu().numpy(),
                self.spc.cpu().numpy(),
                self.cmos.cpu().numpy(),
            )
        else:
            return self.x, self.spc, self.cmos
