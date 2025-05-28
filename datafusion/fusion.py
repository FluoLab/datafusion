import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torch.nn.functional import conv2d, conv_transpose2d
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from datafusion.baseline import baseline


def squared_l2(x: torch.Tensor | np.ndarray) -> float:
    """
    Computes the squared L2 norm of the input tensor.
    :param x:
    :return: squared L2 norm of the input tensor.
    """
    return (x**2).sum()


class SumOperator:
    """
    A simple sum operator that sums over a specified dimension and normalizes the result.
    This operator can be used to sum over time, spectrum, or any other dimension in a tensor.
    """

    def __init__(self, size: int = 1, integral_dim: int = 0):
        """
        Initializes the SumOperator.
        :param size: The size of the dimension to sum over.
        :param integral_dim: The dimension to sum over.
        """
        self.size = size
        self.integral_dim = integral_dim

    def T(self, y: torch.Tensor) -> torch.Tensor:
        """
        Transpose operation for the sum operator.
        :param y: The input tensor where the transpose operation is applied.
        :return: A tensor with the transposed sum operation applied.
        """
        return y.repeat_interleave(self.size, dim=self.integral_dim) / self.size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the sum operator to the input tensor.
        :param x: The input tensor to apply the sum operator to.
        :return: The summed tensor.
        """
        self.size = x.shape[self.integral_dim]
        return x.sum(dim=self.integral_dim, keepdim=True)


class SumPoolOperator:
    """
    A pooling operator that performs a sum pooling operation in 2D.
    This operator is used to downsample the spatial dimensions of a tensor by summing over non-overlapping regions.
    The batch and channel dimensions are preserved, and the spatial dimensions are reduced by a factor of `size`.
    This implementation uses a convolution transpose operation to achieve the pooling effect.
    """

    def __init__(
        self,
        size: int = 4,
        channels: int = 16,
        device: str = "cpu",
    ):
        """
        Initializes the SumPoolOperator.
        :param size: The size of the pooling operation, which determines the downsampling factor.
        :param channels: Number of channels in the input tensor.
        :param device: Device to perform the operations on (e.g., "cpu" or "cuda").
        """
        self.size = size
        self.kernel = torch.ones(channels, 1, size, size, device=device)
        self.channels = channels

    def T(self, y: torch.Tensor) -> torch.Tensor:
        """
        Transpose operation for the sum pooling operator.
        :param y: The input tensor where the transpose operation is applied.
        :return: The transposed tensor after applying the sum pooling operation, normalized.
        """
        y = y.squeeze(2)
        x = conv_transpose2d(y, self.kernel, stride=self.size, groups=self.channels, bias=None)
        x = x.unsqueeze(2)
        return x / self.size**2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the sum pooling operator to the input tensor.
        :param x: The input tensor to apply the sum pooling operator to.
        :return: The pooled tensor after applying the sum pooling operation.
        """
        x = x.squeeze(2)
        y = conv2d(x, self.kernel, stride=self.size, groups=self.channels, bias=None)
        y = y.unsqueeze(2)
        return y


class Fusion:
    """
    Base class for data fusion methods.
    This class initializes the necessary parameters and operators for data fusion.
    It provides methods for normalization, loss calculation, and sensitivity calculation.
    """

    def __init__(
        self,
        spc: np.ndarray | torch.Tensor,
        cmos: np.ndarray | torch.Tensor,
        *,
        weights: dict,
        init_type: str,
        tol: float | None = 1e-6,
        mask_noise: bool = False,
        total_energy: float = 1.0,
        device: str = "cpu",
        seed: int = 42,
    ):
        """
        Initializes the Fusion class with the provided parameters.
        :param spc: Spectral data tensor of shape (time, lambda, x, y).
        :param cmos: CMOS data tensor of shape (z, x, y).
        :param weights: Dictionary containing weights for different loss components.
        :param init_type: Type of initialization for the parameters ("random", "zeros", "baseline").
        :param tol: Tolerance for convergence. If None, sensitivity is not considered.
        :param mask_noise: If True, applies a mask to the noise in the input data.
        :param total_energy: Total energy to normalize the input data.
        :param device: Device to perform the computations on (e.g., "cpu" or "cuda").
        :param seed: Random seed for reproducibility.
        """

        match spc:
            case np.ndarray():
                self.spc = torch.from_numpy(spc.astype(np.float32)).to(device)
            case torch.Tensor():
                self.spc = spc.to(device)
            case _:
                raise TypeError("spc must be a numpy.ndarray or torch.Tensor")
        match cmos:
            case np.ndarray():
                self.cmos = torch.from_numpy(cmos.astype(np.float32)).to(device)
            case torch.Tensor():
                self.cmos = cmos.to(device)
            case _:
                raise TypeError("cmos must be a numpy.ndarray or torch.Tensor")

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

        self.T = SumOperator(size=self.n_times, integral_dim=0)
        self.S = SumOperator(size=self.n_lambdas, integral_dim=1)
        self.D = SumOperator(size=self.z_dim, integral_dim=2)
        self.R = SumPoolOperator(
            size=self.spatial_increase,
            channels=self.n_lambdas,
            device=self.device,
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
    def normalize_energy(
        tensor: torch.Tensor,
        total_energy: int | float = 1,
    ) -> torch.Tensor:
        return total_energy * tensor / tensor.sum()

    def loss(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss for the fusion method.
        :return: loss components for spatial and lambda time.
        """
        if self.mask_noise:
            x1 = self.cmos[self.cmos_mask]
            x2 = self.T(self.S(self.x))[0, 0, self.cmos_mask]
            spatial_loss = self.weights["spatial"] * squared_l2(x1 - x2)

            x1 = self.spc[:, :, self.spc_mask]
            x2 = self.R(self.D(self.x)).squeeze(2)[:, :, self.spc_mask]
            spectro_temporal_loss = self.weights["spectro_temporal"] * squared_l2(x1 - x2)

        else:
            x1 = self.cmos.flatten()
            x2 = self.T(self.S(self.x)).flatten()
            spatial_loss = self.weights["spatial"] * squared_l2(x1 - x2)

            x1 = self.spc.flatten()
            x2 = self.R(self.D(self.x)).squeeze(2).flatten()
            spectro_temporal_loss = self.weights["spectro_temporal"] * squared_l2(x1 - x2)

        # There is a possible global loss that can be added to the total loss.
        # We notice that adding it does not improve the results.
        # Global has no spatial dimension, so no need to mask.

        # x1 = self.spc.sum(dim=(2, 3)).flatten()
        # x2 = self.x.sum(dim=(2, 3, 4)).flatten()
        # global_loss = weights["global"] * squared_l2(x1 - x2)

        return spatial_loss, spectro_temporal_loss

    def sensitivity(self) -> torch.Tensor:
        """
        Computes the sensitivity of the current solution, computed as:
            || x - prev_x ||_2
        :return: sensitivity
        """
        return torch.linalg.vector_norm(self.x.flatten() - self.prev_x.flatten())

    def _initialize(self) -> torch.Tensor:
        """
        Initializes the parameters based on the specified initialization type.
        :return: Initialized tensor x for fusion.
        """
        torch.manual_seed(self.seed)
        match self.init_type:
            case "random":
                x = (self.cmos.min() - self.cmos.max()) * torch.rand(
                    self.x_shape, device=self.device
                ) + self.cmos.max()
            case "zeros":
                x = torch.zeros(self.x_shape).to(self.device)
            case "baseline":
                x = baseline(self.cmos, self.spc, device="cpu", return_numpy=False)
                x = x.to(self.device)
            case _:
                raise ValueError("Invalid initialization type.")
        return x

    def _mask_gradients(self):
        """
        Masks the gradients of the x tensor, this way we do not optimize for masked regions.
        """
        self.x.grad[:, :, ~self.cmos_mask] = 0.0

    def _get_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the masks for the SPC and CMOS data.
        :return: Tuple of masks for SPC and CMOS data.
        """
        cmos_mask = self.cmos > (0.05 * self.cmos.max())
        spc_mask = resize(
            cmos_mask.any(dim=0).unsqueeze(0).float(),
            size=[self.spc.shape[-2], self.spc.shape[-1]],
            interpolation=InterpolationMode.BILINEAR,
            antialias=False,
        ).bool()
        return spc_mask, cmos_mask


class FusionAdam(Fusion):
    """
    This class implements data fusion using the Adam optimizer and backpropagation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        lr: float,
        max_iterations: int,
        non_neg: bool = False,
        return_numpy: bool = True,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray, np.ndarray]
    ):
        """
        Runs the Adam optimization algorithm to minimize the loss function.
        :param lr: Learning rate for the Adam optimizer.
        :param max_iterations: Maximum number of iterations for the optimization.
        :param non_neg: If True, enforces non-negativity constraints on the solution.
        :param return_numpy: If True, returns the results as numpy arrays; otherwise, returns torch tensors.
        :return: A tuple containing the optimized x tensor, normalized SPC, and normalized CMOS.
        """

        history = np.zeros((max_iterations, len(self.weights) + 1))

        self.x = torch.nn.Parameter(self.x, requires_grad=True)
        optimizer = torch.optim.Adam([self.x], lr=lr, amsgrad=False)

        for i in (progress_bar := tqdm(range(max_iterations))):
            self.prev_x = self.x.detach().clone() if self.tol is not None else None
            optimizer.zero_grad()

            spatial_loss, spectro_temporal_loss = self.loss()
            loss = spatial_loss + spectro_temporal_loss

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
                f"Spectro Temporal: {spectro_temporal_loss.item():.2E} | "
                f"Total: {loss.item():.2E} | "
                f"Sensitivity: {f'{sensitivity:.2E}' if sensitivity is not None else 'Not considered'} | "
            )

            if sensitivity is not None and sensitivity < self.tol:
                break

            history[i] = np.array(
                [
                    spatial_loss.item(),
                    spectro_temporal_loss.item(),
                    loss.item(),
                    # global_loss.item(),
                ]
            )

        _, ax = plt.subplots(1, history.shape[1], figsize=(4 * history.shape[1], 4))
        for i, title in enumerate(["Spatial", "Spectro Temporal", "Total"]):
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
    """
    This class implements data fusion using the conjugate gradient method in pytorch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w1 = self.weights["spatial"]
        self.w2 = self.weights["spectro_temporal"]

    def __call__(
        self,
        max_iterations: int,
        eps: float = 1e-10,
        return_numpy: bool = True,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray, np.ndarray]
    ):
        """
        Runs the conjugate gradient method to solve Ax = b:
            A is defined as a lambda function, applied at each iteration.
            x is the tensor to optimize
            b is computed based on the input tensors once
        A and b are derived from setting the gradient of the loss function to zero.
        :param max_iterations: Maximum number of iterations for the optimization.
        :param eps: Small value to avoid division by zero in the conjugate gradient method.
        :param return_numpy: If True, returns the results as numpy arrays; otherwise, returns torch tensors.
        :return: A tuple containing the optimized x tensor, normalized SPC, and normalized CMOS.
        """

        history = np.zeros((max_iterations, len(self.weights) + 2))

        A = lambda x: (
            self.w1 * self.T.T(self.S.T(self.S(self.T(x))))
            + self.w2 * self.D.T(self.R.T(self.R(self.D(x))))
        )

        b = (self.w1 * self.T.T(self.S.T(self.cmos.unsqueeze(0).unsqueeze(0))) +
             self.w2 * self.D.T(self.R.T(self.spc.unsqueeze(2))
        ))

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

            # We break based on sensitivity or max iterations
            sensitivity = self.sensitivity() if self.tol is not None else None
            if self.tol is not None and sensitivity < self.tol:
                break

            p = r + p * (rsnew / (rsold + eps))
            rsold = rsnew

            spatial_loss, spectro_temporal_loss = self.loss()
            loss = spatial_loss + spectro_temporal_loss
            progress_bar.set_description(
                f"Spatial: {spatial_loss.item():.2E} | "
                f"Spectro Temporal: {spectro_temporal_loss.item():.2E} | "
                f"Total: {loss.item():.2E} | "
                f"Sensitivity: {f'{sensitivity:.2E}' if sensitivity is not None else 'Not considered'} | "
                f"Residual: {rsnew.item():.2E}"
                # f"Global: {global_loss.item():.2E} | "
            )

            history[i] = np.array(
                [
                    spatial_loss.item(),
                    spectro_temporal_loss.item(),
                    loss.item(),
                    rsnew.item(),
                    # global_loss.item(),
                ]
            )

        _, ax = plt.subplots(1, history.shape[1], figsize=(4 * history.shape[1], 4))
        for i, title in enumerate(["Spatial", "Spectro Temporal", "Total", "Residual"]):
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
