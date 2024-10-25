import torch
import numpy as np

from utils import mono_exponential_decay_torch


class DecayLoss(torch.nn.Module):
    def __init__(self, t):
        super(DecayLoss, self).__init__()
        if isinstance(t, np.ndarray):
            self.t = torch.from_numpy(t).float()
        else:
            self.t = t

    def forward(self, pred_coeffs: torch.Tensor, target: torch.Tensor):
        pred_coeffs = pred_coeffs.T
        pred = mono_exponential_decay_torch(pred_coeffs[0], pred_coeffs[1], pred_coeffs[2], self.t)
        return torch.nn.functional.mse_loss(pred.T, target)


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
