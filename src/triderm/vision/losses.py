"""Self-supervised objectives for the wound pooling head (VICReg; SupCon/triplet ablations)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from triderm.config import Loss

PairLoss = Callable[[torch.Tensor, torch.Tensor, Sequence[str]], torch.Tensor]


def vicreg(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
) -> torch.Tensor:
    """VICReg (Bardes et al., 2022): invariance + variance + covariance terms."""
    sim = F.mse_loss(z1, z2)
    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1 - std1)) + torch.mean(F.relu(1 - std2))
    d = z1.shape[1]

    def off_diag_sq(z: torch.Tensor) -> torch.Tensor:
        zc = z - z.mean(dim=0)
        cov = (zc.T @ zc) / (z.shape[0] - 1)
        return cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].pow(2).sum()

    cov = (off_diag_sq(z1) + off_diag_sq(z2)) / d
    return sim_weight * sim + var_weight * var + cov_weight * cov


def _photo_labels(photo_ids: Sequence[str]) -> torch.Tensor:
    """Both views of all wounds from the same photo share a label."""
    index = {p: i for i, p in enumerate(dict.fromkeys(photo_ids))}
    labels = [index[p] for p in photo_ids]
    return torch.tensor(labels + labels)


def make_loss(name: Loss) -> PairLoss:
    if name == "vicreg":
        return lambda z1, z2, _ids: vicreg(z1, z2)

    from pytorch_metric_learning.losses import SupConLoss, TripletMarginLoss

    if name == "supcon":
        metric_loss = SupConLoss(temperature=0.07)
    elif name == "triplet":
        metric_loss = TripletMarginLoss(margin=0.1)
    else:
        raise ValueError(f"Unknown loss: {name}")

    def loss(z1: torch.Tensor, z2: torch.Tensor, photo_ids: Sequence[str]) -> torch.Tensor:
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
        return metric_loss(z, _photo_labels(photo_ids))

    return loss
