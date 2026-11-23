"""Wound-token selection and the trainable pooling + predictor head."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from triderm.config import Pooling


def wound_tokens(
    tokens: torch.Tensor, grid: tuple[int, int], masks: torch.Tensor
) -> list[torch.Tensor]:
    """For each wound in the batch, the [n_i, C] tokens whose grid cell lies inside its mask."""
    keep = F.interpolate(masks.to(tokens.device).unsqueeze(0).float(), size=grid, mode="nearest")
    keep = (keep[0] > 0).flatten(1)
    return [tokens[i][keep[i]] for i in range(len(tokens))]


class WoundPooler(nn.Module):
    """Pools the tokens inside a wound mask, then maps them to the wound representation.

    `attention`: a two-layer tanh MLP scores each token, softmax over the in-mask tokens.
    `mean`: masked average. The predictor is Linear-LayerNorm-GELU-Linear.
    """

    def __init__(self, input_dim: int, output_dim: int = 512, pooling: Pooling = "attention"):
        super().__init__()
        hidden = 2 * output_dim
        self.pooling = pooling
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
            )

    def pool(self, tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Tokens [B, N, C] on a square grid and masks [B, H, W] -> pooled [B, C]."""
        side = int(tokens.shape[1] ** 0.5)
        # Raw mask values serve as weights (binary masks give a plain masked mean).
        keep = F.interpolate(masks.unsqueeze(1).float(), size=(side, side), mode="nearest")
        keep = keep.flatten(1).unsqueeze(-1)
        if self.pooling == "mean":
            return (tokens * keep).sum(dim=1) / (keep.sum(dim=1) + 1e-6)
        scores = self.attention(tokens).masked_fill(keep == 0, -1e9)
        return (tokens * F.softmax(scores, dim=1)).sum(dim=1)

    def forward(self, tokens: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(tokens, masks.to(tokens.device)))
