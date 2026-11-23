"""Soft Ordinal Embedding (SOE; Terada & von Luxburg, 2014) of LLM triplet answers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from triderm.config import SOEConfig
from triderm.runtime import resolve_device, seed_everything

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SOEResult:
    embedding: np.ndarray
    train_error: float
    epochs_run: int


def soe_loss(x_i: torch.Tensor, x_j: torch.Tensor, x_k: torch.Tensor, delta: float) -> torch.Tensor:
    """Sum of hinge losses max(0, delta + ||x_i - x_j|| - ||x_i - x_k||)."""
    loss = delta + torch.norm(x_i - x_j, p=2, dim=1) - torch.norm(x_i - x_k, p=2, dim=1)
    return torch.sum(loss[loss > 0])


def triplet_error(emb: torch.Tensor, triplets: torch.Tensor) -> float:
    """Fraction of (anchor, closer, farther) constraints violated by the embedding."""
    d1 = torch.sum((emb[triplets[:, 0]] - emb[triplets[:, 1]]) ** 2, dim=1)
    d2 = torch.sum((emb[triplets[:, 0]] - emb[triplets[:, 2]]) ** 2, dim=1)
    return float((d2 < d1).float().mean().item())


def anchor_balanced_order(triplets: np.ndarray) -> np.ndarray:
    """Triplet indices for one epoch: draw an anchor uniformly, then one of its triplets.

    Every anchor gets equal weight regardless of how many triplets it has.
    """
    anchors = triplets[:, 0]
    unique = np.unique(anchors)
    by_anchor = {int(a): np.flatnonzero(anchors == a) for a in unique}
    order = np.empty(len(triplets), dtype=np.int64)
    for i, a in enumerate(np.random.choice(unique, size=len(triplets), replace=True)):
        choices = by_anchor[int(a)]
        order[i] = choices[np.random.randint(0, len(choices))]
    return order


def fit_soe(
    triplets: np.ndarray, n_items: int, cfg: SOEConfig, device: str | None = None
) -> SOEResult:
    """Fit an SOE embedding to (anchor, closer, farther) index triplets.

    Adam with AMSGrad from a uniform [0, 1) init. The training error is checked every
    `eval_every` epochs; the embedding with the lowest error is returned, and training stops
    after `patience // eval_every` checks without improvement.
    """
    seed_everything(cfg.seed)
    dev = resolve_device(device)
    x = torch.rand(size=(n_items, cfg.dim), dtype=torch.float32, device=dev, requires_grad=True)
    optimizer = torch.optim.Adam(params=[x], lr=cfg.learning_rate, amsgrad=True)
    trips = torch.tensor(triplets, dtype=torch.long, device=dev)
    n = len(triplets)
    batch_size = min(cfg.batch_size, n)

    with torch.no_grad():
        best_error = triplet_error(x, trips)
    best_x = x.detach().clone()
    stale = 0
    epoch = 0
    for epoch in range(cfg.epochs):
        order = anchor_balanced_order(triplets)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            xs = x[trips[torch.as_tensor(order[start : start + batch_size], device=dev)]]
            loss = soe_loss(xs[:, 0], xs[:, 1], xs[:, 2], delta=cfg.delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            with torch.no_grad():
                err = triplet_error(x, trips)
            log.info("epoch %d loss=%.5f train_error=%.4f", epoch, epoch_loss / n, err)
            if err < best_error - 0.001:
                best_error, best_x, stale = err, x.detach().clone(), 0
            else:
                stale += 1
            if stale >= cfg.patience // cfg.eval_every:
                break

    return SOEResult(best_x.cpu().numpy(), best_error, epoch + 1)


def restrict_triplets(
    triplets: np.ndarray, photo_ids: list[str], keep: set[str]
) -> tuple[np.ndarray, list[str]]:
    """Keep only items in `keep` and the triplets among them, re-indexed."""
    kept = [i for i, p in enumerate(photo_ids) if p in keep]
    remap = {old: new for new, old in enumerate(kept)}
    rows = [[remap[int(v)] for v in t] for t in triplets if all(int(v) in remap for v in t)]
    return np.array(rows, dtype=np.int64).reshape(-1, 3), [photo_ids[i] for i in kept]


def subsample_triplets(triplets: np.ndarray, percent: float, seed: int) -> np.ndarray:
    """Random `percent`% of the triplets (triplet-budget ablation), order preserved."""
    n_keep = max(1, int(round(len(triplets) * percent / 100)))
    if n_keep >= len(triplets):
        return triplets.copy()
    chosen = np.random.default_rng(seed).choice(len(triplets), size=n_keep, replace=False)
    return triplets[np.sort(chosen)]
