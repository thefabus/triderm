"""Zero-shot wound features and self-supervised training of the wound pooling head."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from triderm.config import VisionTrainConfig
from triderm.embeddings import PhotoEmbeddings
from triderm.runtime import seed_everything
from triderm.vision.backbones import Backbone
from triderm.vision.dataset import EvalTransform, TrainTransform, WoundDataset, WoundSample
from triderm.vision.losses import make_loss
from triderm.vision.pooling import WoundPooler, wound_tokens


def _photo_embeddings(
    per_photo: dict[str, list[torch.Tensor]], meta: dict[str, Any]
) -> PhotoEmbeddings:
    """Average each photo's rows (wound tokens or wound embeddings) into one vector."""
    ids, vecs = [], []
    for photo_id, rows in per_photo.items():
        stacked = torch.cat(rows, dim=0)
        if stacked.shape[0] == 0:
            print(f"Warning: no wound features for photo {photo_id}")
            continue
        ids.append(photo_id)
        vecs.append(stacked.float().mean(dim=0).cpu().numpy())
    return PhotoEmbeddings(np.stack(vecs), ids, metric="cosine", meta=meta)


@torch.no_grad()
def extract_zero_shot(
    backbone: Backbone, samples: list[WoundSample], input_size: int, batch_size: int = 2
) -> PhotoEmbeddings:
    """Zero-shot baseline: mean of all patch tokens inside the photo's wound masks."""
    dataset = WoundDataset(samples, EvalTransform(input_size, backbone.imagenet_normalize))
    per_photo: dict[str, list[torch.Tensor]] = defaultdict(list)
    for images, masks, photo_ids, _ in DataLoader(dataset, batch_size=batch_size, num_workers=2):
        tokens, grid = backbone.tokens(images)
        for photo_id, feats in zip(photo_ids, wound_tokens(tokens, grid, masks), strict=True):
            per_photo[photo_id].append(feats)
    return _photo_embeddings(per_photo, {"input_size": input_size})


def train_pooler(
    backbone: Backbone, samples: list[WoundSample], cfg: VisionTrainConfig
) -> WoundPooler:
    """Train the pooling head on two augmented views of each wound (backbone frozen)."""
    seed_everything(cfg.seed)
    pooler = WoundPooler(backbone.feature_dim, cfg.output_dim, cfg.pooling).to(backbone.device)
    loss_fn = make_loss(cfg.loss)
    loader = DataLoader(
        WoundDataset(
            samples, TrainTransform(cfg.input_size, backbone.imagenet_normalize), two_views=True
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        pooler.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * cfg.epochs, eta_min=1e-6
    )

    epoch_losses: list[float] = []
    for epoch in range(cfg.epochs):
        backbone.model.train()
        running = 0.0
        for images, masks, photo_ids, _, images2, masks2 in loader:
            optimizer.zero_grad()
            z1 = pooler(backbone.tokens(images)[0], masks)
            z2 = pooler(backbone.tokens(images2)[0], masks2)
            loss = loss_fn(z1, z2, photo_ids)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
        epoch_losses.append(running / len(loader))
        print(f"epoch {epoch}: loss {epoch_losses[-1]:.4f}")
        # Stop once the epoch loss has increased four times in a row.
        recent = epoch_losses[-5:]
        if len(recent) == 5 and all(a < b for a, b in pairwise(recent)):
            print("Early stopping: loss increasing")
            break
    return pooler


@torch.no_grad()
def extract_trained(
    backbone: Backbone, pooler: WoundPooler, samples: list[WoundSample], cfg: VisionTrainConfig
) -> PhotoEmbeddings:
    """One embedding per wound from the trained head, averaged per photo."""
    backbone.model.eval()
    dataset = WoundDataset(samples, EvalTransform(cfg.input_size, backbone.imagenet_normalize))
    per_photo: dict[str, list[torch.Tensor]] = defaultdict(list)
    for images, masks, photo_ids, _ in DataLoader(dataset, batch_size=cfg.batch_size):
        z = pooler(backbone.tokens(images)[0], masks)
        for photo_id, vec in zip(photo_ids, z, strict=True):
            per_photo[photo_id].append(vec.unsqueeze(0))
    return _photo_embeddings(per_photo, {"backbone": cfg.backbone, "loss": cfg.loss})
