"""Vision data pipeline, pooling head and training loop with a tiny stand-in backbone."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from triderm.config import VisionTrainConfig
from triderm.vision.backbones import Backbone
from triderm.vision.dataset import (
    EvalTransform,
    TrainTransform,
    WoundDataset,
    find_wound_samples,
)
from triderm.vision.losses import make_loss, vicreg
from triderm.vision.pooling import WoundPooler, wound_tokens
from triderm.vision.train import extract_trained, extract_zero_shot, train_pooler

SIZE = 32


class TinyBackbone(Backbone):
    """Patchify with a frozen conv: 8x8 patches -> 4x4 grid of 16-d tokens."""

    feature_dim = 16

    def __init__(self) -> None:
        super().__init__("cpu")
        torch.manual_seed(0)
        self.model = nn.Conv2d(3, 16, kernel_size=8, stride=8)

    def tokens(self, images):
        feats = self.model(images)
        return feats.flatten(2).transpose(1, 2), (feats.shape[2], feats.shape[3])


@pytest.fixture
def wound_dirs(tmp_path):
    """4 synthetic photos (one excluded later), 1-2 square wound masks each."""
    rng = np.random.default_rng(0)
    photos, masks = tmp_path / "photos", tmp_path / "masks"
    photos.mkdir()
    masks.mkdir()
    for i, pid in enumerate(["1", "2", "3a", "4"]):
        Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(
            photos / f"{pid}.jpg"
        )
        for k in range(1 + i % 2):
            m = np.zeros((60, 80), dtype=np.uint8)
            m[10 + 20 * k : 30 + 20 * k, 20:50] = 255
            Image.fromarray(m).save(masks / f"{pid}_wounds_{k}.png")
    return photos, masks


def test_find_samples_and_transforms(wound_dirs):
    samples = find_wound_samples(*wound_dirs, exclude={"4"})
    assert sorted({s.photo_id for s in samples}) == ["1", "2", "3a"]
    assert len(samples) == 4

    img, msk, pid, wid = WoundDataset(samples, EvalTransform(SIZE, True))[0]
    assert img.shape == (3, SIZE, SIZE) and msk.shape == (SIZE, SIZE) and msk.sum() > 0
    ds = WoundDataset(samples, TrainTransform(SIZE, False), two_views=True)
    item = ds[1]
    assert len(item) == 6 and item[0].shape == item[4].shape == (3, SIZE, SIZE)


def test_wound_tokens_and_pooler():
    tokens = torch.randn(2, 16, 8)
    masks = torch.zeros(2, 32, 32, dtype=torch.long)
    masks[0, :8, :8] = 255  # exactly one 8x8 patch
    masks[1, :, :16] = 255  # left half
    sel = wound_tokens(tokens, (4, 4), masks)
    assert [s.shape[0] for s in sel] == [1, 8]
    torch.testing.assert_close(sel[0][0], tokens[0, 0])

    for pooling in ("attention", "mean"):
        pooler = WoundPooler(8, output_dim=256, pooling=pooling)
        pooled = pooler.pool(tokens, masks)
        if pooling == "mean":
            torch.testing.assert_close(pooled[0], tokens[0, 0])
        assert pooler(tokens, masks).shape == (2, 256)


def test_losses_are_finite():
    z1, z2 = torch.randn(6, 256), torch.randn(6, 256)
    assert vicreg(z1, z1) < vicreg(z1, z2)
    ids = ["1", "1", "2", "3", "3", "4"]
    for name in ("vicreg", "supcon", "triplet"):
        assert torch.isfinite(make_loss(name)(z1, z2, ids))


def test_zero_shot_and_training(wound_dirs):
    samples = find_wound_samples(*wound_dirs, exclude=set())
    backbone = TinyBackbone()
    backbone.freeze()

    zs = extract_zero_shot(backbone, samples, SIZE)
    assert sorted(zs.photo_ids) == ["1", "2", "3a", "4"] and zs.vectors.shape == (4, 16)
    assert zs.metric == "cosine"

    cfg = VisionTrainConfig(
        backbone="tiny", input_size=SIZE, output_dim=256, epochs=1, batch_size=4, learning_rate=1e-3
    )
    pooler = train_pooler(backbone, samples, cfg)
    emb = extract_trained(backbone, pooler, samples, cfg)
    assert emb.vectors.shape == (4, 256) and np.isfinite(emb.vectors).all()
    # Backbone stays frozen.
    assert all(not p.requires_grad for p in backbone.model.parameters())
