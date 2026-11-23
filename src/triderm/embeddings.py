"""One embedding vector per photo, its distance metric, and the .npz file format."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.spatial.distance import cdist

Metric = Literal["cosine", "euclidean"]


@dataclass
class PhotoEmbeddings:
    """Photo embeddings; SOE lives in Euclidean space, foundation-model features use cosine."""

    vectors: np.ndarray
    photo_ids: list[str]
    metric: Metric
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or len(self.vectors) != len(self.photo_ids):
            raise ValueError(f"{self.vectors.shape} vectors for {len(self.photo_ids)} photo IDs")

    def distances(self) -> np.ndarray:
        """Pairwise distance matrix under `metric` (cosine on explicitly L2-normalized vectors)."""
        if self.metric == "euclidean":
            return cdist(self.vectors, self.vectors, metric="euclidean").astype(np.float32)
        emb = np.asarray(self.vectors, dtype=np.float32)
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
        return (1.0 - np.clip(emb @ emb.T, -1.0, 1.0)).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=self.vectors.astype(np.float32),
            photo_ids=np.array(self.photo_ids, dtype=str),
            metric=np.array(self.metric),
            meta=np.array(json.dumps(self.meta)),
        )
        print(f"Saved {len(self.photo_ids)} x {self.vectors.shape[1]} embeddings to {path}")

    @classmethod
    def load(cls, path: Path) -> PhotoEmbeddings:
        with np.load(path, allow_pickle=False) as f:
            return cls(
                vectors=f["vectors"].astype(np.float32),
                photo_ids=[str(p) for p in f["photo_ids"]],
                metric=str(f["metric"]),  # type: ignore[arg-type]
                meta=json.loads(str(f["meta"])),
            )
