"""Training-free late fusion of vision and text distance matrices (paper Sec. 2.2)."""

from __future__ import annotations

import numpy as np

from triderm.config import Fusion
from triderm.embeddings import PhotoEmbeddings


def minmax(dist: np.ndarray) -> np.ndarray:
    """Scale a distance matrix to [0, 1] using its global min and max."""
    lo, hi = float(dist.min()), float(dist.max())
    return (dist - lo) / (hi - lo) if hi > lo else dist


def row_variance(dist: np.ndarray) -> np.ndarray:
    """Variance of each sample's distances to all other samples (diagonal excluded)."""
    off_diag = ~np.eye(len(dist), dtype=bool)
    return np.array([dist[i, off_diag[i]].var() for i in range(len(dist))])


def fuse_uncertainty(d_vision: np.ndarray, d_text: np.ndarray, alpha: float) -> np.ndarray:
    """Uncertainty-aware fusion: weight each modality by its per-sample distance variance.

    w_i = a*var_V,i / (a*var_V,i + (1-a)*var_T,i);  D(i,j) = w_ij*D_V + (1-w_ij)*D_T,
    with w_ij the mean of w_i and w_j. Inputs must already be min-max normalized.
    """
    var_v, var_t = row_variance(d_vision), row_variance(d_text)
    denom = alpha * var_v + (1 - alpha) * var_t
    w = np.divide(alpha * var_v, denom, out=np.full_like(denom, 0.5), where=denom > 0)
    w_pair = 0.5 * (w[:, None] + w[None, :])
    return w_pair * d_vision + (1 - w_pair) * d_text


def fuse_similarity(d_vision: np.ndarray, d_text: np.ndarray) -> np.ndarray:
    """Similarity-based fusion D = 1 - (1 - D_V)(1 - D_T): close only if both modalities agree."""
    return 1.0 - (1.0 - d_vision) * (1.0 - d_text)


def _shared_distances(emb: PhotoEmbeddings, shared: list[str]) -> np.ndarray:
    index = {p: i for i, p in enumerate(emb.photo_ids)}
    rows = [index[p] for p in shared]
    return minmax(emb.distances()[np.ix_(rows, rows)])


def fused_distances(
    vision: PhotoEmbeddings, text: PhotoEmbeddings, fusion: Fusion, alpha: float
) -> tuple[list[str], np.ndarray]:
    """Fused distance matrix over the photos present in both modalities."""
    shared = sorted(set(vision.photo_ids) & set(text.photo_ids))
    if len(shared) < 3:
        raise ValueError(f"Only {len(shared)} photos shared between vision and text embeddings")
    d_v, d_t = _shared_distances(vision, shared), _shared_distances(text, shared)
    if fusion == "uncertainty":
        return shared, fuse_uncertainty(d_v, d_t, alpha)
    return shared, fuse_similarity(d_v, d_t)
