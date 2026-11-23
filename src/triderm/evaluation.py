"""Agreement of a distance matrix with expert triplet judgments (paper Sec. 3.1)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from triderm.data import Choice, ExpertTriplet
from triderm.embeddings import PhotoEmbeddings


@dataclass(frozen=True)
class TripletDiffs:
    """Per evaluable expert triplet: anchor, d(anchor, left) - d(anchor, right), expert label."""

    anchors: list[str]
    diffs: np.ndarray
    expert_left: np.ndarray


@dataclass(frozen=True)
class Agreement:
    balanced: float
    micro: float
    macro_f1: float
    kappa: float
    n_triplets: int
    n_anchors: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def triplet_diffs(
    triplets: list[ExpertTriplet], photo_ids: list[str], dist: np.ndarray
) -> TripletDiffs:
    """Distance differences for the expert triplets whose three photos all have an embedding."""
    index = {p: i for i, p in enumerate(photo_ids)}
    anchors, diffs, labels = [], [], []
    for t in triplets:
        if t.choice == Choice.skip or not all(p in index for p in (t.anchor, t.left, t.right)):
            continue
        a, lf, rt = index[t.anchor], index[t.left], index[t.right]
        anchors.append(t.anchor)
        diffs.append(float(dist[a, lf]) - float(dist[a, rt]))
        labels.append(t.choice == Choice.left)
    return TripletDiffs(anchors, np.asarray(diffs, dtype=np.float64), np.asarray(labels, bool))


def _f1(pred: np.ndarray, true: np.ndarray, cls: bool) -> float:
    tp = int(np.sum((pred == cls) & (true == cls)))
    fp = int(np.sum((pred == cls) & (true != cls)))
    fn = int(np.sum((pred != cls) & (true == cls)))
    if tp == 0:
        return float("nan")
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def agreement(td: TripletDiffs) -> Agreement:
    """Anchor-balanced agreement, micro agreement, macro-F1 and Cohen's kappa.

    A triplet is correct when the margin Delta_t = d(i,k) - d(i,j) toward the expert's choice
    j is strictly positive. For macro-F1 and kappa, "left" is predicted when d_left < d_right.
    """
    n = len(td.diffs)
    if n == 0:
        nan = float("nan")
        return Agreement(nan, nan, nan, nan, 0, 0)

    correct = np.where(td.expert_left, -td.diffs, td.diffs) > 0
    by_anchor: dict[str, list[bool]] = defaultdict(list)
    for anchor, ok in zip(td.anchors, correct, strict=True):
        by_anchor[anchor].append(bool(ok))

    pred_left, true_left = td.diffs < 0, td.expert_left
    macro_f1 = np.nanmean([_f1(pred_left, true_left, True), _f1(pred_left, true_left, False)])

    p_o = float(np.mean(pred_left == true_left))
    p_true, p_pred = float(np.mean(true_left)), float(np.mean(pred_left))
    p_e = p_true * p_pred + (1 - p_true) * (1 - p_pred)
    kappa = float("nan") if abs(1 - p_e) < 1e-12 else (p_o - p_e) / (1 - p_e)

    return Agreement(
        balanced=float(np.mean([np.mean(v) for v in by_anchor.values()])),
        micro=float(np.mean(correct)),
        macro_f1=float(macro_f1),
        kappa=kappa,
        n_triplets=n,
        n_anchors=len(by_anchor),
    )


def evaluate_distances(
    triplets: list[ExpertTriplet], photo_ids: list[str], dist: np.ndarray
) -> Agreement:
    return agreement(triplet_diffs(triplets, photo_ids, dist))


def evaluate(emb: PhotoEmbeddings, triplets: list[ExpertTriplet]) -> Agreement:
    return evaluate_distances(triplets, emb.photo_ids, emb.distances())
