"""Data loading, agreement metrics, fusion and SOE on synthetic inputs."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from conftest import latent_points, oracle_triplets, photo_ids, write_expert_db

from triderm.config import SOEConfig
from triderm.data import (
    Choice,
    ExpertTriplet,
    load_descriptions,
    load_excluded_photos,
    load_expert_triplets,
)
from triderm.embeddings import PhotoEmbeddings
from triderm.evaluation import TripletDiffs, agreement, evaluate, triplet_diffs
from triderm.fusion import fuse_similarity, fuse_uncertainty, fused_distances, minmax, row_variance
from triderm.llm_triplets import all_triplet_questions, answer_to_constraint
from triderm.soe import anchor_balanced_order, fit_soe, restrict_triplets, subsample_triplets


def test_expert_triplets_sqlite_and_csv(tmp_path):
    triplets = oracle_triplets(20) + [ExpertTriplet(999, "1", "2", "3", Choice.skip)]
    write_expert_db(tmp_path / "t.db", triplets)
    loaded = load_expert_triplets(tmp_path / "t.db")
    assert len(loaded) == 20 and loaded[0] == triplets[0]  # skipped judgment dropped

    csv_path = tmp_path / "t.csv"
    csv_path.write_text("id,left,middle,right,result\n1,2,1,3,left\n2,4,1,3,skip\n")
    assert load_expert_triplets(csv_path) == [ExpertTriplet(1, "1", "2", "3", Choice.left)]


def test_descriptions_and_exclusions(tmp_path):
    desc = tmp_path / "d.txt"
    desc.write_text("1: deep ulcer\n2a, 2b: shallow erosion\n3 (both wounds): crusted\nnoise\n")
    assert load_descriptions(desc) == {
        "1": "deep ulcer",
        "2a": "shallow erosion",
        "2b": "shallow erosion",
        "3": "crusted",
    }
    excl = tmp_path / "x.txt"
    excl.write_text("# excluded\n4\n5a  # note\n")
    assert load_excluded_photos(excl) == {"4", "5a"}
    assert load_excluded_photos(tmp_path / "missing.txt") == set()


def test_embedding_roundtrip(tmp_path):
    emb = PhotoEmbeddings(np.eye(3, dtype=np.float32), ["a", "b", "c"], "euclidean", {"k": 1})
    emb.save(tmp_path / "e.npz")
    back = PhotoEmbeddings.load(tmp_path / "e.npz")
    assert back.photo_ids == emb.photo_ids and back.metric == "euclidean" and back.meta == {"k": 1}
    np.testing.assert_allclose(back.vectors, emb.vectors)


def test_agreement_metrics_by_hand():
    # anchor A: 2 triplets (1 correct), anchor B: 1 triplet (correct).
    td = TripletDiffs(
        anchors=["A", "A", "B"],
        diffs=np.array([-1.0, -1.0, 1.0]),  # predicted left, left, right
        expert_left=np.array([True, False, False]),
    )
    res = agreement(td)
    assert res.micro == pytest.approx(2 / 3)
    assert res.balanced == pytest.approx((0.5 + 1.0) / 2)
    # pred=[L,L,R], true=[L,R,R]: F1_left = 2/3, F1_right = 2/3
    assert res.macro_f1 == pytest.approx(2 / 3)
    p_e = (1 / 3) * (2 / 3) + (2 / 3) * (1 / 3)
    assert res.kappa == pytest.approx((2 / 3 - p_e) / (1 - p_e))


def test_true_geometry_agrees_fully(expert_triplets):
    emb = PhotoEmbeddings(latent_points(), photo_ids(), "euclidean")
    res = evaluate(emb, expert_triplets)
    assert res.balanced == 1.0 and res.kappa == pytest.approx(1.0)
    # Missing photos are skipped, not counted as wrong.
    res_sub = evaluate(
        PhotoEmbeddings(latent_points()[:6], photo_ids()[:6], "euclidean"), expert_triplets
    )
    assert res_sub.n_triplets < res.n_triplets and res_sub.micro == 1.0


def test_ties_are_not_correct():
    td = triplet_diffs(
        [ExpertTriplet(0, "a", "b", "c", Choice.right)], ["a", "b", "c"], np.zeros((3, 3))
    )
    assert agreement(td).micro == 0.0


def test_fusion_formulas():
    rng = np.random.default_rng(0)
    d_v, d_t = (minmax(np.abs(rng.normal(size=(5, 5)))) for _ in range(2))
    assert d_v.min() == 0 and d_v.max() == 1
    np.testing.assert_allclose(fuse_similarity(d_v, d_t), 1 - (1 - d_v) * (1 - d_t))
    np.testing.assert_allclose(fuse_uncertainty(d_v, d_t, alpha=1.0), d_v)
    np.testing.assert_allclose(fuse_uncertainty(d_v, d_t, alpha=0.0), d_t)

    alpha = 0.7
    var_v, var_t = row_variance(d_v), row_variance(d_t)
    assert var_v[0] == pytest.approx(np.var(d_v[0, 1:]))
    w = alpha * var_v / (alpha * var_v + (1 - alpha) * var_t)
    w01 = (w[0] + w[1]) / 2
    fused = fuse_uncertainty(d_v, d_t, alpha)
    assert fused[0, 1] == pytest.approx(w01 * d_v[0, 1] + (1 - w01) * d_t[0, 1])


def test_fused_distances_align_shared_photos(expert_triplets):
    ids = photo_ids()
    vision = PhotoEmbeddings(latent_points(), ids, "cosine")
    text = PhotoEmbeddings(latent_points()[::-1][:10], ids[::-1][:10], "euclidean")
    shared, dist = fused_distances(vision, text, "uncertainty", alpha=0.7)
    assert shared == sorted(set(ids[2:])) and dist.shape == (10, 10)


def _planted_triplets(n_items: int = 12) -> np.ndarray:
    pts = latent_points()[:n_items]
    rows = []
    for a, b, c in combinations(range(n_items), 3):
        for anchor, x, y in ((a, b, c), (b, a, c), (c, a, b)):
            closer_x = np.linalg.norm(pts[anchor] - pts[x]) < np.linalg.norm(pts[anchor] - pts[y])
            rows.append((anchor, x, y) if closer_x else (anchor, y, x))
    return np.array(rows)


def test_soe_recovers_planted_order():
    triplets = _planted_triplets()
    cfg = SOEConfig(dim=2, epochs=60, batch_size=256, learning_rate=0.05, delta=0.0)
    result = fit_soe(triplets, 12, cfg, device="cpu")
    assert result.embedding.shape == (12, 2)
    assert result.train_error < 0.1
    # Deterministic under the configured seed.
    again = fit_soe(triplets, 12, cfg, device="cpu")
    np.testing.assert_allclose(result.embedding, again.embedding)


def test_anchor_balanced_sampling_covers_anchors():
    triplets = np.array([[0, 1, 2]] * 98 + [[1, 0, 2], [2, 0, 1]])
    np.random.seed(0)
    order = anchor_balanced_order(triplets)
    anchors = triplets[order, 0]
    # Each anchor drawn about a third of the time despite the 98/1/1 imbalance.
    assert all(20 < np.sum(anchors == a) < 50 for a in (0, 1, 2))


def test_restrict_and_subsample():
    triplets = np.array([[0, 1, 2], [1, 2, 3], [0, 3, 1]])
    kept, ids = restrict_triplets(triplets, ["a", "b", "c", "d"], {"a", "b", "d"})
    assert ids == ["a", "b", "d"] and kept.tolist() == [[0, 2, 1]]
    big = np.arange(3000).reshape(1000, 3)
    sub = subsample_triplets(big, 10, seed=42)
    assert len(sub) == 100 and np.all(np.diff(sub[:, 0]) > 0)


def test_triplet_questions():
    ids = ["1", "2", "3", "4a", "4b"]
    questions = all_triplet_questions(ids)
    # 10 triples minus 3 containing both 4a and 4b, each with 3 anchors.
    assert len(questions) == 3 * 7
    assert len(set(questions)) == len(questions)
    assert answer_to_constraint(0, 1, 2, "A") == (1, 0, 2)
    assert answer_to_constraint(0, 1, 2, "C") == (1, 2, 0)
