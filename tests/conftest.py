"""Synthetic fixtures: a tiny fake cohort with planted similarity structure."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from triderm.data import Choice, ExpertTriplet

N_PHOTOS = 12


def photo_ids() -> list[str]:
    return [f"{i + 1}" for i in range(N_PHOTOS)]


def latent_points(seed: int = 0) -> np.ndarray:
    """Ground-truth 2D positions that define who is similar to whom."""
    return np.random.default_rng(seed).normal(size=(N_PHOTOS, 2))


def oracle_triplets(n: int = 300, seed: int = 1) -> list[ExpertTriplet]:
    """Expert-like triplets answered from the latent points."""
    rng = np.random.default_rng(seed)
    pts, ids = latent_points(), photo_ids()
    out = []
    for t in range(n):
        a, lf, rt = rng.choice(N_PHOTOS, size=3, replace=False)
        closer_left = np.linalg.norm(pts[a] - pts[lf]) < np.linalg.norm(pts[a] - pts[rt])
        choice = Choice.left if closer_left else Choice.right
        out.append(ExpertTriplet(t, ids[a], ids[lf], ids[rt], choice))
    return out


def write_expert_db(path: Path, triplets: list[ExpertTriplet]) -> None:
    with sqlite3.connect(path) as con:
        con.execute(
            "CREATE TABLE tripletresult (id INTEGER PRIMARY KEY, left_fn TEXT, middle_fn TEXT, "
            "right_fn TEXT, result TEXT)"
        )
        con.executemany(
            "INSERT INTO tripletresult VALUES (?, ?, ?, ?, ?)",
            [
                (t.id, f"{t.left}.jpg", f"{t.anchor}.jpg", f"{t.right}.jpg", t.choice.value)
                for t in triplets
            ],
        )


@pytest.fixture
def expert_triplets() -> list[ExpertTriplet]:
    return oracle_triplets()
