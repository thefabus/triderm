"""Expert triplet judgments, expert wound descriptions and the photo exclusion list."""

from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Choice(StrEnum):
    """Expert answer: which reference image the anchor ("middle") is more similar to."""

    left = "left"
    right = "right"
    skip = "skip"


@dataclass(frozen=True)
class ExpertTriplet:
    """One expert judgment: is `anchor` more similar to `left` or to `right`?"""

    id: int
    anchor: str
    left: str
    right: str
    choice: Choice


def load_expert_triplets(path: Path) -> list[ExpertTriplet]:
    """Answered (not skipped) expert triplets from the annotation SQLite DB or a CSV export.

    SQLite: table `tripletresult(id, left_fn, middle_fn, right_fn, result)`; photo IDs are the
    file stems of the `*_fn` columns. CSV: columns `id,left,middle,right,result`.
    """
    if not path.exists():  # sqlite3 would only report "unable to open database file"
        raise FileNotFoundError(f"Expert triplets not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            rows = [
                (int(r["id"]), r["middle"], r["left"], r["right"], r["result"])
                for r in csv.DictReader(f)
            ]
    else:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            rows = [
                (int(i), Path(m).stem, Path(lf).stem, Path(rt).stem, res)
                for i, lf, m, rt, res in con.execute(
                    "SELECT id, left_fn, middle_fn, right_fn, result FROM tripletresult"
                )
            ]
    triplets = (
        ExpertTriplet(id=i, anchor=str(m), left=str(lf), right=str(rt), choice=Choice(res))
        for i, m, lf, rt, res in rows
    )
    return [t for t in triplets if t.choice != Choice.skip]


def expert_photo_ids(triplets: list[ExpertTriplet]) -> set[str]:
    """All photo IDs that occur in the expert triplets."""
    return {p for t in triplets for p in (t.anchor, t.left, t.right)}


def load_descriptions(path: Path) -> dict[str, str]:
    """Load `ID: description` lines.

    Keys like `26a, 26b, 26c` (one text for several photos) are expanded to one entry per
    photo; parenthetical notes in the key, e.g. `19 (both wounds)`, are dropped.
    """
    descriptions: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key_part, text = (s.strip() for s in line.split(":", 1))
        if not text:
            continue
        key_clean = re.sub(r"\s*\([^)]*\)", "", key_part)
        for photo_id in (k.strip() for k in key_clean.split(",")):
            if photo_id:
                descriptions[photo_id] = text
    return descriptions


def load_descriptions_sorted(path: Path) -> tuple[list[str], list[str]]:
    """Descriptions and their photo IDs as parallel lists, sorted by photo ID."""
    desc = load_descriptions(path)
    photo_ids = sorted(desc)
    return [desc[p] for p in photo_ids], photo_ids


def load_excluded_photos(path: Path) -> set[str]:
    """Photo IDs to leave out (one per line, `#` comments). Missing file means none."""
    if not path.exists():
        return set()
    ids = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return {i for i in ids if i}


def photo_base(photo_id: str) -> str:
    """Base photo number shared by sub-images, e.g. '26a' -> '26'."""
    match = re.match(r"^(\d+)", photo_id)
    return match.group(1) if match else photo_id
