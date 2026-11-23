"""Table 1 (evaluate the configured rows, render Markdown and LaTeX) and Figure 2 neighbours."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from triderm.config import Config, Table1Row
from triderm.data import ExpertTriplet
from triderm.embeddings import PhotoEmbeddings
from triderm.evaluation import Agreement, evaluate, evaluate_distances
from triderm.fusion import fused_distances


@dataclass(frozen=True)
class RowResult:
    row: Table1Row
    result: Agreement | None
    missing: str | None = None

    def as_dict(self) -> dict[str, Any]:
        scores = self.result.as_dict() if self.result else {}
        return {"block": self.row.block, "label": self.row.label, "missing": self.missing} | scores


def evaluate_table1_row(cfg: Config, row: Table1Row, triplets: list[ExpertTriplet]) -> RowResult:
    """Score one row; rows whose embeddings have not been computed yet are marked missing."""
    paths = [cfg.paths.embedding(name) for name in row.embeddings]
    if missing := [str(p) for p in paths if not p.exists()]:
        return RowResult(row, None, ", ".join(missing))
    embeddings = [PhotoEmbeddings.load(p) for p in paths]
    if row.fusion is None:
        return RowResult(row, evaluate(embeddings[0], triplets))
    photo_ids, dist = fused_distances(*embeddings, row.fusion, cfg.fusion_alpha)
    return RowResult(row, evaluate_distances(triplets, photo_ids, dist))


def _cells(r: Agreement | None) -> list[str]:
    if r is None:
        return ["--"] * 4
    percents = (f"{100 * v:.1f}" for v in (r.balanced, r.micro, r.macro_f1))
    return [*percents, f"{r.kappa:.3f}"]


def markdown_table(rows: list[RowResult]) -> str:
    lines = [
        "| Modality | Method | Bal. Agr. | Micro | Macro-F1 | kappa | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        n = str(r.result.n_triplets) if r.result else "missing"
        cells = " | ".join([r.row.block, r.row.label, *_cells(r.result), n])
        lines.append(f"| {cells} |")
    return "\n".join(lines)


def latex_table(rows: list[RowResult]) -> str:
    """Body of Table 1 with the best value per block in bold."""
    out = []
    for block in dict.fromkeys(r.row.block for r in rows):
        block_rows = [r for r in rows if r.row.block == block]
        values = np.array(
            [
                [r.result.balanced, r.result.micro, r.result.macro_f1, r.result.kappa]
                if r.result
                else [np.nan] * 4
                for r in block_rows
            ]
        )
        best = np.nanmax(values, axis=0) if np.isfinite(values).any() else [np.nan] * 4
        out.append(f"\\multirow{{{len(block_rows)}}}{{*}}{{\\textit{{{block.upper()}}}}}")
        for r, vals in zip(block_rows, values, strict=True):
            cells = [
                f"\\textbf{{{c}}}" if np.isfinite(v) and v == b else c
                for c, v, b in zip(_cells(r.result), vals, best, strict=True)
            ]
            out.append(f" & {r.row.label} & " + " & ".join(cells) + " \\\\")
        out.append("\\midrule")
    return "\n".join(out[:-1])


def nearest_and_farthest(photo_ids: list[str], dist: np.ndarray, k: int) -> list[dict[str, Any]]:
    """For every reference photo, its k most similar and k most dissimilar photos."""
    rows = []
    for i, ref in enumerate(photo_ids):
        order = [j for j in np.argsort(dist[i]) if j != i]
        for kind, picks in (("similar", order[:k]), ("dissimilar", order[::-1][:k])):
            for rank, j in enumerate(picks, start=1):
                rows.append(
                    {
                        "reference": ref,
                        "kind": kind,
                        "rank": rank,
                        "photo": photo_ids[j],
                        "distance": float(dist[i, j]),
                    }
                )
    return rows
