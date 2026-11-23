"""End-to-end CLI run on a synthetic cohort (no models, no network)."""

from __future__ import annotations

import asyncio
import json
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
from conftest import latent_points, oracle_triplets, photo_ids, write_expert_db
from typer.testing import CliRunner

from triderm import llm_triplets
from triderm.embeddings import PhotoEmbeddings
from triderm.llm_triplets import LLMEndpoint, ResponseCache, ask_all

REPO = Path(__file__).resolve().parents[1]


def _llm_answers() -> dict:
    """Noise-free 'LLM' answers derived from the latent geometry."""
    pts, ids = latent_points(), photo_ids()
    rows = []
    for a, b, c in combinations(range(len(ids)), 3):
        for anchor, x, y in ((b, a, c), (a, b, c), (c, a, b)):
            closer = np.linalg.norm(pts[anchor] - pts[x]) < np.linalg.norm(pts[anchor] - pts[y])
            rows.append([anchor, x, y] if closer else [anchor, y, x])
    return {"metadata": {}, "photo_ids": ids, "triplets": rows}


def test_pipeline(tmp_path, monkeypatch):
    data, out = tmp_path / "data", tmp_path / "out"
    (data / "annotations").mkdir(parents=True)
    write_expert_db(data / "annotations" / "expert_triplets.db", oracle_triplets())
    monkeypatch.setenv("TRIDERM_DATA_ROOT", str(data))
    monkeypatch.setenv("TRIDERM_OUT", str(out))

    config = tmp_path / "paper.toml"
    shutil.copy(REPO / "configs" / "paper.toml", config)
    text = config.read_text().replace(
        "epochs = 50\nbatch_size = 2048", "epochs = 30\nbatch_size = 256"
    )
    config.write_text(text.replace("dims = [2, 3, 4, 5, 6]", "dims = [2]"))

    (out / "triplets").mkdir(parents=True)
    for oracle in ("gpt-oss-120b", "gpt-oss-20b", "medgemma-27b"):
        (out / "triplets" / f"{oracle}.json").write_text(json.dumps(_llm_answers()))

    # Stand-in vision embedding: noisy latent geometry in 8-d.
    rng = np.random.default_rng(3)
    vision = np.hstack([latent_points(), np.zeros((12, 6))]) + 2.0
    vision += rng.normal(scale=0.3, size=vision.shape)
    PhotoEmbeddings(vision, photo_ids(), "cosine").save(out / "embeddings" / "vision-triderm.npz")

    from triderm.cli import app

    runner = CliRunner()

    def run(*args: str) -> str:
        result = runner.invoke(app, ["--config", str(config), *args])
        assert result.exit_code == 0, result.output + repr(result.exception)
        return result.output

    run("fit-soe", "--oracle", "gpt-oss-120b")
    soe = PhotoEmbeddings.load(out / "embeddings" / "soe-gpt-oss-120b-4d.npz")
    assert soe.metric == "euclidean" and soe.vectors.shape == (12, 4)

    out_eval = run("evaluate", "soe-gpt-oss-120b-4d")
    assert "bal=" in out_eval

    run("table1")
    rows = json.loads((out / "results" / "table1.json").read_text())
    by_label = {r["label"]: r for r in rows}
    assert by_label["TriDerm (Text only)"]["balanced"] > 0.8
    assert by_label["TriDerm (uncertainty)"]["balanced"] > 0.8
    assert by_label["RoBERTa"]["missing"]
    assert "TriDerm (similarity)" in (out / "results" / "table1.tex").read_text()

    run("soe-ablation")
    ablation = json.loads((out / "results" / "soe_ablation.json").read_text())
    assert {r["ablation"] for r in ablation} == {"dim", "budget_percent", "oracle"}

    run("neighbors", "--k", "1")
    lines = (out / "results" / "neighbors-uncertainty.csv").read_text().splitlines()
    assert len(lines) == 1 + 12 * 2


def test_llm_cache_and_failures(tmp_path, monkeypatch):
    calls: list[str] = []

    async def fake_ask(client, model, prompt, seed):
        calls.append(prompt)
        if "fail" in prompt:
            raise RuntimeError("boom")
        return "A"

    monkeypatch.setattr(llm_triplets, "_ask", fake_ask)
    monkeypatch.setattr(LLMEndpoint, "client", lambda self: None)
    endpoint = LLMEndpoint("http://unused", "m", seed=42, max_concurrent=2)
    cache = ResponseCache(tmp_path / "c.sqlite3")

    assert asyncio.run(ask_all(["p1", "fail"], endpoint, cache)) == ["A", None]
    assert asyncio.run(ask_all(["p1", "fail"], endpoint, cache)) == ["A", None]
    # p1 answered from cache the second time; the failure is retried.
    assert calls.count("p1") == 1 and calls.count("fail") == 2
