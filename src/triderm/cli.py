"""TriDerm: RDEB wound embeddings (vision, LLM-triplet text, fused) scored by expert triplets."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer
from dotenv import load_dotenv

from triderm.config import DEFAULT_CONFIG, Config, Fusion, load_config
from triderm.data import (
    expert_photo_ids,
    load_descriptions_sorted,
    load_excluded_photos,
    load_expert_triplets,
)
from triderm.embeddings import PhotoEmbeddings
from triderm.evaluation import Agreement, evaluate

if TYPE_CHECKING:
    from triderm.vision.dataset import WoundSample

TEXT, VISION, RESULTS = "Text branch", "Vision branch", "Paper results"
Oracle = Annotated[str, typer.Option(help="Oracle, a key of llm.oracles.")]
Device = Annotated[str | None, typer.Option(help="Torch device; default CUDA, MPS or CPU.")]

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[Path, typer.Option(help="TOML config.")] = DEFAULT_CONFIG,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log progress.")] = False,
) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")
    ctx.obj = config


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path}")


def _summary(name: str, res: Agreement) -> str:
    return (
        f"{name}: bal={100 * res.balanced:.1f} micro={100 * res.micro:.1f} "
        f"F1={100 * res.macro_f1:.1f} kappa={res.kappa:.3f} (n={res.n_triplets})"
    )


@app.command("ask-triplets", rich_help_panel=TEXT)
def ask_triplets(
    ctx: typer.Context,
    oracle: Oracle = "gpt-oss-120b",
    n_questions: Annotated[int | None, typer.Option(help="Ask a random subset.")] = None,
) -> None:
    """Ask the LLM oracle every triplet question over the expert descriptions."""
    from triderm.llm_triplets import LLMEndpoint, ResponseCache, run_triplet_queries

    cfg = load_config(ctx.obj)
    texts, photo_ids = load_descriptions_sorted(cfg.paths.descriptions)
    endpoint = LLMEndpoint(
        cfg.llm.base_url, cfg.llm.oracles[oracle], cfg.llm.seed, cfg.llm.max_concurrent
    )
    cache = ResponseCache(cfg.paths.llm_cache)
    payload = run_triplet_queries(texts, photo_ids, endpoint, cache, n_questions)
    _write_json(cfg.paths.llm_triplets(oracle), payload)


def _soe_embedding(
    cfg: Config, oracle: str, dim: int, budget_percent: float = 100.0
) -> PhotoEmbeddings:
    """SOE fitted to the oracle's answers on the photos that occur in the expert triplets."""
    from triderm.llm_triplets import load_llm_triplets
    from triderm.soe import fit_soe, restrict_triplets, subsample_triplets

    raw, photo_ids = load_llm_triplets(cfg.paths.llm_triplets(oracle))
    keep = expert_photo_ids(load_expert_triplets(cfg.paths.expert_triplets))
    triplets, photo_ids = restrict_triplets(np.asarray(raw, dtype=np.int64), photo_ids, keep)
    if budget_percent < 100:
        triplets = subsample_triplets(triplets, budget_percent, cfg.soe.seed)
    soe_cfg = replace(cfg.soe, dim=dim)
    print(f"SOE: {len(triplets)} triplets, {len(photo_ids)} items, D={dim}")
    result = fit_soe(triplets, len(photo_ids), soe_cfg)
    print(f"SOE training triplet error: {result.train_error:.4f} ({result.epochs_run} epochs)")
    meta = {"oracle": oracle, "n_triplets": len(triplets), **asdict(soe_cfg)}
    return PhotoEmbeddings(result.embedding, photo_ids, metric="euclidean", meta=meta)


@app.command("fit-soe", rich_help_panel=TEXT)
def fit_soe(
    ctx: typer.Context,
    oracle: Oracle = "gpt-oss-120b",
    dim: Annotated[int | None, typer.Option(help="Embedding dimension; default soe.dim.")] = None,
) -> None:
    """Fit the TriDerm text embedding (SOE) to the oracle's triplet answers."""
    cfg = load_config(ctx.obj)
    dim = dim or cfg.soe.dim
    _soe_embedding(cfg, oracle, dim).save(cfg.paths.embedding(f"soe-{oracle}-{dim}d"))


@app.command("embed-text", rich_help_panel=TEXT)
def embed_text(
    ctx: typer.Context,
    encoder: Annotated[
        str | None, typer.Option(help="Encoder, a key of text_encoders; default all.")
    ] = None,
) -> None:
    """Sentence-encoder baselines (MiniLM, RoBERTa, PubMedBERT)."""
    from triderm.text_embed import encode_sentences

    cfg = load_config(ctx.obj)
    texts, photo_ids = load_descriptions_sorted(cfg.paths.descriptions)
    for name in [encoder] if encoder else cfg.text_encoders:
        vectors = encode_sentences(texts, cfg.text_encoders[name])
        emb = PhotoEmbeddings(vectors, photo_ids, metric="cosine", meta={"model": name})
        emb.save(cfg.paths.embedding(f"text-{name}"))


@app.command("embed-llm", rich_help_panel=TEXT)
def embed_llm(
    ctx: typer.Context,
    model: Annotated[str, typer.Option(help="Model, a key of llm_embeddings.")] = "gpt-oss-120b",
    batch_size: Annotated[int, typer.Option()] = 4,
) -> None:
    """LLM baselines: mean-pooled last-layer hidden states."""
    from triderm.text_embed import llm_mean_hidden_states

    cfg = load_config(ctx.obj)
    texts, photo_ids = load_descriptions_sorted(cfg.paths.descriptions)
    vectors = llm_mean_hidden_states(texts, cfg.llm_embeddings[model], batch_size=batch_size)
    emb = PhotoEmbeddings(vectors, photo_ids, metric="cosine", meta={"model": model})
    emb.save(cfg.paths.embedding(f"llm-{model}-mean"))


def _wound_samples(cfg: Config) -> list[WoundSample]:
    from triderm.vision.dataset import find_wound_samples

    exclude = load_excluded_photos(cfg.paths.excluded_photos)
    return find_wound_samples(cfg.paths.photos, cfg.paths.masks, exclude)


@app.command("embed-vision", rich_help_panel=VISION)
def embed_vision(
    ctx: typer.Context,
    backbone: Annotated[
        str | None, typer.Option(help="Backbone, a key of vision.zero_shot; default all.")
    ] = None,
    device: Device = None,
) -> None:
    """Zero-shot baselines: masked mean pooling of wound tokens."""
    from triderm.runtime import resolve_device
    from triderm.vision.backbones import load_backbone
    from triderm.vision.train import extract_zero_shot

    cfg = load_config(ctx.obj)
    samples = _wound_samples(cfg)
    dev = str(resolve_device(device))
    for name in [backbone] if backbone else cfg.zero_shot_input_sizes:
        emb = extract_zero_shot(load_backbone(name, dev), samples, cfg.zero_shot_input_sizes[name])
        emb.meta["backbone"] = name
        emb.save(cfg.paths.embedding(f"vision-zs-{name}"))


@app.command("train-vision", rich_help_panel=VISION)
def train_vision(
    ctx: typer.Context,
    variant: Annotated[str, typer.Option(help="Variant, a key of vision.train.")] = "triderm",
    device: Device = None,
) -> None:
    """Train the wound attention-pooling head (VICReg) on frozen DermLIP; save photo embeddings."""
    from triderm.runtime import resolve_device
    from triderm.vision.backbones import load_backbone
    from triderm.vision.train import extract_trained, train_pooler

    cfg = load_config(ctx.obj)
    vcfg = cfg.vision_variants[variant]
    samples = _wound_samples(cfg)
    backbone = load_backbone(vcfg.backbone, str(resolve_device(device)))
    pooler = train_pooler(backbone, samples, vcfg)
    extract_trained(backbone, pooler, samples, vcfg).save(cfg.paths.embedding(f"vision-{variant}"))


@app.command("evaluate", rich_help_panel=RESULTS)
def evaluate_cmd(
    ctx: typer.Context, names: Annotated[list[str], typer.Argument(help="Embedding names.")]
) -> None:
    """Agreement of saved embeddings with the expert triplets."""
    cfg = load_config(ctx.obj)
    triplets = load_expert_triplets(cfg.paths.expert_triplets)
    for name in names:
        print(_summary(name, evaluate(PhotoEmbeddings.load(cfg.paths.embedding(name)), triplets)))


@app.command("table1", rich_help_panel=RESULTS)
def table1(ctx: typer.Context) -> None:
    """Table 1: text, vision and fused embeddings against the expert triplets."""
    from triderm.report import evaluate_table1_row, latex_table, markdown_table

    cfg = load_config(ctx.obj)
    triplets = load_expert_triplets(cfg.paths.expert_triplets)
    rows = [evaluate_table1_row(cfg, row, triplets) for row in cfg.table1]
    _write_json(cfg.paths.results_dir / "table1.json", [r.as_dict() for r in rows])
    (cfg.paths.results_dir / "table1.tex").write_text(latex_table(rows) + "\n")
    print(markdown_table(rows))


@app.command("soe-ablation", rich_help_panel=RESULTS)
def soe_ablation(ctx: typer.Context) -> None:
    """Table 2a: SOE dimension, triplet budget and LLM oracle."""
    cfg = load_config(ctx.obj)
    ab = cfg.soe_ablation
    runs = (
        [("dim", d, ab.oracle, d, 100.0) for d in ab.dims]
        + [("budget_percent", p, ab.oracle, cfg.soe.dim, p) for p in ab.budget_percents]
        + [("oracle", o, o, cfg.soe.dim, 100.0) for o in ab.oracles]
    )
    triplets = load_expert_triplets(cfg.paths.expert_triplets)
    results = []
    for kind, value, oracle, dim, budget in runs:
        res = evaluate(_soe_embedding(cfg, oracle, dim, budget), triplets)
        results.append({"ablation": kind, "value": value, "oracle": oracle, **res.as_dict()})
        print(_summary(f"{kind}={value} ({oracle})", res))
    _write_json(cfg.paths.results_dir / "soe_ablation.json", results)


@app.command("vision-ablation", rich_help_panel=RESULTS)
def vision_ablation(ctx: typer.Context) -> None:
    """Table 2b: pooling and loss variants (run `train-vision --variant ...` first)."""
    cfg = load_config(ctx.obj)
    triplets = load_expert_triplets(cfg.paths.expert_triplets)
    results = []
    for variant in cfg.vision_variants:
        path = cfg.paths.embedding(f"vision-{variant}")
        if not path.exists():
            print(f"{variant}: missing {path}")
            continue
        res = evaluate(PhotoEmbeddings.load(path), triplets)
        results.append({"variant": variant, **res.as_dict()})
        print(_summary(variant, res))
    _write_json(cfg.paths.results_dir / "vision_ablation.json", results)


@app.command("neighbors", rich_help_panel=RESULTS)
def neighbors(
    ctx: typer.Context,
    fusion: Annotated[Fusion, typer.Option(help="Fusion strategy.")] = "uncertainty",
    vision: Annotated[str, typer.Option(help="Vision embedding name.")] = "vision-triderm",
    text: Annotated[str, typer.Option(help="Text embedding name.")] = "soe-gpt-oss-120b-4d",
    k: Annotated[int, typer.Option(help="Neighbours per side.")] = 2,
) -> None:
    """Figure 2: most similar and most dissimilar photos per reference under fused distance."""
    from triderm.fusion import fused_distances
    from triderm.report import nearest_and_farthest

    cfg = load_config(ctx.obj)
    photo_ids, dist = fused_distances(
        PhotoEmbeddings.load(cfg.paths.embedding(vision)),
        PhotoEmbeddings.load(cfg.paths.embedding(text)),
        fusion,
        cfg.fusion_alpha,
    )
    rows = nearest_and_farthest(photo_ids, dist, k)
    path = cfg.paths.results_dir / f"neighbors-{fusion}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")
