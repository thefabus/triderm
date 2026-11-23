"""Typed settings from `configs/paper.toml`, with environment overrides for paths and endpoint."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

DEFAULT_CONFIG = Path("configs/paper.toml")

Loss = Literal["vicreg", "triplet", "supcon"]
Pooling = Literal["attention", "mean"]
Fusion = Literal["uncertainty", "similarity"]


@dataclass(frozen=True)
class Paths:
    data_root: Path
    out_root: Path
    expert_triplets: Path
    descriptions: Path
    photos: Path
    masks: Path
    excluded_photos: Path

    @property
    def results_dir(self) -> Path:
        return self.out_root / "results"

    @property
    def llm_cache(self) -> Path:
        return self.out_root / "cache" / "llm_responses.sqlite3"

    def embedding(self, name: str) -> Path:
        return self.out_root / "embeddings" / f"{name}.npz"

    def llm_triplets(self, oracle: str) -> Path:
        return self.out_root / "triplets" / f"{oracle}.json"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    seed: int
    max_concurrent: int
    oracles: dict[str, str]


@dataclass(frozen=True)
class SOEConfig:
    dim: int = 4
    epochs: int = 50
    batch_size: int = 2048
    learning_rate: float = 0.05
    delta: float = 0.0
    patience: int = 30
    eval_every: int = 10
    seed: int = 42


@dataclass(frozen=True)
class SOEAblationConfig:
    oracle: str
    dims: list[int]
    budget_percents: list[float]
    oracles: list[str]


@dataclass(frozen=True)
class VisionTrainConfig:
    backbone: str = "dermlip"
    input_size: int = 224
    pooling: Pooling = "attention"
    loss: Loss = "vicreg"
    output_dim: int = 512
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    seed: int = 42


@dataclass(frozen=True)
class Table1Row:
    """One row of Table 1: a single embedding, or a fusion of a vision and a text embedding."""

    block: Literal["text", "vision", "fused"]
    label: str
    embedding: str | None = None
    vision: str | None = None
    text: str | None = None
    fusion: Fusion | None = None

    def __post_init__(self) -> None:
        fused = self.block == "fused"
        if fused and not (self.vision and self.text and self.fusion):
            raise ValueError(f"Table 1 row {self.label!r}: needs vision, text and fusion")
        if not fused and not self.embedding:
            raise ValueError(f"Table 1 row {self.label!r}: needs an embedding")

    @property
    def embeddings(self) -> list[str]:
        """Names of the embeddings this row reads."""
        names = [self.vision, self.text] if self.block == "fused" else [self.embedding]
        return [n for n in names if n]


@dataclass(frozen=True)
class Config:
    paths: Paths
    llm: LLMConfig
    soe: SOEConfig
    soe_ablation: SOEAblationConfig
    text_encoders: dict[str, str]
    llm_embeddings: dict[str, str]
    zero_shot_input_sizes: dict[str, int]
    vision_variants: dict[str, VisionTrainConfig]
    fusion_alpha: float
    table1: list[Table1Row]


def _paths(raw: dict[str, str]) -> Paths:
    data_root = Path(os.environ.get("TRIDERM_DATA_ROOT", raw["data_root"]))
    out_root = Path(os.environ.get("TRIDERM_OUT", raw["out_root"]))
    inputs = ("expert_triplets", "descriptions", "photos", "masks", "excluded_photos")
    return Paths(data_root, out_root, **{key: data_root / raw[key] for key in inputs})


def _vision_variants(raw: dict[str, dict[str, Any]]) -> dict[str, VisionTrainConfig]:
    """The `triderm` variant; every other entry overrides some of its fields."""
    base = VisionTrainConfig(**raw.pop("triderm"))
    allowed = {f.name for f in fields(VisionTrainConfig)}
    for name, overrides in raw.items():
        if unknown := set(overrides) - allowed:
            raise ValueError(f"vision.train.{name}: unknown keys {sorted(unknown)}")
    return {"triderm": base} | {name: replace(base, **o) for name, o in raw.items()}


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    """Load the TOML config; TRIDERM_DATA_ROOT, TRIDERM_OUT and LLM_BASE_URL override it."""
    raw = tomllib.loads(path.read_text())
    llm = raw["llm"]
    return Config(
        paths=_paths(raw["paths"]),
        llm=LLMConfig(
            base_url=os.environ.get("LLM_BASE_URL", llm["base_url"]),
            seed=llm["seed"],
            max_concurrent=llm["max_concurrent"],
            oracles=llm["oracles"],
        ),
        soe=SOEConfig(**raw["soe"]),
        soe_ablation=SOEAblationConfig(**raw["soe_ablation"]),
        text_encoders=raw["text_encoders"],
        llm_embeddings=raw["llm_embeddings"],
        zero_shot_input_sizes=raw["vision"]["zero_shot"],
        vision_variants=_vision_variants(raw["vision"]["train"]),
        fusion_alpha=raw["fusion"]["alpha"],
        table1=[Table1Row(**row) for row in raw["table1"]],
    )
