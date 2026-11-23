"""Ask an LLM triplet questions over expert wound descriptions (the "synthetic expert")."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sqlite3
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm as async_tqdm

from triderm.data import photo_base

SYSTEM_PROMPT = (
    "You are a medical doctor specializing in Recessive Dystrophic Epidermolysis Bullosa (RDEB), "
    "a rare genetic skin disorder. You have extensive experience in evaluating patient outcomes "
    "and prognoses based on clinical, genetic, and histological findings."
)
CHOICES = ("A", "C")

# Structured output restricts the answer to A or C (guided decoding on vLLM).
AnswerChoice = Enum("Choice", {c: c for c in CHOICES})


class Answer(BaseModel):
    choice: AnswerChoice  # type: ignore[valid-type]


def triplet_prompt(desc_a: str, desc_b: str, desc_c: str) -> str:
    """Single user message asking whether patient B (the anchor) is closer to A or to C."""
    return "\n".join(
        [
            SYSTEM_PROMPT,
            "",
            "Below are descriptions of wound appearance as described by a clinician "
            "for three patients.",
            "",
            "Patient A:",
            desc_a,
            "",
            "Patient B:",
            desc_b,
            "",
            "Patient C:",
            desc_c,
            "",
            "Based on the descriptions, is Patient B's condition more similar to Patient A or "
            "Patient C?",
            "Answer with just A or C.",
        ]
    )


def all_triplet_questions(photo_ids: list[str]) -> list[tuple[int, int, int]]:
    """All (left, anchor, right) index triples, each of the three items once as anchor.

    Triples with two images of the same photo series (e.g. '26a' and '26b') are left out.
    """
    base = [photo_base(p) for p in photo_ids]
    questions = []
    for i, j, k in combinations(range(len(photo_ids)), 3):
        if base[i] == base[j] or base[j] == base[k] or base[i] == base[k]:
            continue
        questions += [(i, j, k), (j, i, k), (i, k, j)]
    return questions


def sample_questions(
    questions: list[tuple[int, int, int]], n: int | None, seed: int
) -> list[tuple[int, int, int]]:
    if n is None or n >= len(questions):
        return questions
    return random.Random(seed).sample(questions, n)


def answer_to_constraint(left: int, anchor: int, right: int, answer: str) -> tuple[int, int, int]:
    """(anchor, closer, farther) from the LLM's A/C answer."""
    return (anchor, left, right) if answer == "A" else (anchor, right, left)


class ResponseCache:
    """SQLite cache of LLM answers keyed by a hash of (model, prompt, seed, choices)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, request TEXT, "
            "response TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )

    @staticmethod
    def key(request: dict) -> str:
        return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

    def get(self, request: dict) -> str | None:
        row = self.con.execute(
            "SELECT response FROM responses WHERE key = ?", (self.key(request),)
        ).fetchone()
        return row[0] if row else None

    def set(self, request: dict, response: str) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO responses (key, request, response) VALUES (?, ?, ?)",
            (self.key(request), json.dumps(request), response),
        )
        self.con.commit()


@dataclass(frozen=True)
class LLMEndpoint:
    """OpenAI-compatible endpoint serving the oracle; the API key comes from LLM_API_KEY."""

    base_url: str
    model: str
    seed: int
    max_concurrent: int

    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(base_url=self.base_url, api_key=os.environ.get("LLM_API_KEY", "EMPTY"))


async def _ask(client: AsyncOpenAI, model: str, prompt: str, seed: int) -> str:
    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        seed=seed,
        response_format=Answer,
    )
    return completion.choices[0].message.parsed.choice.value  # type: ignore[union-attr]


async def ask_all(
    prompts: list[str], endpoint: LLMEndpoint, cache: ResponseCache
) -> list[str | None]:
    """Answers for all prompts (cache first); failed requests yield None and are not cached."""
    requests = [
        {"model": endpoint.model, "prompt": p, "seed": endpoint.seed, "choices": list(CHOICES)}
        for p in prompts
    ]
    answers: list[str | None] = [cache.get(r) for r in requests]
    todo = [i for i, a in enumerate(answers) if a is None]
    print(f"Cache: {len(prompts) - len(todo)}/{len(prompts)} cached, {len(todo)} to fetch")
    if not todo:
        return answers

    client = endpoint.client()
    semaphore = asyncio.Semaphore(endpoint.max_concurrent)

    async def fetch(i: int) -> tuple[int, str | None]:
        async with semaphore:
            try:
                return i, await _ask(client, endpoint.model, prompts[i], endpoint.seed)
            except Exception as exc:  # noqa: BLE001 - collect failures, report at the end
                print(f"Request {i} failed: {exc}")
                return i, None

    for coro in async_tqdm.as_completed([fetch(i) for i in todo], total=len(todo)):
        i, answer = await coro
        answers[i] = answer
        if answer is not None:
            cache.set(requests[i], answer)
    return answers


def run_triplet_queries(
    descriptions: list[str],
    photo_ids: list[str],
    endpoint: LLMEndpoint,
    cache: ResponseCache,
    n_questions: int | None = None,
) -> dict:
    """Ask all (or `n_questions` sampled) triplet questions; returns the JSON payload to save."""
    questions = sample_questions(all_triplet_questions(photo_ids), n_questions, endpoint.seed)
    print(f"{len(descriptions)} descriptions, {len(questions)} triplet questions")
    prompts = [
        triplet_prompt(descriptions[lf], descriptions[a], descriptions[rt])
        for lf, a, rt in questions
    ]
    answers = asyncio.run(ask_all(prompts, endpoint, cache))
    constraints = [
        answer_to_constraint(lf, a, rt, ans)
        for (lf, a, rt), ans in zip(questions, answers, strict=True)
        if ans is not None
    ]
    n_failed = sum(a is None for a in answers)
    if n_failed:
        print(f"WARNING: {n_failed} questions failed; re-run to retry them")
    return {
        "metadata": {
            "model": endpoint.model,
            "seed": endpoint.seed,
            "n_items": len(photo_ids),
            "n_questions": len(questions),
            "n_answered": len(constraints),
        },
        "photo_ids": photo_ids,
        "triplets": constraints,
    }


def load_llm_triplets(path: Path) -> tuple[list[list[int]], list[str]]:
    """(anchor, closer, farther) index triplets and the photo IDs they index."""
    payload = json.loads(path.read_text())
    return payload["triplets"], [str(p) for p in payload["photo_ids"]]
