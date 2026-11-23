set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

# Lint and test on synthetic data
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest -q

# Text branch: LLM triplet answers, then SOE (needs an endpoint serving the oracle)
text oracle="gpt-oss-120b":
    uv run triderm ask-triplets --oracle {{oracle}}
    uv run triderm fit-soe --oracle {{oracle}}

# Text baselines: sentence encoders and LLM mean pooling (large GPU)
text-baselines:
    uv run triderm embed-text
    uv run triderm embed-llm --model gpt-oss-120b
    uv run triderm embed-llm --model medgemma-27b

# Vision: zero-shot baselines and the TriDerm vision branch (GPU)
vision:
    uv run triderm embed-vision
    uv run triderm train-vision --variant triderm

# Table 1 (after text, text-baselines and vision)
table1:
    uv run triderm table1

# Table 2a: SOE dimension, triplet budget and LLM oracle
soe-ablation:
    uv run triderm ask-triplets --oracle gpt-oss-20b
    uv run triderm ask-triplets --oracle medgemma-27b
    uv run triderm soe-ablation

# Table 2b: pooling and loss variants
vision-ablation:
    for v in mean-pooling triplet contrastive; do uv run triderm train-vision --variant $v; done
    uv run triderm vision-ablation

# Figure 2: nearest and farthest neighbours under uncertainty-aware fusion
neighbors:
    uv run triderm neighbors --fusion uncertainty

# Everything, in paper order
paper: text text-baselines vision table1 soe-ablation vision-ablation neighbors
