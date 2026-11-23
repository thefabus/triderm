"""Text baselines: sentence-encoder embeddings and mean-pooled LLM hidden states."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


def encode_sentences(texts: list[str], model_id: str, batch_size: int = 32) -> np.ndarray:
    """Sentence-transformers embeddings (model default pooling)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)


def llm_mean_hidden_states(
    texts: list[str], model_id: str, batch_size: int = 4, max_length: int = 2048
) -> np.ndarray:
    """Mean over all non-padding tokens of the last hidden layer of a causal LLM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", output_hidden_states=True
    )
    model.eval()

    out = []
    for start in tqdm(range(0, len(texts), batch_size), desc="LLM hidden states"):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(model.device)
        with torch.no_grad():
            hidden = model(**inputs).hidden_states[-1]
        for j in range(len(batch)):
            # Assumes right padding (the tokenizer default for these models).
            seq_len = int(inputs["attention_mask"][j].sum().item())
            out.append(hidden[j, :seq_len].mean(dim=0).float().cpu().numpy())
    return np.stack(out).astype(np.float32)
