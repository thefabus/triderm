"""Torch device selection and global seeding."""

from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(device: str | None = None) -> torch.device:
    """The given device, else CUDA, MPS or CPU (first available)."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
