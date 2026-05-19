"""Small utilities: seeding, average meter, logging helpers."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_hash(cfg: dict) -> str:
    """Stable short hash of a config dict — used to verify resume compatibility."""
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


class AverageMeter:
    """Tracks a running average over a training epoch."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class WallClockBudget:
    """Stops training cleanly before Colab kicks us off."""

    def __init__(self, max_hours: float) -> None:
        self.max_seconds = max_hours * 3600.0
        self.start = time.time()

    def exhausted(self) -> bool:
        return (time.time() - self.start) >= self.max_seconds

    def elapsed_hours(self) -> float:
        return (time.time() - self.start) / 3600.0


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
