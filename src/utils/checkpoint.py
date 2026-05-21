"""Checkpoint save / resume with RNG state, atomic writes, and config-hash sanity check.

Designed for Colab: a checkpoint can survive a session disconnect mid-save.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    best_val_dice: float,
    best_epoch: int,
    config_hash: str,
    no_improve: int = 0,
) -> None:
    """Atomic save: write to .tmp then replace, so a crash mid-write cannot corrupt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_dice": best_val_dice,
        "best_epoch": best_epoch,
        "no_improve": no_improve,
        "rng_states": capture_rng_state(),
        "config_hash": config_hash,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    expected_config_hash: str | None = None,
    strict_model: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load a checkpoint and (optionally) verify config compatibility.

    Returns a small dict with bookkeeping fields the caller needs (epoch, best, etc.).
    """
    ckpt = torch.load(path, map_location=map_location)
    if expected_config_hash is not None and ckpt.get("config_hash") != expected_config_hash:
        print(
            f"[warn] Config hash mismatch: expected {expected_config_hash}, "
            f"got {ckpt.get('config_hash')}. Resuming anyway."
        )
    model.load_state_dict(ckpt["model_state_dict"], strict=strict_model)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if ckpt.get("rng_states") is not None:
        restore_rng_state(ckpt["rng_states"])
    return {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "best_val_dice": ckpt.get("best_val_dice", 0.0),
        "best_epoch": ckpt.get("best_epoch", -1),
        "no_improve": ckpt.get("no_improve", 0),
    }


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Look for last.pth in a run directory; return None if not found."""
    p = Path(run_dir) / "last.pth"
    return p if p.exists() else None
