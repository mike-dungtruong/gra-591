"""Training logging extras: persistent text log and per-epoch progress plot.

We don't get nnUNet's `training_log_*.txt` and `progress.png` for free since
TextSwinUMamba runs on a plain PyTorch loop. These helpers make the run
directory self-contained enough to inspect without a separate dashboard.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import IO, Iterable


# --------------------------------------------------------------------------- #
# Persistent text log via Tee
# --------------------------------------------------------------------------- #

class _Tee:
    """Write to multiple file-like objects at once (and flush them eagerly)."""

    def __init__(self, *streams: IO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                # Don't let a broken stream kill the training loop.
                pass
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def attach_text_log(log_path: str | Path):
    """Tee stdout + stderr to `log_path` (append mode).

    Append mode means a resumed run continues writing to the same file rather
    than truncating it. Call once at the start of a training entrypoint. Returns
    the open file handle so the caller can close it cleanly at shutdown.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    return fh


# --------------------------------------------------------------------------- #
# Progress plot regenerated every epoch
# --------------------------------------------------------------------------- #

def _read_history(history_csv: Path) -> list[dict]:
    if not history_csv.exists():
        return []
    rows: list[dict] = []
    with history_csv.open() as f:
        for row in csv.DictReader(f):
            parsed: dict = {}
            for k, v in row.items():
                if v == "" or v is None:
                    parsed[k] = float("nan")
                    continue
                try:
                    parsed[k] = int(v) if k == "epoch" else float(v)
                except ValueError:
                    parsed[k] = float("nan")
            rows.append(parsed)
    return rows


def _column(rows: Iterable[dict], key: str) -> list[float]:
    return [r.get(key, float("nan")) for r in rows]


def plot_progress(
    history_csv: str | Path,
    out_path: str | Path,
    *,
    best_epoch: int | None = None,
    best_val_dice: float | None = None,
    best_val_loss: float | None = None,
) -> None:
    """Read `history.csv` and write a 4-panel `progress.png`:
        (loss curves) (val Dice + IoU)
        (learning rate) (summary text)

    Idempotent — overwrites the existing file every call. Failures are silent;
    we never want plotting to take down training.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib not installed; skip silently

    rows = _read_history(Path(history_csv))
    if not rows:
        return

    epochs = _column(rows, "epoch")
    train_loss = _column(rows, "train_loss")
    val_loss = _column(rows, "val_loss")
    val_dice = _column(rows, "val_dice")
    val_iou = _column(rows, "val_iou")
    lr = _column(rows, "lr")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (0,0) losses
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color="tab:blue", label="train_loss")
    ax.plot(epochs, val_loss, color="tab:red", label="val_loss")
    if best_epoch is not None and best_val_loss is not None:
        ax.scatter([best_epoch], [best_val_loss], color="black", zorder=5,
                   label=f"best val_loss {best_val_loss:.4f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Loss"); ax.grid(alpha=0.3); ax.legend()

    # (0,1) validation metrics
    ax = axes[0, 1]
    ax.plot(epochs, val_dice, color="tab:green", label="val_dice")
    ax.plot(epochs, val_iou,  color="tab:orange", label="val_iou")
    if best_epoch is not None and best_val_dice is not None and best_val_loss is None:
        ax.scatter([best_epoch], [best_val_dice], color="red", zorder=5,
                   label=f"best dice {best_val_dice:.4f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title("Validation metrics"); ax.grid(alpha=0.3); ax.legend()

    # (1,0) learning rate
    ax = axes[1, 0]
    ax.plot(epochs, lr, color="tab:purple")
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.set_yscale("log"); ax.set_title("Learning rate"); ax.grid(alpha=0.3, which="both")

    # (1,1) summary text
    ax = axes[1, 1]; ax.axis("off")
    last = rows[-1]
    lines = [
        f"epochs trained : {int(last['epoch']) + 1}",
        f"latest train_loss : {last.get('train_loss', float('nan')):.4f}",
        f"latest val_loss : {last.get('val_loss', float('nan')):.4f}",
        f"latest val_dice : {last.get('val_dice', float('nan')):.4f}",
        f"latest val_iou : {last.get('val_iou', float('nan')):.4f}",
        f"latest lr : {last.get('lr', float('nan')):.2e}",
        f"wall hours : {last.get('wall_hours', float('nan')):.2f}",
    ]
    if best_epoch is not None and best_val_loss is not None:
        lines.append("")
        lines.append(f"best val_loss : {best_val_loss:.4f}")
        lines.append(f"  @ epoch {best_epoch}")
    elif best_epoch is not None and best_val_dice is not None:
        lines.append("")
        lines.append(f"best val_dice : {best_val_dice:.4f}")
        lines.append(f"  @ epoch {best_epoch}")
    ax.text(0.05, 0.95, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=11)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, dpi=90)
    finally:
        plt.close(fig)
