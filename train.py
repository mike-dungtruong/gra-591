"""Training entry point for TextSwinUMambaD on ISIC 2017.

Features:
  - YAML config (configs/isic2017.yaml).
  - Mixed precision (AMP) by default.
  - Encoder freeze for the first `freeze_encoder_epochs`.
  - Cosine LR with linear warmup.
  - Checkpoint resume from `last.pth` (atomic save + RNG restore).
  - TensorBoard logging to the run directory (mountable to Drive for Colab).
  - Wall-clock budget so Colab kicks-offs don't lose work.

Two text modes:
  - 'features' (recommended for Colab): precompute BERT features once with
    scripts/precompute_text_features.py, then training is a tensor lookup.
  - 'tokens': run BERT every batch. Simpler but slower / more memory.

Usage:
    python train.py --config configs/isic2017.yaml
    python train.py --config configs/isic2017.yaml --resume auto
"""
from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Local imports
sys.path.append(str(Path(__file__).resolve().parent))
from data.isic_dataset import build_isic_dataset
from data.transforms import train_transform, val_transform
from models.text_encoder import FrozenBertTextEncoder
from models.text_swin_umamba_d import build_text_swin_umamba_d
from utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from utils.losses import DiceBCEWithDeepSupervision
from utils.metrics import binary_dice, binary_iou, first_scale
from utils.misc import (
    AverageMeter,
    WallClockBudget,
    config_hash,
    count_parameters,
    ensure_dir,
    set_seed,
)

# ----------------------------- helpers -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", default="auto",
                   help="'auto' (load last.pth if present), 'none', or path to checkpoint")
    p.add_argument("--text_mode", default=None, choices=[None, "tokens", "features"],
                   help="Override text mode from config")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_lr_lambda(total_epochs: int, warmup_epochs: int):
    def fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


def append_history(history_csv: Path, row: dict) -> None:
    new_file = not history_csv.exists()
    with history_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


# ----------------------------- main -----------------------------

def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    text_mode = args.text_mode or ("features" if Path(cfg["data"].get(
        "text_features_cache", "")).exists() else "tokens")
    print(f"[info] text_mode = {text_mode}")

    run_dir = ensure_dir(Path(cfg["output"]["base_dir"]) / cfg["run_name"])
    print(f"[info] run_dir = {run_dir}")
    cfg_hash = config_hash(cfg)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    # --------- model ---------
    text_dim = 768  # bert-base hidden size
    model = build_text_swin_umamba_d(
        num_input_channels=cfg["model"]["num_input_channels"],
        num_classes=cfg["model"]["num_classes"],
        features_per_stage=tuple(cfg["model"]["features_per_stage"]),
        d_state=cfg["model"]["d_state"],
        drop_path_rate=cfg["model"]["drop_path_rate"],
        deep_supervision=cfg["model"]["deep_supervision"],
        text_dim=text_dim,
        tgcm_k=cfg["model"]["tgcm"]["k_filters"],
        tgcm_kernel=cfg["model"]["tgcm"]["kernel_size"],
        tgcm_iterative=cfg["model"]["tgcm"]["iterative"],
        tgcm_beta_init=cfg["model"]["tgcm"]["beta_init"],
        pretrained_ckpt=cfg["model"].get("pretrained_ckpt"),
    ).to(device)

    text_encoder = None
    if text_mode == "tokens":
        text_encoder = FrozenBertTextEncoder(
            model_name=cfg["text"]["model_name"],
            pool=cfg["text"]["pool"],
            freeze=cfg["text"]["freeze"],
        ).to(device)
        text_encoder.eval()

    print(f"[info] model params: {count_parameters(model)/1e6:.2f}M "
          f"(trainable: {count_parameters(model, True)/1e6:.2f}M)")

    # --------- data ---------
    tokenizer = text_encoder.tokenizer if text_encoder is not None else None
    train_ds = build_isic_dataset(
        root=cfg["data"]["isic_root"],
        split="train",
        captions_jsonl=cfg["data"]["captions_jsonl"],
        transform=train_transform(cfg["data"]["image_size"]),
        text_mode=text_mode,
        tokenizer=tokenizer,
        text_max_length=cfg["text"]["max_length"],
        text_features_cache=cfg["data"].get("text_features_cache"),
        require_caption=True,
    )
    val_ds = build_isic_dataset(
        root=cfg["data"]["isic_root"],
        split="val",
        captions_jsonl=cfg["data"]["captions_jsonl"],
        transform=val_transform(cfg["data"]["image_size"]),
        text_mode=text_mode,
        tokenizer=tokenizer,
        text_max_length=cfg["text"]["max_length"],
        text_features_cache=cfg["data"].get("text_features_cache"),
        require_caption=True,
    )
    print(f"[info] dataset sizes: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    # --------- optim / sched / loss / scaler ---------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(cfg["train"]["epochs"], cfg["train"]["warmup_epochs"]),
    )
    criterion = DiceBCEWithDeepSupervision(
        dice_weight=cfg["loss"]["dice_weight"],
        bce_weight=cfg["loss"]["bce_weight"],
        deep_supervision_weights=cfg["loss"]["deep_supervision_weights"],
    )
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --------- resume ---------
    start_epoch = 0
    global_step = 0
    best_val_dice = 0.0
    best_epoch = -1

    resume_path = None
    if args.resume == "auto":
        resume_path = find_latest_checkpoint(run_dir)
    elif args.resume not in ("none", ""):
        resume_path = Path(args.resume)
    if resume_path is not None and Path(resume_path).exists():
        print(f"[resume] loading {resume_path}")
        bookkeeping = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_amp else None,
            expected_config_hash=cfg_hash,
            map_location=device,
        )
        start_epoch = bookkeeping["epoch"] + 1
        global_step = bookkeeping["global_step"]
        best_val_dice = bookkeeping["best_val_dice"]
        best_epoch = bookkeeping["best_epoch"]
        print(f"[resume] start_epoch={start_epoch} best_val_dice={best_val_dice:.4f}")

    # Apply encoder-freeze policy based on current epoch (handles resume mid-schedule).
    if start_epoch < cfg["train"]["freeze_encoder_epochs"]:
        model.freeze_encoder()
        print(f"[info] encoder frozen for first {cfg['train']['freeze_encoder_epochs']} epochs")
    else:
        model.unfreeze_encoder()

    # --------- logging ---------
    writer = SummaryWriter(log_dir=str(run_dir / "tb")) if cfg["output"]["tensorboard"] else None
    history_csv = run_dir / "history.csv"
    budget = WallClockBudget(cfg["train"].get("max_hours", 1e9))

    # --------- graceful SIGINT save ---------
    interrupted = {"flag": False}

    def _sigint(signum, frame):
        print("\n[signal] caught SIGINT, saving last.pth then exiting...", flush=True)
        interrupted["flag"] = True
    signal.signal(signal.SIGINT, _sigint)

    # --------- training loop ---------
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if epoch == cfg["train"]["freeze_encoder_epochs"]:
            print(f"[info] epoch {epoch}: unfreezing encoder")
            model.unfreeze_encoder()
            # Rebuild optimizer to include newly-trainable params.
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg["train"]["lr"],
                weight_decay=cfg["train"]["weight_decay"],
            )

        # ---- train epoch ----
        model.train()
        if text_encoder is not None:
            text_encoder.eval()
        loss_meter = AverageMeter()
        t0 = time.time()
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            if text_mode == "tokens":
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attn_mask = batch["attention_mask"].to(device, non_blocking=True)
                with torch.no_grad():
                    text_pooled, _ = text_encoder(input_ids, attn_mask)
            else:
                text_pooled = batch["text_pooled"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(image, text_pooled)
                loss = criterion(output, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), n=image.size(0))
            global_step += 1

            if interrupted["flag"]:
                break

        scheduler.step()

        # ---- val ----
        model.eval()
        dice_meter = AverageMeter()
        iou_meter = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                if text_mode == "tokens":
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attn_mask = batch["attention_mask"].to(device, non_blocking=True)
                    text_pooled, _ = text_encoder(input_ids, attn_mask)
                else:
                    text_pooled = batch["text_pooled"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = model(image, text_pooled)
                logits = first_scale(output)
                d = binary_dice(logits, mask).item()
                i = binary_iou(logits, mask).item()
                dice_meter.update(d, n=image.size(0))
                iou_meter.update(i, n=image.size(0))

        epoch_time = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d} | "
            f"train_loss {loss_meter.avg:.4f} | "
            f"val_dice {dice_meter.avg:.4f} | val_iou {iou_meter.avg:.4f} | "
            f"lr {lr_now:.2e} | t {epoch_time:.1f}s | "
            f"wall {budget.elapsed_hours():.2f}h"
        )

        # ---- log ----
        if writer is not None:
            writer.add_scalar("train/loss", loss_meter.avg, epoch)
            writer.add_scalar("val/dice", dice_meter.avg, epoch)
            writer.add_scalar("val/iou", iou_meter.avg, epoch)
            writer.add_scalar("train/lr", lr_now, epoch)
            writer.add_scalar("time/epoch_seconds", epoch_time, epoch)
        append_history(history_csv, {
            "epoch": epoch, "train_loss": loss_meter.avg,
            "val_dice": dice_meter.avg, "val_iou": iou_meter.avg,
            "lr": lr_now, "epoch_seconds": epoch_time,
            "wall_hours": budget.elapsed_hours(),
        })

        # ---- save ----
        improved = dice_meter.avg > best_val_dice
        if improved:
            best_val_dice = dice_meter.avg
            best_epoch = epoch
        save_checkpoint(
            run_dir / "last.pth",
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler if use_amp else None,
            epoch=epoch, global_step=global_step,
            best_val_dice=best_val_dice, best_epoch=best_epoch,
            config_hash=cfg_hash,
        )
        if improved and cfg["output"]["keep_best"]:
            save_checkpoint(
                run_dir / "best.pth",
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler if use_amp else None,
                epoch=epoch, global_step=global_step,
                best_val_dice=best_val_dice, best_epoch=best_epoch,
                config_hash=cfg_hash,
            )
            print(f"  -> new best, saved best.pth (epoch {best_epoch}, dice {best_val_dice:.4f})")

        # ---- bail-out conditions ----
        if interrupted["flag"]:
            print("[info] graceful exit after save.")
            break
        if budget.exhausted():
            print(f"[info] wall-clock budget exhausted ({budget.elapsed_hours():.2f}h). "
                  "Saved last.pth — resume next session.")
            break

    if writer is not None:
        writer.close()
    print(f"[done] best val_dice {best_val_dice:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
