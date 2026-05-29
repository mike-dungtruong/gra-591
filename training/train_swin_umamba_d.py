"""Training entry point for SwinUMamba-D (Mamba encoder + Mamba decoder) on ISIC 2017.

This is the baseline without text guidance. Compared to TextSwinUMambaD:
  - Uses SwinUMambaD (Mamba decoder) instead of TextSwinUMambaD
  - No text encoder, no captions, no text features
  - Dataset loaded with text_mode="none"

Usage:
    python training/train_swin_umamba_d.py --config configs/isic2017_swin_umamba_d.yaml
    python training/train_swin_umamba_d.py --config configs/isic2017_swin_umamba_d.yaml --resume auto
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.isic_dataset import build_isic_dataset
from src.data.transforms import train_transform, val_transform
from src.models.swin_umamba_d import build_swin_umamba_d
from src.utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.utils.logging import attach_text_log, plot_progress
from src.utils.losses import DiceBCEWithDeepSupervision
from src.utils.metrics import binary_confusion_counts, first_scale, vmunet_metrics_from_counts
from src.utils.misc import (
    AverageMeter,
    WallClockBudget,
    config_hash,
    count_parameters,
    ensure_dir,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", default="auto",
                   help="'auto' (load last.pth if present), 'none', or path to checkpoint")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def append_history(history_csv: Path, row: dict) -> None:
    new_file = not history_csv.exists()
    with history_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    run_dir = ensure_dir(Path(cfg["output"]["base_dir"]) / cfg["run_name"])
    log_fh = attach_text_log(run_dir / "training_log.txt")
    print(f"[info] run_dir = {run_dir}")
    cfg_hash = config_hash(cfg)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    # --------- model ---------
    model = build_swin_umamba_d(
        num_input_channels=cfg["model"]["num_input_channels"],
        num_classes=cfg["model"]["num_classes"],
        features_per_stage=tuple(cfg["model"]["features_per_stage"]),
        d_state=cfg["model"]["d_state"],
        drop_path_rate=cfg["model"]["drop_path_rate"],
        deep_supervision=cfg["model"]["deep_supervision"],
        pretrained_ckpt=cfg["model"].get("pretrained_ckpt"),
    ).to(device)

    print(f"[info] model params: {count_parameters(model)/1e6:.2f}M "
          f"(trainable: {count_parameters(model, True)/1e6:.2f}M)")

    # --------- data ---------
    image_glob    = cfg["data"].get("image_glob",    "ISIC_*.jpg")
    mask_template = cfg["data"].get("mask_template", "{stem}_segmentation.png")
    train_ds = build_isic_dataset(
        root=cfg["data"]["isic_root"],
        split="train",
        transform=train_transform(cfg["data"]["image_size"]),
        text_mode="none",
        image_glob=image_glob,
        mask_template=mask_template,
    )
    val_ds = build_isic_dataset(
        root=cfg["data"]["isic_root"],
        split="val",
        transform=val_transform(cfg["data"]["image_size"]),
        text_mode="none",
        image_glob=image_glob,
        mask_template=mask_template,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["scheduler_t_max"],
        eta_min=cfg["train"]["scheduler_eta_min"],
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
    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}
    best_epoch = -1
    no_improve = 0

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
        if (
            bookkeeping.get("monitor_metric") == "val_loss"
            and bookkeeping.get("best_monitor_value") is not None
        ):
            best_val_loss = float(bookkeeping["best_monitor_value"])
            best_epoch = bookkeeping["best_epoch"]
            best_metrics = bookkeeping.get("best_metrics", {})
            no_improve = bookkeeping.get("no_improve", 0)
        else:
            print("[resume] checkpoint has no val_loss monitor state; starting loss-based patience fresh.")
        print(f"[resume] start_epoch={start_epoch} best_val_loss={best_val_loss:.4f} no_improve={no_improve}")

    if start_epoch < cfg["train"]["freeze_encoder_epochs"]:
        model.freeze_encoder()
        print(f"[info] encoder frozen for first {cfg['train']['freeze_encoder_epochs']} epochs")
    else:
        model.unfreeze_encoder()

    # --------- logging ---------
    history_csv = run_dir / "history.csv"
    budget = WallClockBudget(cfg["train"].get("max_hours", 1e9))

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
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg["train"]["lr"],
                weight_decay=cfg["train"]["weight_decay"],
            )
            for pg in optimizer.param_groups:
                pg.setdefault('initial_lr', pg['lr'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg["train"]["scheduler_t_max"],
                eta_min=cfg["train"]["scheduler_eta_min"],
                last_epoch=epoch - 1,
            )

        # ---- train epoch ----
        model.train()
        loss_meter = AverageMeter()
        t0 = time.time()
        accum_steps = max(1, cfg["train"].get("grad_accum_steps", 1))
        for micro_step, batch in enumerate(train_loader):
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            if micro_step % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(image)
                loss = criterion(output, mask) / accum_steps
            scaler.scale(loss).backward()

            is_update_step = (micro_step + 1) % accum_steps == 0 or (micro_step + 1 == len(train_loader))
            if is_update_step:
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

            loss_meter.update(loss.item() * accum_steps, n=image.size(0))

            if interrupted["flag"]:
                break

        scheduler.step()

        # ---- val ----
        model.eval()
        tp = torch.tensor(0.0, dtype=torch.float64, device=device)
        fp = torch.tensor(0.0, dtype=torch.float64, device=device)
        fn = torch.tensor(0.0, dtype=torch.float64, device=device)
        tn = torch.tensor(0.0, dtype=torch.float64, device=device)
        val_loss_meter = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = model(image)
                    val_loss = criterion(output, mask)
                val_loss_meter.update(val_loss.item(), n=image.size(0))
                logits = first_scale(output)
                b_tp, b_fp, b_fn, b_tn = binary_confusion_counts(logits, mask)
                tp += b_tp; fp += b_fp; fn += b_fn; tn += b_tn

        val_metrics = vmunet_metrics_from_counts(tp, fp, fn, tn)
        val_miou = val_metrics["miou"].item()
        val_f1_or_dsc = val_metrics["f1_or_dsc"].item()
        val_accuracy = val_metrics["accuracy"].item()
        val_specificity = val_metrics["specificity"].item()
        val_sensitivity = val_metrics["sensitivity"].item()

        epoch_time = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d} | "
            f"train_loss {loss_meter.avg:.4f} | "
            f"val_loss {val_loss_meter.avg:.4f} | "
            f"miou {val_miou:.4f} | f1_or_dsc {val_f1_or_dsc:.4f} | "
            f"accuracy {val_accuracy:.4f} | specificity {val_specificity:.4f} | "
            f"sensitivity {val_sensitivity:.4f} | "
            f"lr {lr_now:.2e} | t {epoch_time:.1f}s | "
            f"wall {budget.elapsed_hours():.2f}h"
        )

        epoch_metrics = {
            "train_loss": loss_meter.avg,
            "val_loss": val_loss_meter.avg,
            "val_miou": val_miou,
            "val_f1_or_dsc": val_f1_or_dsc,
            "val_accuracy": val_accuracy,
            "val_specificity": val_specificity,
            "val_sensitivity": val_sensitivity,
            "val_dice": val_f1_or_dsc,
            "val_iou": val_miou,
        }
        append_history(history_csv, {
            "epoch": epoch, "train_loss": loss_meter.avg,
            "val_loss": val_loss_meter.avg,
            "val_dice": val_f1_or_dsc, "val_iou": val_miou,
            "lr": lr_now, "epoch_seconds": epoch_time,
            "wall_hours": budget.elapsed_hours(),
            "val_miou": val_miou, "val_f1_or_dsc": val_f1_or_dsc,
            "val_accuracy": val_accuracy, "val_specificity": val_specificity,
            "val_sensitivity": val_sensitivity,
        })

        improved = val_loss_meter.avg < best_val_loss
        if improved:
            best_val_loss = val_loss_meter.avg
            best_metrics = epoch_metrics.copy()
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        best_val_dice = best_metrics.get("val_f1_or_dsc", 0.0)

        plot_progress(
            history_csv, run_dir / "progress.png",
            best_epoch=best_epoch if best_epoch >= 0 else None,
            best_val_dice=best_val_dice if best_epoch >= 0 else None,
            best_val_loss=best_val_loss if best_epoch >= 0 else None,
        )
        save_checkpoint(
            run_dir / "last.pth",
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler if use_amp else None,
            epoch=epoch, global_step=global_step,
            best_val_dice=best_val_dice, best_epoch=best_epoch,
            config_hash=cfg_hash, no_improve=no_improve,
            monitor_metric="val_loss", monitor_mode="min",
            best_monitor_value=best_val_loss if best_epoch >= 0 else None,
            latest_metrics=epoch_metrics, best_metrics=best_metrics,
        )
        if improved and cfg["output"]["keep_best"]:
            save_checkpoint(
                run_dir / "best.pth",
                model=model, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler if use_amp else None,
                epoch=epoch, global_step=global_step,
                best_val_dice=best_val_dice, best_epoch=best_epoch,
                config_hash=cfg_hash,
                monitor_metric="val_loss", monitor_mode="min",
                best_monitor_value=best_val_loss,
                latest_metrics=epoch_metrics, best_metrics=best_metrics,
            )
            print(
                f"  -> new best, saved best.pth "
                f"(epoch {best_epoch}, val_loss {best_val_loss:.4f}, f1_or_dsc {best_val_dice:.4f})"
            )

        if interrupted["flag"]:
            print("[info] graceful exit after save.")
            break
        patience = cfg["train"].get("patience", 50)
        if no_improve >= patience:
            print(f"[info] early stopping: val_loss did not improve for {patience} epochs.")
            break
        if budget.exhausted():
            print(f"[info] wall-clock budget exhausted ({budget.elapsed_hours():.2f}h). "
                  "Saved last.pth — resume next session.")
            break

    print(f"[done] best val_loss {best_val_loss:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
