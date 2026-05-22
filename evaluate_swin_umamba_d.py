"""Standalone evaluation for SwinUMamba-D (Mamba encoder + Mamba decoder).

Usage:
    python evaluate_swin_umamba_d.py --config configs/isic2017_swin_umamba_d.yaml \
        --ckpt runs/swin_umamba_d_isic2017/checkpoints/best.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.isic_dataset import build_isic_dataset
from src.data.transforms import val_transform
from src.models.swin_umamba_d import build_swin_umamba_d
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import binary_confusion_counts, first_scale, vmunet_metrics_from_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_swin_umamba_d(
        num_input_channels=cfg["model"]["num_input_channels"],
        num_classes=cfg["model"]["num_classes"],
        features_per_stage=tuple(cfg["model"]["features_per_stage"]),
        d_state=cfg["model"]["d_state"],
        drop_path_rate=cfg["model"]["drop_path_rate"],
        deep_supervision=cfg["model"]["deep_supervision"],
        pretrained_ckpt=None,  # weights loaded from --ckpt
    ).to(device)

    bookkeeping = load_checkpoint(args.ckpt, model=model, map_location=device)
    print(f"loaded ckpt: epoch={bookkeeping['epoch']} "
          f"best_val_f1_or_dsc={bookkeeping['best_val_dice']:.4f}")

    val_ds = build_isic_dataset(
        root=cfg["data"]["isic_root"],
        split="val",
        transform=val_transform(cfg["data"]["image_size"]),
        text_mode="none",
        image_glob=cfg["data"].get("image_glob", "ISIC_*.jpg"),
        mask_template=cfg["data"].get("mask_template", "{stem}_segmentation.png"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    model.eval()
    tp = torch.tensor(0.0, dtype=torch.float64, device=device)
    fp = torch.tensor(0.0, dtype=torch.float64, device=device)
    fn = torch.tensor(0.0, dtype=torch.float64, device=device)
    tn = torch.tensor(0.0, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            output = model(image)
            logits = first_scale(output)
            b_tp, b_fp, b_fn, b_tn = binary_confusion_counts(logits, mask)
            tp += b_tp; fp += b_fp; fn += b_fn; tn += b_tn

    metrics = {k: v.item() for k, v in vmunet_metrics_from_counts(tp, fp, fn, tn).items()}
    print(
        f"miou: {metrics['miou']:.4f} | "
        f"f1_or_dsc: {metrics['f1_or_dsc']:.4f} | "
        f"accuracy: {metrics['accuracy']:.4f} | "
        f"specificity: {metrics['specificity']:.4f} | "
        f"sensitivity: {metrics['sensitivity']:.4f}"
    )
    print(
        "confusion_matrix: "
        f"TN={int(tn.item())}, FP={int(fp.item())}, "
        f"FN={int(fn.item())}, TP={int(tp.item())}"
    )


if __name__ == "__main__":
    main()
