"""Standalone evaluation: load a checkpoint, run on the val set, report Dice/IoU.

Usage:
    python evaluate.py --config configs/isic2017.yaml --ckpt runs/<run>/best.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))
from data.isic_dataset import build_isic_dataset
from data.transforms import val_transform
from models.text_encoder import FrozenBertTextEncoder
from models.text_swin_umamba_d import build_text_swin_umamba_d
from utils.checkpoint import load_checkpoint
from utils.metrics import (binary_accuracy, binary_dice, binary_iou,
                           binary_sensitivity, binary_specificity, first_scale)
from utils.misc import AverageMeter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--text_mode", default=None, choices=[None, "tokens", "features"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    text_mode = args.text_mode or ("features" if Path(cfg["data"].get(
        "text_features_cache", "")).exists() else "tokens")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_text_swin_umamba_d(
        num_input_channels=cfg["model"]["num_input_channels"],
        num_classes=cfg["model"]["num_classes"],
        features_per_stage=tuple(cfg["model"]["features_per_stage"]),
        d_state=cfg["model"]["d_state"],
        drop_path_rate=cfg["model"]["drop_path_rate"],
        deep_supervision=cfg["model"]["deep_supervision"],
        text_dim=768,
        tgcm_k=cfg["model"]["tgcm"]["k_filters"],
        tgcm_kernel=cfg["model"]["tgcm"]["kernel_size"],
        tgcm_iterative=cfg["model"]["tgcm"]["iterative"],
        tgcm_beta_init=cfg["model"]["tgcm"]["beta_init"],
        pretrained_ckpt=None,  # we'll load from --ckpt
    ).to(device)

    bookkeeping = load_checkpoint(args.ckpt, model=model, map_location=device)
    print(f"loaded ckpt: epoch={bookkeeping['epoch']} "
          f"best_val_dice={bookkeeping['best_val_dice']:.4f}")

    text_encoder = None
    tokenizer = None
    if text_mode == "tokens":
        text_encoder = FrozenBertTextEncoder(
            model_name=cfg["text"]["model_name"], pool=cfg["text"]["pool"], freeze=True
        ).to(device)
        text_encoder.eval()
        tokenizer = text_encoder.tokenizer

    val_ds = build_isic_dataset(
        root=cfg["data"]["isic_root"],
        split="val",
        captions_jsonl=cfg["data"]["captions_jsonl"],
        transform=val_transform(cfg["data"]["image_size"]),
        text_mode=text_mode,
        tokenizer=tokenizer,
        text_max_length=cfg["text"]["max_length"],
        text_features_cache=cfg["data"].get("text_features_cache"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    model.eval()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    acc_meter = AverageMeter()
    sen_meter = AverageMeter()
    spe_meter = AverageMeter()
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
            output = model(image, text_pooled)
            logits = first_scale(output)
            n = image.size(0)
            dice_meter.update(binary_dice(logits, mask).item(), n=n)
            iou_meter.update(binary_iou(logits, mask).item(), n=n)
            acc_meter.update(binary_accuracy(logits, mask).item(), n=n)
            sen_meter.update(binary_sensitivity(logits, mask).item(), n=n)
            spe_meter.update(binary_specificity(logits, mask).item(), n=n)

    print(f"mIoU {iou_meter.avg:.4f} | DSC {dice_meter.avg:.4f} | "
          f"Acc {acc_meter.avg:.4f} | Spe {spe_meter.avg:.4f} | Sen {sen_meter.avg:.4f}")


if __name__ == "__main__":
    main()
