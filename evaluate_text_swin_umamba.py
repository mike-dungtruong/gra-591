"""Standalone evaluation for TextSwinUMamba (CNN decoder + TGCM text injection).

Usage:
    python evaluate_text_swin_umamba.py \
        --config configs/isic2017_text_swin_umamba.yaml \
        --ckpt runs/text_swin_umamba_isic2017/best.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.isic_dataset import build_isic_dataset
from src.data.transforms import val_transform
from src.models.text_encoder import FrozenBertTextEncoder
from src.models.text_swin_umamba import build_text_swin_umamba
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import (
    binary_confusion_counts,
    first_scale,
    vmunet_metrics_from_counts,
)


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

    model = build_text_swin_umamba(
        num_input_channels=cfg["model"]["num_input_channels"],
        num_classes=cfg["model"]["num_classes"],
        feat_size=cfg["model"].get("feat_size", [48, 96, 192, 384, 768]),
        drop_path_rate=cfg["model"]["drop_path_rate"],
        deep_supervision=cfg["model"]["deep_supervision"],
        pretrained_ckpt=None,  # weights loaded from --ckpt
        text_dim=768,
        tgcm_k=cfg["model"]["tgcm"]["k_filters"],
        tgcm_kernel=cfg["model"]["tgcm"]["kernel_size"],
        tgcm_iterative=cfg["model"]["tgcm"]["iterative"],
        tgcm_beta_init=cfg["model"]["tgcm"]["beta_init"],
        tgcm_enabled=cfg["model"]["tgcm"].get("enabled", True),
    ).to(device)

    bookkeeping = load_checkpoint(args.ckpt, model=model, map_location=device)
    print(f"loaded ckpt: epoch={bookkeeping['epoch']} "
          f"best_val_f1_or_dsc={bookkeeping['best_val_dice']:.4f}")

    text_encoder = None
    tokenizer = None
    if text_mode == "tokens":
        text_encoder = FrozenBertTextEncoder(
            model_name=cfg["text"]["model_name"],
            pool=cfg["text"]["pool"],
            freeze=True,
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
            if text_mode == "tokens":
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attn_mask = batch["attention_mask"].to(device, non_blocking=True)
                text_pooled, _ = text_encoder(input_ids, attn_mask)
            else:
                text_pooled = batch["text_pooled"].to(device, non_blocking=True)

            output = model(image, text_pooled)
            logits = first_scale(output)
            b_tp, b_fp, b_fn, b_tn = binary_confusion_counts(logits, mask)
            tp += b_tp
            fp += b_fp
            fn += b_fn
            tn += b_tn

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
