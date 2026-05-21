"""Segmentation metrics, computed at the highest-resolution output.

VM-UNet's ISIC evaluation flattens the full validation set, thresholds predictions
and masks, builds one global confusion matrix, then computes metrics from global
TP/FP/FN/TN. The count-based helpers below mirror that behavior.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def binary_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
                eps: float = 1e-6) -> torch.Tensor:
    """Per-sample binary Dice, then mean over batch.

    Args:
        logits: (B, 1, H, W).
        target: (B, 1, H, W) in {0, 1}.
    """
    pred = (torch.sigmoid(logits) > threshold).float()
    target = target.float()
    pred_f = pred.flatten(1)
    targ_f = target.flatten(1)
    intersection = (pred_f * targ_f).sum(dim=1)
    denom = pred_f.sum(dim=1) + targ_f.sum(dim=1)
    dice = (2 * intersection + eps) / (denom + eps)
    return dice.mean()


@torch.no_grad()
def binary_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
               eps: float = 1e-6) -> torch.Tensor:
    """Per-sample foreground IoU, then mean over batch."""
    pred = (torch.sigmoid(logits) > threshold).float()
    target = target.float()
    pred_f = pred.flatten(1)
    targ_f = target.flatten(1)
    intersection = (pred_f * targ_f).sum(dim=1)
    union = pred_f.sum(dim=1) + targ_f.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


@torch.no_grad()
def binary_confusion_counts(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dataset-additive TP, FP, FN, TN counts for binary masks.

    This matches VM-UNet's ISIC path except our model emits raw logits, so sigmoid
    is applied before thresholding.
    """
    pred = torch.sigmoid(logits) >= threshold
    target = target >= 0.5
    tp = (pred & target).sum(dtype=torch.float64)
    fp = (pred & ~target).sum(dtype=torch.float64)
    fn = (~pred & target).sum(dtype=torch.float64)
    tn = (~pred & ~target).sum(dtype=torch.float64)
    return tp, fp, fn, tn


def _safe_div(num: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    """VM-UNet returns 0 when a metric denominator is zero."""
    return torch.where(denom != 0, num / denom, torch.zeros_like(num))


def dice_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor,
                     eps: float | None = None) -> torch.Tensor:
    del eps  # kept for backward-compatible calls; VM-UNet uses no smoothing here.
    return _safe_div(2 * tp, 2 * tp + fp + fn)


def foreground_iou_from_counts(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor,
                               eps: float | None = None) -> torch.Tensor:
    del eps
    return _safe_div(tp, tp + fp + fn)


def two_class_miou_from_counts(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean IoU over background and foreground classes."""
    del eps
    fg_iou = foreground_iou_from_counts(tp, fp, fn)
    bg_iou = _safe_div(tn, tn + fp + fn)
    return 0.5 * (bg_iou + fg_iou)


def sensitivity_from_counts(tp: torch.Tensor, fn: torch.Tensor,
                             eps: float | None = None) -> torch.Tensor:
    del eps
    return _safe_div(tp, tp + fn)


def specificity_from_counts(tn: torch.Tensor, fp: torch.Tensor,
                             eps: float | None = None) -> torch.Tensor:
    del eps
    return _safe_div(tn, tn + fp)


def accuracy_from_counts(tp: torch.Tensor, fp: torch.Tensor,
                          fn: torch.Tensor, tn: torch.Tensor,
                          eps: float | None = None) -> torch.Tensor:
    del eps
    return _safe_div(tp + tn, tp + fp + fn + tn)


def vmunet_metrics_from_counts(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tn: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return VM-UNet ISIC metric names from global confusion counts."""
    return {
        "miou": foreground_iou_from_counts(tp, fp, fn),
        "f1_or_dsc": dice_from_counts(tp, fp, fn),
        "accuracy": accuracy_from_counts(tp, fp, fn, tn),
        "specificity": specificity_from_counts(tn, fp),
        "sensitivity": sensitivity_from_counts(tp, fn),
    }


@torch.no_grad()
def binary_accuracy(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
                    eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > threshold).float()
    pred_f = pred.flatten(1)
    targ_f = target.float().flatten(1)
    correct = (pred_f == targ_f).float().sum(dim=1)
    return (correct / pred_f.size(1)).mean()


@torch.no_grad()
def binary_sensitivity(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
                       eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > threshold).float()
    pred_f = pred.flatten(1)
    targ_f = target.float().flatten(1)
    tp = (pred_f * targ_f).sum(dim=1)
    fn = ((1 - pred_f) * targ_f).sum(dim=1)
    return ((tp + eps) / (tp + fn + eps)).mean()


@torch.no_grad()
def binary_specificity(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
                       eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > threshold).float()
    pred_f = pred.flatten(1)
    targ_f = target.float().flatten(1)
    tn = ((1 - pred_f) * (1 - targ_f)).sum(dim=1)
    fp = (pred_f * (1 - targ_f)).sum(dim=1)
    return ((tn + eps) / (tn + fp + eps)).mean()


def first_scale(output) -> torch.Tensor:
    """Pull the highest-resolution prediction out of a deep-supervision output."""
    if isinstance(output, (list, tuple)):
        return output[0]
    return output
