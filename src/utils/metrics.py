"""Dice and IoU metrics for binary segmentation, computed at the highest-resolution output."""
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
    pred = (torch.sigmoid(logits) > threshold).float()
    target = target.float()
    pred_f = pred.flatten(1)
    targ_f = target.flatten(1)
    intersection = (pred_f * targ_f).sum(dim=1)
    union = pred_f.sum(dim=1) + targ_f.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


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
