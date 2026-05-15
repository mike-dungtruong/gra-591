"""Loss functions for binary segmentation with optional deep supervision.

The decoder returns either a single logit map (deep_supervision=False) or a list of logit
maps ordered from highest resolution to lowest. We apply Dice + BCE at every scale,
weighted by `deep_supervision_weights`, with masks downsampled to match each scale.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary segmentation.

    Args:
        logits: (B, 1, H, W) raw logits.
        target: (B, 1, H, W) binary mask in {0, 1}.
    """
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1).float()
    intersection = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target.float())


class DiceBCEWithDeepSupervision(nn.Module):
    """Sum of Dice + BCE, applied at every deep-supervision scale.

    The model is expected to output either a single tensor or a list of tensors
    sorted from highest resolution (index 0) to lowest. Targets are downsampled
    to match each scale via nearest-neighbor.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        deep_supervision_weights: Sequence[float] = (1.0,),
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.ds_weights = list(deep_supervision_weights)

    def _single_scale(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
        return self.dice_weight * dice_loss(logits, target) + self.bce_weight * bce_loss(
            logits, target
        )

    def forward(self, output, target: torch.Tensor) -> torch.Tensor:
        # Normalize to a list-of-tensors interface.
        if isinstance(output, (list, tuple)):
            scales = list(output)
        else:
            scales = [output]
        if len(self.ds_weights) < len(scales):
            # If config underspecified, pad with 0.5^k decay.
            extra = [0.5 ** (i + 1) for i in range(len(scales) - len(self.ds_weights))]
            ws = self.ds_weights + extra
        else:
            ws = self.ds_weights[: len(scales)]
        total = sum(w * self._single_scale(s, target) for w, s in zip(ws, scales))
        return total / max(sum(ws), 1e-8)
