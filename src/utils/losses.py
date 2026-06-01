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


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss for segmentation logits."""
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def _as_scales(output) -> list[torch.Tensor]:
    if isinstance(output, (list, tuple)):
        return list(output)
    return [output]


def _weights_for(scales: Sequence[torch.Tensor], weights: Sequence[float]) -> list[float]:
    if len(weights) < len(scales):
        last = float(weights[-1]) if weights else 1.0
        extra = [last * (0.5 ** (i + 1)) for i in range(len(scales) - len(weights))]
        return list(weights) + extra
    return list(weights[: len(scales)])


def morphological_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Return a binary/soft morphological-gradient boundary map."""
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def soft_dice_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = probs.flatten(1)
    target = target.flatten(1).float()
    intersection = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


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
        scales = _as_scales(output)
        ws = _weights_for(scales, self.ds_weights)
        total = sum(w * self._single_scale(s, target) for w, s in zip(ws, scales))
        return total / max(sum(ws), 1e-8)


class BoundaryAwareSegLossWithDeepSupervision(nn.Module):
    """Dice + focal segmentation loss with boundary and edge supervision."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        boundary_weight: float = 0.5,
        edge_weight: float = 0.2,
        deep_supervision_weights: Sequence[float] = (1.0,),
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        boundary_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.edge_weight = edge_weight
        self.ds_weights = list(deep_supervision_weights)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.boundary_kernel_size = boundary_kernel_size

    def _resize_target(self, target: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if target.shape[-2:] == size:
            return target.float()
        return F.interpolate(target.float(), size=size, mode="nearest")

    def _seg_scale_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = self._resize_target(target, logits.shape[-2:])
        return self.dice_weight * dice_loss(logits, target) + self.focal_weight * focal_loss(
            logits,
            target,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )

    def _boundary_scale_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = self._resize_target(target, logits.shape[-2:])
        target_boundary = morphological_boundary(target, self.boundary_kernel_size)
        pred_boundary = morphological_boundary(torch.sigmoid(logits), self.boundary_kernel_size)
        return soft_dice_loss(pred_boundary, target_boundary)

    def _edge_scale_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = self._resize_target(target, logits.shape[-2:])
        target_boundary = morphological_boundary(target, self.boundary_kernel_size)
        return F.binary_cross_entropy_with_logits(logits, target_boundary)

    def _weighted_average(self, losses: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
        total = sum(w * loss for w, loss in zip(weights, losses))
        return total / max(sum(weights), 1e-8)

    def forward(self, output, target: torch.Tensor) -> torch.Tensor:
        if isinstance(output, dict):
            seg_output = output["seg"]
            edge_output = output.get("edge")
        elif isinstance(output, (list, tuple)) and len(output) == 2:
            seg_output, edge_output = output
        else:
            seg_output = output
            edge_output = None

        seg_scales = _as_scales(seg_output)
        seg_weights = _weights_for(seg_scales, self.ds_weights)
        seg_losses = [self._seg_scale_loss(s, target) for s in seg_scales]
        boundary_losses = [self._boundary_scale_loss(s, target) for s in seg_scales]

        total = self._weighted_average(seg_losses, seg_weights)
        total = total + self.boundary_weight * self._weighted_average(
            boundary_losses,
            seg_weights,
        )

        edge_scales = [] if edge_output is None else _as_scales(edge_output)
        if edge_scales and self.edge_weight > 0:
            edge_weights = _weights_for(edge_scales, self.ds_weights)
            edge_losses = [self._edge_scale_loss(s, target) for s in edge_scales]
            total = total + self.edge_weight * self._weighted_average(
                edge_losses,
                edge_weights,
            )
        return total
