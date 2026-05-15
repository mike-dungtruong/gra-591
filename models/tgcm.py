"""Text-Gated Channel Module (TGCM).

Adapts ViTexNet's Text-Guided Dynamic Convolution (TGDC) to fit Swin-UMamba†'s
(B, H, W, C) feature format. Pooled BERT features produce K softmax weights that
gate K parallel depthwise 1D convolutions over the spatially-flattened token
sequence. Two iterative refinement passes per call, matching ViTexNet.

This is the only "novel" surgery — every other model component is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TGCM(nn.Module):
    """Text-gated dynamic depthwise convolution over image tokens.

    Args:
        c_img: image-feature channels at this decoder stage.
        c_text: text-feature channels (BERT hidden size, e.g. 768).
        k_filters: number of parallel depthwise convolutions (ViTexNet uses K=4).
        kernel_size: 1D kernel size for the depthwise convs over the flattened tokens.
        iterative: if True, run the gated convolution twice (ViTexNet's iterative refinement).
        beta_init: initial value of the residual scale that controls how strongly the
                   text-modulated features contribute on top of the raw image features.

    Inputs to forward:
        x_img:    (B, H, W, C) image features.
        t_pooled: (B, c_text) pooled text features.

    Returns:
        (B, H, W, C) — same shape, text-modulated.
    """

    def __init__(
        self,
        c_img: int,
        c_text: int,
        k_filters: int = 4,
        kernel_size: int = 3,
        iterative: bool = True,
        beta_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.k = k_filters
        self.iterative = iterative

        # Text -> K softmax weights. Hidden dim follows ViTexNet (matches image channels).
        self.text_to_weights = nn.Sequential(
            nn.Linear(c_text, c_img),
            nn.ReLU(inplace=True),
            nn.Linear(c_img, k_filters),
        )
        # K parallel depthwise 1D convolutions over the flattened spatial dim.
        # groups=c_img => depthwise (per-channel).
        pad = kernel_size // 2
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(c_img, c_img, kernel_size, padding=pad, groups=c_img, bias=False)
                for _ in range(k_filters)
            ]
        )
        self.norm1 = nn.LayerNorm(c_img)
        self.norm2 = nn.LayerNorm(c_img)
        # Learnable scales (gamma for the fused output, beta for the residual mix).
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def _apply_filters(self, x_bnc: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Run K depthwise convs over (B, N, C) features and weight-sum the outputs."""
        # Reshape to (B, C, N) for conv1d.
        B, N, C = x_bnc.shape
        x = x_bnc.transpose(1, 2).contiguous()  # (B, C, N)
        # Stack K conv outputs.
        outs = torch.stack([conv(x) for conv in self.convs], dim=1)  # (B, K, C, N)
        w = weights.view(B, self.k, 1, 1)  # (B, K, 1, 1)
        fused = (outs * w).sum(dim=1)  # (B, C, N)
        return fused.transpose(1, 2).contiguous()  # (B, N, C)

    def forward(self, x_img: torch.Tensor, t_pooled: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x_img.shape
        N = H * W

        # 1) Text-driven filter weights.
        weights = F.softmax(self.text_to_weights(t_pooled), dim=-1)  # (B, K)

        # 2) First pass: K weighted depthwise convolutions over flattened tokens.
        x_flat = x_img.view(B, N, C)
        fused = self._apply_filters(x_flat, weights)
        fused = self.gamma1 * self.norm1(fused)

        # 3) Optional iterative refinement: feed back through the same filters and weights.
        if self.iterative:
            fused = self._apply_filters(fused, weights)
            fused = self.gamma2 * self.norm2(fused)

        # 4) Residual: add text-modulated features back to image features with learnable scale.
        out = x_flat + self.beta * fused
        return out.view(B, H, W, C)
