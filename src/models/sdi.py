"""Semantics and Detail Infusion modules for multi-scale skip refinement.

This adapts U-Net v2 / VM-UNetV2's SDI skip-fusion idea to existing
Swin-UMamba feature pyramids while preserving the original decoder channels.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """CBAM-style channel attention used by U-Net v2 before SDI."""

    def __init__(self, in_channels: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, in_channels // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention used by U-Net v2 before SDI."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat((avg_out, max_out), dim=1)))


class AttentionProject(nn.Module):
    """Attention-refine a feature map and project it to the shared SDI width."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        attention: bool = True,
        attention_ratio: int = 16,
    ) -> None:
        super().__init__()
        self.attention = attention
        if attention:
            self.channel_attn = ChannelAttention(in_channels, ratio=attention_ratio)
            self.spatial_attn = SpatialAttention()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attention:
            x = self.channel_attn(x) * x
            x = self.spatial_attn(x) * x
        return self.project(x)


class SDI(nn.Module):
    """Resize all feature levels to an anchor resolution and multiply them."""

    def __init__(self, channels: int, num_levels: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
                for _ in range(num_levels)
            ]
        )

    def forward(self, xs: Sequence[torch.Tensor], anchor: torch.Tensor) -> torch.Tensor:
        target_size = anchor.shape[-2:]
        out = torch.ones_like(anchor, dtype=torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            for x, conv in zip(xs, self.convs):
                x = x.float()
                if x.shape[-2:] != target_size:
                    if x.shape[-2] >= target_size[0] and x.shape[-1] >= target_size[1]:
                        x = F.adaptive_avg_pool2d(x, target_size)
                    else:
                        x = F.interpolate(
                            x,
                            size=target_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                gate = 2.0 * torch.sigmoid(conv(x))
                out = out * gate
        return out.to(dtype=anchor.dtype)


class MultiScaleSDIRefiner(nn.Module):
    """Refine each encoder level with SDI while keeping original channels.

    Encoder levels are first attention-filtered and projected to a shared
    SDI width. Each level then acts as an anchor for SDI fusion. The result is
    projected back to the anchor's original channel count and added as a
    learnable residual so pretrained Swin-UMamba behavior is not disrupted.
    """

    def __init__(
        self,
        channels_per_level: Sequence[int],
        *,
        sdi_channels: int | None = None,
        attention: bool = True,
        attention_ratio: int = 16,
        residual: bool = True,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        if not channels_per_level:
            raise ValueError("channels_per_level must not be empty")
        self.channels_per_level = list(channels_per_level)
        self.sdi_channels = int(sdi_channels or self.channels_per_level[0])
        self.residual = residual

        self.input_projects = nn.ModuleList(
            [
                AttentionProject(
                    channels,
                    self.sdi_channels,
                    attention=attention,
                    attention_ratio=attention_ratio,
                )
                for channels in self.channels_per_level
            ]
        )
        self.sdi_blocks = nn.ModuleList(
            [SDI(self.sdi_channels, len(self.channels_per_level)) for _ in self.channels_per_level]
        )
        self.output_projects = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.sdi_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                )
                for channels in self.channels_per_level
            ]
        )
        self.alpha = nn.Parameter(torch.full((len(self.channels_per_level),), float(alpha_init)))

    def zero_init_residual_projection(self) -> None:
        """Start SDI as an exact no-op when used in residual mode."""
        for project in self.output_projects:
            conv = project[0]
            bn = project[1]
            nn.init.zeros_(conv.weight)
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)
            nn.init.zeros_(bn.running_mean)
            nn.init.ones_(bn.running_var)

    def forward(self, xs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(xs) != len(self.channels_per_level):
            raise ValueError(
                f"expected {len(self.channels_per_level)} feature levels, got {len(xs)}"
            )
        projected = [project(x) for project, x in zip(self.input_projects, xs)]
        refined: list[torch.Tensor] = []
        for i, anchor in enumerate(projected):
            delta = self.output_projects[i](self.sdi_blocks[i](projected, anchor))
            if self.residual:
                refined.append(xs[i] + self.alpha[i].view(1, 1, 1, 1) * delta)
            else:
                refined.append(delta)
        return refined
