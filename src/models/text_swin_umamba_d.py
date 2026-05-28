"""TextSwinUMambaD: Swin-UMamba† encoder + TGCM-injecting decoder.

We reuse the author's `VSSMEncoder` untouched (so VMamba-Tiny ImageNet weights load
identically) and define a `TextUNetResDecoder` that mirrors the author's
`UNetResDecoder` but inserts a `TGCM` call between `concat_back_dim[s]` and
`stages[s]` at every decoder stage. The model's forward signature accepts pooled
text features as an extra input.

This file depends on `models/swin_umamba_d.py`, which contains the upstream
encoder / decoder classes (see that file's docstring for attribution).
"""
from __future__ import annotations

import math
from typing import List, Sequence, Union

import torch
import torch.nn as nn

# Author code (extracted from Swin-UMamba):
from .swin_umamba_d import (  # type: ignore[attr-defined]
    FinalPatchExpand_X4,
    PatchExpand,
    VSSLayer,
    VSSMEncoder,
    load_pretrained_ckpt,
)
from .tgcm import TGCM


class TextIdentity(nn.Module):
    """Decoder hook used when text gating is disabled for ablations."""

    def forward(self, x_img: torch.Tensor, text_pooled: torch.Tensor) -> torch.Tensor:
        return x_img


class TextStageFusion(nn.Module):
    """LViT-style encoder fusion for NHWC VMamba feature maps."""

    def __init__(
        self,
        dim: int,
        text_dim: int,
        method: str = "film",
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        if method not in {"add", "film"}:
            raise ValueError(f"Unsupported text fusion method: {method}")
        self.method = method
        out_dim = dim if method == "add" else 2 * dim
        self.proj = nn.Linear(text_dim, out_dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # Start from the pretrained image-only encoder behavior; the projection
        # learns text conditioning without perturbing the first forward pass.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, text_pooled: torch.Tensor) -> torch.Tensor:
        if self.method == "add":
            bias = self.proj(text_pooled).unsqueeze(1).unsqueeze(1)
            return x + self.alpha * bias

        gamma, beta = self.proj(text_pooled).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1).unsqueeze(1)
        beta = beta.unsqueeze(1).unsqueeze(1)
        return x * (1.0 + self.alpha * gamma) + self.alpha * beta


class TextVSSMEncoder(VSSMEncoder):
    """VSSMEncoder with optional text fusion before selected VSS stages."""

    def __init__(
        self,
        *args,
        text_dim: int = 768,
        fusion_enabled: bool = False,
        fusion_method: str = "film",
        fusion_stages: Sequence[int] = (0, 1, 2, 3),
        fusion_alpha_init: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fusion_enabled = fusion_enabled
        self.fusion_stages = set(int(s) for s in fusion_stages)
        self.text_fusions = nn.ModuleDict()
        if fusion_enabled:
            for stage_idx in sorted(self.fusion_stages):
                if stage_idx < 0 or stage_idx >= self.num_layers:
                    raise ValueError(
                        f"fusion stage {stage_idx} is outside encoder stages 0..{self.num_layers - 1}"
                    )
                self.text_fusions[str(stage_idx)] = TextStageFusion(
                    dim=self.dims[stage_idx],
                    text_dim=text_dim,
                    method=fusion_method,
                    alpha_init=fusion_alpha_init,
                )

    def forward(self, x: torch.Tensor, text_pooled: torch.Tensor | None = None) -> List[torch.Tensor]:
        x_ret = [x]

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            if self.fusion_enabled and s in self.fusion_stages:
                if text_pooled is None:
                    raise ValueError("text_pooled is required when encoder text fusion is enabled")
                x = self.text_fusions[str(s)](x, text_pooled)
            x = layer(x)
            x_ret.append(x.permute(0, 3, 1, 2))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        return x_ret


class TextUNetResDecoder(nn.Module):
    """Mirror of the author's UNetResDecoder, plus a TGCM call per stage."""

    def __init__(
        self,
        num_classes: int,
        deep_supervision: bool,
        features_per_stage: Sequence[int],
        drop_path_rate: float = 0.2,
        d_state: int = 16,
        text_dim: int = 768,
        tgcm_k: int = 4,
        tgcm_kernel: int = 3,
        tgcm_iterative: bool = True,
        tgcm_beta_init: float = 0.5,
        tgcm_enabled: bool = True,
    ) -> None:
        super().__init__()
        encoder_channels = list(features_per_stage)
        n_stages = len(encoder_channels)
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # Drop-path schedule matches author code: linspace(rate, 0, (n-1)*2), depths [2]*4.
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages - 1) * 2)]
        depths = [2, 2, 2, 2]

        stages: List[nn.Module] = []
        expand_layers: List[nn.Module] = []
        seg_layers: List[nn.Module] = []
        concat_back_dim: List[nn.Module] = []
        tgcms: List[nn.Module] = []
        last_skip_dim = encoder_channels[0]

        for s in range(1, n_stages):
            below = encoder_channels[-s]            # input from the stage below (or bottleneck)
            skip = encoder_channels[-(s + 1)]        # matching skip channels
            expand_layers.append(
                PatchExpand(input_resolution=None, dim=below, dim_scale=2, norm_layer=nn.LayerNorm)
            )
            stages.append(
                VSSLayer(
                    dim=skip,
                    depth=2,
                    attn_drop=0.0,
                    drop_path=dpr[sum(depths[: s - 1]) : sum(depths[:s])],
                    d_state=math.ceil(2 * skip / 6) if d_state is None else d_state,
                    norm_layer=nn.LayerNorm,
                    downsample=None,
                    use_checkpoint=False,
                )
            )
            seg_layers.append(nn.Conv2d(skip, num_classes, 1, 1, 0, bias=True))
            concat_back_dim.append(nn.Linear(2 * skip, skip))
            if tgcm_enabled:
                tgcms.append(
                    TGCM(
                        c_img=skip,
                        c_text=text_dim,
                        k_filters=tgcm_k,
                        kernel_size=tgcm_kernel,
                        iterative=tgcm_iterative,
                        beta_init=tgcm_beta_init,
                    )
                )
            else:
                tgcms.append(TextIdentity())
            last_skip_dim = skip

        # Final 4x patch expand to reach input resolution, then a 1x1 seg head.
        expand_layers.append(
            FinalPatchExpand_X4(
                input_resolution=None, dim=encoder_channels[0], dim_scale=4,
                norm_layer=nn.LayerNorm,
            )
        )
        stages.append(nn.Identity())
        seg_layers.append(nn.Conv2d(last_skip_dim, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)
        self.tgcms = nn.ModuleList(tgcms)

    def forward(self, skips: List[torch.Tensor], text_pooled: torch.Tensor):
        """skips: list of encoder feature maps (B, C, H, W), bottleneck last.

        Mirrors the author's forward exactly, with a TGCM call inserted after
        concat_back_dim at each stage that has a skip connection.
        """
        lres_input = skips[-1]
        seg_outputs: List[torch.Tensor] = []
        for s in range(len(self.stages)):
            x = self.expand_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                # Concat with corresponding encoder skip, then project back to C.
                x = torch.cat((x, skips[-(s + 2)].permute(0, 2, 3, 1)), -1)
                x = self.concat_back_dim[s](x)
                # >>> TGCM injection: text gates channel-wise features before VSS blocks.
                x = self.tgcms[s](x, text_pooled)
            x = self.stages[s](x).permute(0, 3, 1, 2)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # Highest resolution first to match the loss / metric conventions.
        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            return seg_outputs[0]
        return seg_outputs


class TextSwinUMambaD(nn.Module):
    """Swin-UMamba† encoder + TGCM-injecting decoder.

    Forward signature: model(image, text_pooled) where text_pooled is (B, text_dim).
    """

    def __init__(
        self,
        *,
        num_input_channels: int = 3,
        num_classes: int = 1,
        features_per_stage: Sequence[int] = (96, 192, 384, 768),
        d_state: int = 16,
        drop_path_rate: float = 0.2,
        deep_supervision: bool = True,
        text_dim: int = 768,
        tgcm_k: int = 4,
        tgcm_kernel: int = 3,
        tgcm_iterative: bool = True,
        tgcm_beta_init: float = 0.5,
        tgcm_enabled: bool = True,
        text_fusion_enabled: bool = False,
        text_fusion_method: str = "film",
        text_fusion_stages: Sequence[int] = (0, 1, 2, 3),
        text_fusion_alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.vssm_encoder = TextVSSMEncoder(
            in_chans=num_input_channels,
            patch_size=4,
            depths=[2, 2, 9, 2],
            dims=96,
            drop_path_rate=drop_path_rate,
            text_dim=text_dim,
            fusion_enabled=text_fusion_enabled,
            fusion_method=text_fusion_method,
            fusion_stages=text_fusion_stages,
            fusion_alpha_init=text_fusion_alpha_init,
        )
        self.decoder = TextUNetResDecoder(
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            features_per_stage=features_per_stage,
            drop_path_rate=drop_path_rate,
            d_state=d_state,
            text_dim=text_dim,
            tgcm_k=tgcm_k,
            tgcm_kernel=tgcm_kernel,
            tgcm_iterative=tgcm_iterative,
            tgcm_beta_init=tgcm_beta_init,
            tgcm_enabled=tgcm_enabled,
        )

    def forward(
        self, image: torch.Tensor, text_pooled: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.vssm_encoder(image, text_pooled)
        return self.decoder(skips, text_pooled)

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        """Freeze pretrained encoder weights while leaving new adapters trainable."""
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name and "text_fusions" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


def build_text_swin_umamba_d(
    *,
    num_input_channels: int = 3,
    num_classes: int = 1,
    features_per_stage: Sequence[int] = (96, 192, 384, 768),
    d_state: int = 16,
    drop_path_rate: float = 0.2,
    deep_supervision: bool = True,
    text_dim: int = 768,
    tgcm_k: int = 4,
    tgcm_kernel: int = 3,
    tgcm_iterative: bool = True,
    tgcm_beta_init: float = 0.5,
    tgcm_enabled: bool = True,
    text_fusion_enabled: bool = False,
    text_fusion_method: str = "film",
    text_fusion_stages: Sequence[int] = (0, 1, 2, 3),
    text_fusion_alpha_init: float = 0.1,
    pretrained_ckpt: str | None = None,
) -> TextSwinUMambaD:
    """Instantiate TextSwinUMambaD and optionally load VMamba-Tiny pretrained weights."""
    model = TextSwinUMambaD(
        num_input_channels=num_input_channels,
        num_classes=num_classes,
        features_per_stage=features_per_stage,
        d_state=d_state,
        drop_path_rate=drop_path_rate,
        deep_supervision=deep_supervision,
        text_dim=text_dim,
        tgcm_k=tgcm_k,
        tgcm_kernel=tgcm_kernel,
        tgcm_iterative=tgcm_iterative,
        tgcm_beta_init=tgcm_beta_init,
        tgcm_enabled=tgcm_enabled,
        text_fusion_enabled=text_fusion_enabled,
        text_fusion_method=text_fusion_method,
        text_fusion_stages=text_fusion_stages,
        text_fusion_alpha_init=text_fusion_alpha_init,
    )
    if pretrained_ckpt:
        model = load_pretrained_ckpt(
            model, num_input_channels=num_input_channels, ckpt_path=pretrained_ckpt
        )
    return model
