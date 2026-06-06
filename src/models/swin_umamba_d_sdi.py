"""SwinUMambaD with U-Net v2 / VM-UNetV2-style SDI skip refinement."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .sdi import MultiScaleSDIRefiner
from .swin_umamba_d import (
    InitWeights_He,
    UNetResDecoder,
    VSSMEncoder,
    init_last_bn_before_add_to_0,
    load_pretrained_ckpt,
)


class UNetResDecoderWithSDI(UNetResDecoder):
    """UNetResDecoder preceded by residual multi-scale SDI skip refinement."""

    def __init__(
        self,
        *,
        num_classes: int,
        deep_supervision: bool,
        features_per_stage: Sequence[int],
        drop_path_rate: float = 0.2,
        d_state: int = 16,
        sdi_channels: int | None = None,
        sdi_attention: bool = True,
        sdi_attention_ratio: int = 16,
        sdi_residual: bool = True,
        sdi_alpha_init: float = 0.1,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            features_per_stage=list(features_per_stage),
            drop_path_rate=drop_path_rate,
            d_state=d_state,
        )
        self.skip_refiner = MultiScaleSDIRefiner(
            channels_per_level=list(features_per_stage),
            sdi_channels=sdi_channels,
            attention=sdi_attention,
            attention_ratio=sdi_attention_ratio,
            residual=sdi_residual,
            alpha_init=sdi_alpha_init,
        )

    def forward(self, skips):
        refined_encoder_skips = self.skip_refiner(skips[1:])
        return super().forward([skips[0], *refined_encoder_skips])


class SwinUMambaDSDI(nn.Module):
    """Swin-UMambaD plus residual SDI over encoder feature levels."""

    def __init__(self, vss_args: dict, decoder_args: dict) -> None:
        super().__init__()
        self.vssm_encoder = VSSMEncoder(**vss_args)
        self.decoder = UNetResDecoderWithSDI(**decoder_args)

    def forward(self, x: torch.Tensor):
        skips = self.vssm_encoder(x)
        return self.decoder(skips)

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


def build_swin_umamba_d_sdi(
    num_input_channels: int = 3,
    num_classes: int = 1,
    features_per_stage: Sequence[int] = (96, 192, 384, 768),
    d_state: int = 16,
    drop_path_rate: float = 0.2,
    deep_supervision: bool = True,
    pretrained_ckpt: str | None = None,
    sdi_channels: int | None = None,
    sdi_attention: bool = True,
    sdi_attention_ratio: int = 16,
    sdi_residual: bool = True,
    sdi_alpha_init: float = 0.1,
) -> SwinUMambaDSDI:
    """Factory for the SDI skip-fusion SwinUMambaD experiment."""
    vss_args = dict(
        in_chans=num_input_channels,
        patch_size=4,
        depths=[2, 2, 9, 2],
        dims=96,
        drop_path_rate=drop_path_rate,
    )
    decoder_args = dict(
        num_classes=num_classes,
        deep_supervision=deep_supervision,
        features_per_stage=list(features_per_stage),
        drop_path_rate=drop_path_rate,
        d_state=d_state,
        sdi_channels=sdi_channels,
        sdi_attention=sdi_attention,
        sdi_attention_ratio=sdi_attention_ratio,
        sdi_residual=sdi_residual,
        sdi_alpha_init=sdi_alpha_init,
    )
    model = SwinUMambaDSDI(vss_args, decoder_args)
    model.apply(InitWeights_He(1e-2))
    model.apply(init_last_bn_before_add_to_0)
    model.decoder.skip_refiner.zero_init_residual_projection()
    if pretrained_ckpt is not None:
        model = load_pretrained_ckpt(
            model,
            num_input_channels=num_input_channels,
            ckpt_path=pretrained_ckpt,
        )
    return model
