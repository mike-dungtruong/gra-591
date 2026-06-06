"""TextSwinUMambaD with U-Net v2 / VM-UNetV2-style SDI skip refinement."""
from __future__ import annotations

from typing import List, Sequence, Union

import torch
import torch.nn as nn

from .sdi import MultiScaleSDIRefiner
from .swin_umamba_d import load_pretrained_ckpt
from .text_swin_umamba_d import TextUNetResDecoder, TextVSSMEncoder


class TextUNetResDecoderWithSDI(TextUNetResDecoder):
    """Text decoder with residual SDI applied before TGCM and VSS blocks."""

    def __init__(
        self,
        *,
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
        sdi_channels: int | None = None,
        sdi_attention: bool = True,
        sdi_attention_ratio: int = 16,
        sdi_residual: bool = True,
        sdi_alpha_init: float = 0.1,
    ) -> None:
        super().__init__(
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
        self.skip_refiner = MultiScaleSDIRefiner(
            channels_per_level=list(features_per_stage),
            sdi_channels=sdi_channels,
            attention=sdi_attention,
            attention_ratio=sdi_attention_ratio,
            residual=sdi_residual,
            alpha_init=sdi_alpha_init,
        )

    def forward(self, skips: List[torch.Tensor], text_pooled: torch.Tensor):
        refined_encoder_skips = self.skip_refiner(skips[1:])
        return super().forward([skips[0], *refined_encoder_skips], text_pooled)


class TextSwinUMambaDSDI(nn.Module):
    """Swin-UMambaD text model plus residual SDI over encoder feature levels."""

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
        sdi_channels: int | None = None,
        sdi_attention: bool = True,
        sdi_attention_ratio: int = 16,
        sdi_residual: bool = True,
        sdi_alpha_init: float = 0.1,
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
        self.decoder = TextUNetResDecoderWithSDI(
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
            sdi_channels=sdi_channels,
            sdi_attention=sdi_attention,
            sdi_attention_ratio=sdi_attention_ratio,
            sdi_residual=sdi_residual,
            sdi_alpha_init=sdi_alpha_init,
        )

    def forward(
        self,
        image: torch.Tensor,
        text_pooled: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.vssm_encoder(image, text_pooled)
        return self.decoder(skips, text_pooled)

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name and "text_fusions" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


def build_text_swin_umamba_d_sdi(
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
    sdi_channels: int | None = None,
    sdi_attention: bool = True,
    sdi_attention_ratio: int = 16,
    sdi_residual: bool = True,
    sdi_alpha_init: float = 0.1,
) -> TextSwinUMambaDSDI:
    """Factory for the SDI skip-fusion TextSwinUMambaD experiment."""
    model = TextSwinUMambaDSDI(
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
        sdi_channels=sdi_channels,
        sdi_attention=sdi_attention,
        sdi_attention_ratio=sdi_attention_ratio,
        sdi_residual=sdi_residual,
        sdi_alpha_init=sdi_alpha_init,
    )
    model.decoder.skip_refiner.zero_init_residual_projection()
    if pretrained_ckpt:
        model = load_pretrained_ckpt(
            model,
            num_input_channels=num_input_channels,
            ckpt_path=pretrained_ckpt,
        )
    return model
