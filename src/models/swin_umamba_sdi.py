"""SwinUMamba CNN decoder with U-Net v2 / VM-UNetV2-style SDI skip refinement."""
from __future__ import annotations

from typing import Sequence

import torch

from .sdi import MultiScaleSDIRefiner
from .swin_umamba import SwinUMamba
from .swin_umamba_d import load_pretrained_ckpt


class SwinUMambaSDI(SwinUMamba):
    """SwinUMamba with residual SDI over the CNN decoder skip features."""

    def __init__(
        self,
        *args,
        feat_size: Sequence[int] | None = None,
        sdi_channels: int | None = None,
        sdi_attention: bool = True,
        sdi_attention_ratio: int = 16,
        sdi_residual: bool = True,
        sdi_alpha_init: float = 0.1,
        **kwargs,
    ) -> None:
        if feat_size is None:
            feat_size = [48, 96, 192, 384, 768]
        super().__init__(*args, feat_size=list(feat_size), **kwargs)
        self.skip_refiner = MultiScaleSDIRefiner(
            channels_per_level=list(feat_size),
            sdi_channels=sdi_channels,
            attention=sdi_attention,
            attention_ratio=sdi_attention_ratio,
            residual=sdi_residual,
            alpha_init=sdi_alpha_init,
        )

    def forward(self, x_in: torch.Tensor):
        x1 = self.stem(x_in)
        vss_outs = self.vssm_encoder(x1)

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(vss_outs[0])
        enc3 = self.encoder3(vss_outs[1])
        enc4 = self.encoder4(vss_outs[2])
        enc5 = self.encoder5(vss_outs[3])
        enc_hidden = vss_outs[4]

        enc1, enc2, enc3, enc4, enc5 = self.skip_refiner([enc1, enc2, enc3, enc4, enc5])

        dec4 = self.decoder6(enc_hidden, enc5)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        dec_out = self.decoder1(dec0)

        feat_out = [dec_out, dec1, dec2, dec3]
        if self.deep_supervision:
            return [self.out_layers[i](feat_out[i]) for i in range(4)]
        return self.out_layers[0](feat_out[0])


def build_swin_umamba_sdi(
    num_input_channels: int = 3,
    num_classes: int = 1,
    feat_size: Sequence[int] | None = None,
    drop_path_rate: float = 0.0,
    deep_supervision: bool = True,
    pretrained_ckpt: str | None = None,
    sdi_channels: int | None = None,
    sdi_attention: bool = True,
    sdi_attention_ratio: int = 16,
    sdi_residual: bool = True,
    sdi_alpha_init: float = 0.1,
) -> SwinUMambaSDI:
    """Factory for the SDI skip-fusion SwinUMamba CNN-decoder experiment."""
    if feat_size is None:
        feat_size = [48, 96, 192, 384, 768]
    model = SwinUMambaSDI(
        in_chans=num_input_channels,
        num_classes=num_classes,
        feat_size=list(feat_size),
        drop_path_rate=drop_path_rate,
        deep_supervision=deep_supervision,
        sdi_channels=sdi_channels,
        sdi_attention=sdi_attention,
        sdi_attention_ratio=sdi_attention_ratio,
        sdi_residual=sdi_residual,
        sdi_alpha_init=sdi_alpha_init,
    )
    model.skip_refiner.zero_init_residual_projection()
    if pretrained_ckpt is not None:
        model = load_pretrained_ckpt(
            model,
            num_input_channels=list(feat_size)[0],
            ckpt_path=pretrained_ckpt,
        )
    return model
