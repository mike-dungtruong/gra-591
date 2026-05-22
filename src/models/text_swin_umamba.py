"""TextSwinUMamba — Mamba encoder + CNN decoder with TGCM text injection.

Mirrors SwinUMamba (swin_umamba.py) exactly, adding one TGCM after each of
the four coarsest decoder stages. The CNN decoder uses NCHW tensors, so each
TGCM call is wrapped with permute() to satisfy TGCM's (B, H, W, C) contract.

TGCM injection points (coarse → fine):
  decoder6 output  (768ch, H/16)  → tgcms[0]
  decoder5 output  (384ch, H/8)   → tgcms[1]
  decoder4 output  (192ch, H/4)   → tgcms[2]
  decoder3 output  (96ch,  H/2)   → tgcms[3]

The final two stages (decoder2, decoder1) run without text gating — matching
TextSwinUMambaD's 4-stage convention.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from .swin_umamba_d import VSSMEncoder, load_pretrained_ckpt
from .tgcm import TGCM


class TextSwinUMamba(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1,
        feat_size: list = None,
        drop_path_rate: float = 0.0,
        norm_name: str = "instance",
        res_block: bool = True,
        spatial_dims: int = 2,
        deep_supervision: bool = True,
        text_dim: int = 768,
        tgcm_k: int = 4,
        tgcm_kernel: int = 3,
        tgcm_iterative: bool = True,
        tgcm_beta_init: float = 0.1,
        tgcm_enabled: bool = True,
    ) -> None:
        super().__init__()
        if feat_size is None:
            feat_size = [48, 96, 192, 384, 768]
        assert len(feat_size) == 5, "feat_size must have 5 elements"

        self.deep_supervision = deep_supervision
        self.tgcm_enabled = tgcm_enabled
        hidden_size = feat_size[4]

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, feat_size[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(feat_size[0], eps=1e-5, affine=True),
        )

        self.vssm_encoder = VSSMEncoder(
            patch_size=2,
            in_chans=feat_size[0],
            depths=[2, 2, 9, 2],
            dims=feat_size[1],
            drop_path_rate=drop_path_rate,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=in_chans,
            out_channels=feat_size[0], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=feat_size[0],
            out_channels=feat_size[1], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=feat_size[1],
            out_channels=feat_size[2], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=feat_size[2],
            out_channels=feat_size[3], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=feat_size[3],
            out_channels=feat_size[4], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )

        self.decoder6 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=hidden_size,
            out_channels=feat_size[4], kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=hidden_size,
            out_channels=feat_size[3], kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=feat_size[3],
            out_channels=feat_size[2], kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=feat_size[2],
            out_channels=feat_size[1], kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims, in_channels=feat_size[1],
            out_channels=feat_size[0], kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims, in_channels=feat_size[0],
            out_channels=feat_size[0], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )

        # One TGCM per injected decoder stage: [768, 384, 192, 96]
        tgcm_channels = [feat_size[4], feat_size[3], feat_size[2], feat_size[1]]
        self.tgcms = nn.ModuleList([
            TGCM(
                c_img=c, c_text=text_dim,
                k_filters=tgcm_k, kernel_size=tgcm_kernel,
                iterative=tgcm_iterative, beta_init=tgcm_beta_init,
            )
            for c in tgcm_channels
        ])

        self.out_layers = nn.ModuleList([
            UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[i], out_channels=num_classes)
            for i in range(4)
        ])

    def _apply_tgcm(self, x: torch.Tensor, tgcm: TGCM, text_pooled: torch.Tensor) -> torch.Tensor:
        """NCHW → NHWC → TGCM → NCHW."""
        return tgcm(x.permute(0, 2, 3, 1), text_pooled).permute(0, 3, 1, 2).contiguous()

    def forward(self, x_in: torch.Tensor, text_pooled: torch.Tensor):
        x1 = self.stem(x_in)
        vss_outs = self.vssm_encoder(x1)

        enc1 = self.encoder1(x_in)          # (B, feat[0], H,    W)
        enc2 = self.encoder2(vss_outs[0])   # (B, feat[1], H/2,  W/2)
        enc3 = self.encoder3(vss_outs[1])   # (B, feat[2], H/4,  W/4)
        enc4 = self.encoder4(vss_outs[2])   # (B, feat[3], H/8,  W/8)
        enc5 = self.encoder5(vss_outs[3])   # (B, feat[4], H/16, W/16)
        enc_hidden = vss_outs[4]            # (B, feat[4], H/32, W/32)

        dec4 = self.decoder6(enc_hidden, enc5)
        if self.tgcm_enabled:
            dec4 = self._apply_tgcm(dec4, self.tgcms[0], text_pooled)

        dec3 = self.decoder5(dec4, enc4)
        if self.tgcm_enabled:
            dec3 = self._apply_tgcm(dec3, self.tgcms[1], text_pooled)

        dec2 = self.decoder4(dec3, enc3)
        if self.tgcm_enabled:
            dec2 = self._apply_tgcm(dec2, self.tgcms[2], text_pooled)

        dec1 = self.decoder3(dec2, enc2)
        if self.tgcm_enabled:
            dec1 = self._apply_tgcm(dec1, self.tgcms[3], text_pooled)

        dec0 = self.decoder2(dec1, enc1)
        dec_out = self.decoder1(dec0)

        feat_out = [dec_out, dec1, dec2, dec3]
        if self.deep_supervision:
            return [self.out_layers[i](feat_out[i]) for i in range(4)]
        return self.out_layers[0](feat_out[0])

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


def build_text_swin_umamba(
    num_input_channels: int = 3,
    num_classes: int = 1,
    feat_size: list = None,
    drop_path_rate: float = 0.0,
    deep_supervision: bool = True,
    pretrained_ckpt: str = None,
    text_dim: int = 768,
    tgcm_k: int = 4,
    tgcm_kernel: int = 3,
    tgcm_iterative: bool = True,
    tgcm_beta_init: float = 0.1,
    tgcm_enabled: bool = True,
) -> TextSwinUMamba:
    if feat_size is None:
        feat_size = [48, 96, 192, 384, 768]
    model = TextSwinUMamba(
        in_chans=num_input_channels,
        num_classes=num_classes,
        feat_size=feat_size,
        drop_path_rate=drop_path_rate,
        deep_supervision=deep_supervision,
        text_dim=text_dim,
        tgcm_k=tgcm_k,
        tgcm_kernel=tgcm_kernel,
        tgcm_iterative=tgcm_iterative,
        tgcm_beta_init=tgcm_beta_init,
        tgcm_enabled=tgcm_enabled,
    )
    if pretrained_ckpt is not None:
        model = load_pretrained_ckpt(model, num_input_channels=feat_size[0], ckpt_path=pretrained_ckpt)
    return model
