"""SwinUMamba — Mamba encoder + CNN decoder (original non-D variant).

Ported from papers/code/Swin-UMamba/swin_umamba/nnunetv2/nets/SwinUMamba.py.
Reuses VSSMEncoder from swin_umamba_d.py (identical encoder).

Architecture:
  stem (Conv7×7, stride=2) → VSSMEncoder (patch_size=2) → CNN encoder blocks →
  CNN decoder (UnetrUpBlock) → segmentation heads (deep supervision)

Deep supervision outputs are ordered high-res to low-res to match
DiceBCEWithDeepSupervision expectations.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from .swin_umamba_d import VSSMEncoder, load_pretrained_ckpt


class SwinUMamba(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1,
        feat_size: list = None,       # [48, 96, 192, 384, 768]
        drop_path_rate: float = 0.0,
        norm_name: str = "instance",
        res_block: bool = True,
        spatial_dims: int = 2,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if feat_size is None:
            feat_size = [48, 96, 192, 384, 768]
        assert len(feat_size) == 5, "feat_size must have 5 elements"

        self.deep_supervision = deep_supervision
        hidden_size = feat_size[4]

        # stride-2 stem: maps (B, in_chans, H, W) → (B, feat_size[0], H/2, W/2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, feat_size[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(feat_size[0], eps=1e-5, affine=True),
        )

        # Mamba encoder: takes stem output, returns 5 feature maps
        # vss_outs[0] = stem output (feat_size[0] ch, H/2, W/2)
        # vss_outs[1] = stage0 (feat_size[1] ch, H/4, W/4)
        # vss_outs[2] = stage1 (feat_size[2] ch, H/8, W/8)
        # vss_outs[3] = stage2 (feat_size[3] ch, H/16, W/16)
        # vss_outs[4] = stage3/bottleneck (feat_size[4] ch, H/32, W/32)
        self.vssm_encoder = VSSMEncoder(
            patch_size=2,
            in_chans=feat_size[0],
            depths=[2, 2, 9, 2],
            dims=feat_size[1],        # dims doubles: feat_size[1..4]
            drop_path_rate=drop_path_rate,
        )

        # CNN encoder blocks that refine each encoder feature map
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

        # CNN decoder: each UnetrUpBlock upsamples by 2× then merges skip
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

        # Segmentation heads for deep supervision (high-res to low-res)
        self.out_layers = nn.ModuleList([
            UnetOutBlock(spatial_dims=spatial_dims, in_channels=feat_size[i], out_channels=num_classes)
            for i in range(4)
        ])

    def forward(self, x_in: torch.Tensor):
        x1 = self.stem(x_in)
        vss_outs = self.vssm_encoder(x1)
        # vss_outs[0] = stem output (H/2), vss_outs[1..4] = encoder stages
        enc1 = self.encoder1(x_in)          # (B, feat[0], H,    W)
        enc2 = self.encoder2(vss_outs[0])   # (B, feat[1], H/2,  W/2)
        enc3 = self.encoder3(vss_outs[1])   # (B, feat[2], H/4,  W/4)
        enc4 = self.encoder4(vss_outs[2])   # (B, feat[3], H/8,  W/8)
        enc5 = self.encoder5(vss_outs[3])   # (B, feat[4], H/16, W/16)
        enc_hidden = vss_outs[4]            # (B, feat[4], H/32, W/32)

        dec4 = self.decoder6(enc_hidden, enc5)  # → H/16
        dec3 = self.decoder5(dec4, enc4)         # → H/8
        dec2 = self.decoder4(dec3, enc3)         # → H/4
        dec1 = self.decoder3(dec2, enc2)         # → H/2
        dec0 = self.decoder2(dec1, enc1)         # → H
        dec_out = self.decoder1(dec0)            # → H (feat[0] ch)

        # feat_out is high-res to low-res, matching DiceBCEWithDeepSupervision convention
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


def build_swin_umamba(
    num_input_channels: int = 3,
    num_classes: int = 1,
    feat_size: list = None,
    drop_path_rate: float = 0.0,
    deep_supervision: bool = True,
    pretrained_ckpt: str = None,
) -> SwinUMamba:
    """Factory for SwinUMamba (CNN decoder baseline)."""
    if feat_size is None:
        feat_size = [48, 96, 192, 384, 768]
    model = SwinUMamba(
        in_chans=num_input_channels,
        num_classes=num_classes,
        feat_size=feat_size,
        drop_path_rate=drop_path_rate,
        deep_supervision=deep_supervision,
    )
    if pretrained_ckpt is not None:
        model = load_pretrained_ckpt(model, num_input_channels=feat_size[0], ckpt_path=pretrained_ckpt)
    return model
