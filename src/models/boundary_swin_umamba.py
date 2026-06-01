"""Boundary-aware SwinUMamba baseline.

This variant keeps the SwinUMamba encoder path and replaces decoder skip merging
with boundary-guided fusion plus auxiliary edge logits.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock

from .swin_umamba_d import VSSMEncoder, load_pretrained_ckpt


class BoundaryGuidedFusionBlock(nn.Module):
    """Upsample decoder features and fuse them with an edge-attended skip."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        *,
        norm_name: str = "instance",
        res_block: bool = True,
        spatial_dims: int = 2,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.edge_head = nn.Conv2d(skip_channels, 1, kernel_size=3, padding=1)
        self.fuse = nn.Conv2d(2 * skip_channels, skip_channels, kernel_size=1)
        self.refine = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=skip_channels,
            out_channels=skip_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.up(x)
        edge_logits = self.edge_head(skip)
        skip_attended = skip * (1.0 + torch.sigmoid(edge_logits))
        x = self.fuse(torch.cat((x, skip_attended), dim=1))
        return self.refine(x), edge_logits


class BoundarySwinUMamba(nn.Module):
    """SwinUMamba with boundary-guided decoder skip fusion."""

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1,
        feat_size: list[int] | None = None,
        drop_path_rate: float = 0.0,
        norm_name: str = "instance",
        res_block: bool = True,
        spatial_dims: int = 2,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if feat_size is None:
            feat_size = [48, 96, 192, 384, 768]
        if len(feat_size) != 5:
            raise ValueError("feat_size must have 5 elements")

        self.deep_supervision = deep_supervision
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
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[1],
            out_channels=feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[2],
            out_channels=feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[3],
            out_channels=feat_size[4],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder6 = BoundaryGuidedFusionBlock(
            hidden_size,
            feat_size[4],
            norm_name=norm_name,
            res_block=res_block,
            spatial_dims=spatial_dims,
        )
        self.decoder5 = BoundaryGuidedFusionBlock(
            feat_size[4],
            feat_size[3],
            norm_name=norm_name,
            res_block=res_block,
            spatial_dims=spatial_dims,
        )
        self.decoder4 = BoundaryGuidedFusionBlock(
            feat_size[3],
            feat_size[2],
            norm_name=norm_name,
            res_block=res_block,
            spatial_dims=spatial_dims,
        )
        self.decoder3 = BoundaryGuidedFusionBlock(
            feat_size[2],
            feat_size[1],
            norm_name=norm_name,
            res_block=res_block,
            spatial_dims=spatial_dims,
        )
        self.decoder2 = BoundaryGuidedFusionBlock(
            feat_size[1],
            feat_size[0],
            norm_name=norm_name,
            res_block=res_block,
            spatial_dims=spatial_dims,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feat_size[0],
            out_channels=feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out_layers = nn.ModuleList(
            [
                UnetOutBlock(
                    spatial_dims=spatial_dims,
                    in_channels=feat_size[i],
                    out_channels=num_classes,
                )
                for i in range(4)
            ]
        )

    def forward(self, x_in: torch.Tensor) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        x1 = self.stem(x_in)
        vss_outs = self.vssm_encoder(x1)

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(vss_outs[0])
        enc3 = self.encoder3(vss_outs[1])
        enc4 = self.encoder4(vss_outs[2])
        enc5 = self.encoder5(vss_outs[3])
        enc_hidden = vss_outs[4]

        dec4, edge4 = self.decoder6(enc_hidden, enc5)
        dec3, edge3 = self.decoder5(dec4, enc4)
        dec2, edge2 = self.decoder4(dec3, enc3)
        dec1, edge1 = self.decoder3(dec2, enc2)
        dec0, edge0 = self.decoder2(dec1, enc1)
        dec_out = self.decoder1(dec0)

        feat_out = [dec_out, dec1, dec2, dec3]
        if self.deep_supervision:
            seg_out: list[torch.Tensor] | torch.Tensor = [
                self.out_layers[i](feat_out[i]) for i in range(4)
            ]
        else:
            seg_out = self.out_layers[0](feat_out[0])

        edge_out: List[torch.Tensor] = [edge0, edge1, edge2, edge3, edge4]
        return {"seg": seg_out, "edge": edge_out}

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True


def build_boundary_swin_umamba(
    num_input_channels: int = 3,
    num_classes: int = 1,
    feat_size: list[int] | None = None,
    drop_path_rate: float = 0.0,
    deep_supervision: bool = True,
    pretrained_ckpt: str | None = None,
) -> BoundarySwinUMamba:
    """Factory for the boundary-aware SwinUMamba baseline."""
    if feat_size is None:
        feat_size = [48, 96, 192, 384, 768]
    model = BoundarySwinUMamba(
        in_chans=num_input_channels,
        num_classes=num_classes,
        feat_size=feat_size,
        drop_path_rate=drop_path_rate,
        deep_supervision=deep_supervision,
    )
    if pretrained_ckpt is not None:
        model = load_pretrained_ckpt(
            model,
            num_input_channels=feat_size[0],
            ckpt_path=pretrained_ckpt,
        )
    return model
