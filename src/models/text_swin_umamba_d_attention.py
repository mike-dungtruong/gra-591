"""TextSwinUMambaD Attention Variant: Swin-UMamba† encoder + Cross-Attention decoder.

This variant replaces the Text-Gated Channel Module (TGCM) with a Cross-Attention
mechanism. The visual features act as Queries, and the text sequence tokens act as 
Keys and Values.

This allows fine-grained, word-to-pixel alignment guided by the clinical text.
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
    load_pretrained_ckpt,
)
from .text_swin_umamba_d import TextVSSMEncoder


class TextIdentityAttention(nn.Module):
    """Decoder hook used when text attention is disabled for ablations."""

    def forward(self, x_img: torch.Tensor, text_tokens: torch.Tensor, text_attention_mask: torch.Tensor) -> torch.Tensor:
        return x_img


class CrossAttentionBlock(nn.Module):
    def __init__(self, c_img: int, c_text: int = 768, num_heads: int = 4, beta_init: float = 0.5):
        super().__init__()
        self.c_img = c_img
        self.c_text = c_text
        
        self.mha = nn.MultiheadAttention(
            embed_dim=c_img,
            num_heads=num_heads,
            kdim=c_text,
            vdim=c_text,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(c_img)
        self.norm2 = nn.LayerNorm(c_img)
        
        # FFN to process the attended features
        self.ffn = nn.Sequential(
            nn.Linear(c_img, c_img * 2),
            nn.GELU(),
            nn.Linear(c_img * 2, c_img)
        )
        
        # Learnable scale for the residual connection, similar to TGCM
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x: torch.Tensor, text_tokens: torch.Tensor, text_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        x: Image features (B, H, W, C_img) - NHWC layout
        text_tokens: Text sequence (B, L_text, C_text)
        text_attention_mask: Mask for text (B, L_text) where 1 is valid, 0 is PAD.
        """
        B, H, W, C = x.shape
        x_flat = x.view(B, H * W, C)  # (B, L_img, C)
        
        # In PyTorch MultiheadAttention, key_padding_mask expects True for elements to ignore.
        padding_mask = ~text_attention_mask.bool()  # (B, L_text)
        
        residual = x_flat
        x_norm = self.norm1(x_flat)
        
        attn_out, _ = self.mha(
            query=x_norm,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=padding_mask,
            need_weights=False
        )
        
        x_flat = residual + self.beta * attn_out
        
        # Optional FFN refinement
        x_flat = x_flat + self.beta * self.ffn(self.norm2(x_flat))
        
        return x_flat.view(B, H, W, C)


class AttentionUNetResDecoder(nn.Module):
    """Mirror of the author's UNetResDecoder, but with a CrossAttentionBlock per stage."""

    def __init__(
        self,
        num_classes: int,
        deep_supervision: bool,
        features_per_stage: Sequence[int],
        drop_path_rate: float = 0.2,
        d_state: int = 16,
        text_dim: int = 768,
        attention_heads: int = 4,
        attention_beta_init: float = 0.5,
        attention_enabled: bool = True,
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
        attn_blocks: List[nn.Module] = []
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
            
            if attention_enabled:
                attn_blocks.append(
                    CrossAttentionBlock(
                        c_img=skip,
                        c_text=text_dim,
                        num_heads=attention_heads,
                        beta_init=attention_beta_init,
                    )
                )
            else:
                attn_blocks.append(TextIdentityAttention())
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
        self.attn_blocks = nn.ModuleList(attn_blocks)

    def forward(self, skips: List[torch.Tensor], text_tokens: torch.Tensor, text_attention_mask: torch.Tensor):
        """skips: list of encoder feature maps (B, C, H, W), bottleneck last."""
        lres_input = skips[-1]
        seg_outputs: List[torch.Tensor] = []
        for s in range(len(self.stages)):
            x = self.expand_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                # Concat with corresponding encoder skip, then project back to C.
                x = torch.cat((x, skips[-(s + 2)].permute(0, 2, 3, 1)), -1)
                x = self.concat_back_dim[s](x)
                # >>> Cross-Attention injection: text guides channel-wise features before VSS blocks.
                x = self.attn_blocks[s](x, text_tokens, text_attention_mask)
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


class TextSwinUMambaD_Attention(nn.Module):
    """Swin-UMamba† encoder + Cross-Attention-injecting decoder."""

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
        attention_heads: int = 4,
        attention_beta_init: float = 0.5,
        attention_enabled: bool = True,
        text_fusion_enabled: bool = False,
        text_fusion_method: str = "film",
        text_fusion_stages: Sequence[int] = (0, 1, 2, 3),
        text_fusion_alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        
        # Reusing the TextVSSMEncoder since it still takes text_pooled for optional FiLM/Add fusion.
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
        self.decoder = AttentionUNetResDecoder(
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            features_per_stage=features_per_stage,
            drop_path_rate=drop_path_rate,
            d_state=d_state,
            text_dim=text_dim,
            attention_heads=attention_heads,
            attention_beta_init=attention_beta_init,
            attention_enabled=attention_enabled,
        )

    def forward(
        self, 
        image: torch.Tensor, 
        text_pooled: torch.Tensor, 
        text_tokens: torch.Tensor, 
        text_attention_mask: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Encoder uses text_pooled if text_fusion_enabled=True
        skips = self.vssm_encoder(image, text_pooled)
        # Decoder uses text_tokens and mask for Cross-Attention
        return self.decoder(skips, text_tokens, text_attention_mask)

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


def build_text_swin_umamba_d_attention(
    *,
    num_input_channels: int = 3,
    num_classes: int = 1,
    features_per_stage: Sequence[int] = (96, 192, 384, 768),
    d_state: int = 16,
    drop_path_rate: float = 0.2,
    deep_supervision: bool = True,
    text_dim: int = 768,
    attention_heads: int = 4,
    attention_beta_init: float = 0.5,
    attention_enabled: bool = True,
    text_fusion_enabled: bool = False,
    text_fusion_method: str = "film",
    text_fusion_stages: Sequence[int] = (0, 1, 2, 3),
    text_fusion_alpha_init: float = 0.1,
    pretrained_ckpt: str | None = None,
) -> TextSwinUMambaD_Attention:
    """Instantiate TextSwinUMambaD_Attention and optionally load VMamba-Tiny pretrained weights."""
    model = TextSwinUMambaD_Attention(
        num_input_channels=num_input_channels,
        num_classes=num_classes,
        features_per_stage=features_per_stage,
        d_state=d_state,
        drop_path_rate=drop_path_rate,
        deep_supervision=deep_supervision,
        text_dim=text_dim,
        attention_heads=attention_heads,
        attention_beta_init=attention_beta_init,
        attention_enabled=attention_enabled,
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
