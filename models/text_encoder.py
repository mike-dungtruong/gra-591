"""Frozen BERT text encoder for caption features.

Default: bert-base-uncased, frozen, attention-pooled to a single (B, C_text) vector
plus per-token features (B, L, C_text). Both are returned so downstream modules can
choose between pooled gating and token-level attention.

For ISIC captions (~20 tokens), attention pooling outperforms CLS in practice, but
all three options are wired here: 'attention' | 'mean' | 'cls'.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class FrozenBertTextEncoder(nn.Module):
    """Wraps a HuggingFace BERT, freezes it, and exposes a clean (pooled, tokens) API.

    Tokenization is handled here too via the static `tokenize` method so the dataloader
    can produce (input_ids, attention_mask) tensors that this module's forward consumes.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pool: str = "attention",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        assert pool in {"attention", "mean", "cls"}, f"unknown pool: {pool}"
        self.model_name = model_name
        self.pool_kind = pool

        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
            self.bert.eval()

        # Attention pooling: learnable query attends over token features.
        if pool == "attention":
            self.attn_query = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
            nn.init.trunc_normal_(self.attn_query, std=0.02)
            self.attn_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    @torch.no_grad()
    def tokenize(self, texts, max_length: int = 64) -> dict:
        """Tokenize a list[str] of captions into batched tensors on CPU."""
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def _attention_pool(
        self, tokens: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        # tokens: (B, L, C); attn_mask: (B, L) in {0,1}
        B, L, C = tokens.shape
        q = self.attn_query.expand(B, 1, C)  # (B, 1, C)
        k = self.attn_proj(tokens)  # (B, L, C)
        scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / (C ** 0.5)  # (B, L)
        scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        return (tokens * weights).sum(dim=1)  # (B, C)

    def _mean_pool(
        self, tokens: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attn_mask.unsqueeze(-1).float()
        return (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pooled, tokens) where pooled is (B, C) and tokens is (B, L, C)."""
        # bert is frozen if freeze=True; still wrap in no_grad to be safe and save activations.
        with torch.set_grad_enabled(self.bert.training):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        tokens = outputs.last_hidden_state  # (B, L, C)

        if self.pool_kind == "cls":
            pooled = tokens[:, 0]
        elif self.pool_kind == "mean":
            pooled = self._mean_pool(tokens, attention_mask)
        else:  # attention
            pooled = self._attention_pool(tokens, attention_mask)
        return pooled, tokens
