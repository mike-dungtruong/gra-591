"""One-time BERT encoding of all captions → torch .pt cache.

Saves a dict { image_id: { 'pooled': Tensor[C], 'tokens': Tensor[L, C],
                           'attention_mask': Tensor[L] (long) } }
so training can skip BERT and just look up text features by image_id.

Usage:
    python scripts/precompute_text_features.py \
        --captions /path/to/captions.jsonl \
        --out cache/text_features.pt \
        --model bert-base-uncased \
        --pool mean \
        --max_length 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Make repo root importable when run directly from scripts/.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.captions import load_captions
from src.models.text_encoder import FrozenBertTextEncoder


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--pool", default="mean", choices=["attention", "mean", "cls"])
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading captions from {args.captions}")
    caps = load_captions(args.captions)
    print(f"  -> {len(caps)} captions")

    print(f"Loading text encoder: {args.model} (pool={args.pool})")
    enc = FrozenBertTextEncoder(model_name=args.model, pool=args.pool, freeze=True)
    enc.eval().to(args.device)

    image_ids = sorted(caps.keys())
    out: dict = {}

    for i in tqdm(range(0, len(image_ids), args.batch_size), desc="encoding"):
        batch_ids = image_ids[i : i + args.batch_size]
        texts = [caps[k] for k in batch_ids]
        tok = enc.tokenize(texts, max_length=args.max_length)
        input_ids = tok["input_ids"].to(args.device)
        attn_mask = tok["attention_mask"].to(args.device)
        pooled, tokens = enc(input_ids, attn_mask)
        pooled = pooled.cpu()
        tokens = tokens.cpu()
        attn_mask_cpu = attn_mask.cpu()
        for j, image_id in enumerate(batch_ids):
            out[image_id] = {
                "pooled": pooled[j].clone(),
                "tokens": tokens[j].clone(),
                "attention_mask": attn_mask_cpu[j].clone(),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"Wrote {args.out}  ({len(out)} entries)")


if __name__ == "__main__":
    main()
