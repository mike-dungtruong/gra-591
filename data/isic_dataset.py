"""ISIC 2017 Dataset that yields (image, mask, caption_input_ids, caption_attn_mask, image_id).

Two text modes:
- `text_mode="tokens"` (default): tokenizes captions on the fly with a provided tokenizer
   and emits input_ids + attention_mask. The model's text encoder runs each batch.
- `text_mode="features"`: loads precomputed text features from a .pt cache produced by
   `scripts/precompute_text_features.py`. Faster and recommended for Colab.

The dataset auto-discovers `ISIC_XXXXXXX.jpg` images in `<root>/<split>/images/` and
their masks at `<root>/<split>/masks/ISIC_XXXXXXX_segmentation.png`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .captions import load_captions
from .transforms import mask_to_binary


class ISICDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,                          # 'train' or 'val'
        captions: Dict[str, str],
        transform: Callable,
        *,
        text_mode: str = "tokens",
        tokenizer=None,                      # required if text_mode='tokens'
        text_max_length: int = 64,
        text_features: Optional[Dict[str, dict]] = None,  # required if text_mode='features'
        require_caption: bool = True,
    ) -> None:
        super().__init__()
        assert split in {"train", "val"}, split
        assert text_mode in {"tokens", "features"}, text_mode
        if text_mode == "tokens" and tokenizer is None:
            raise ValueError("tokenizer is required when text_mode='tokens'")
        if text_mode == "features" and text_features is None:
            raise ValueError("text_features dict is required when text_mode='features'")

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.captions = captions
        self.text_mode = text_mode
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.text_features = text_features

        img_dir = self.root / split / "images"
        mask_dir = self.root / split / "masks"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing image dir: {img_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing mask dir: {mask_dir}")

        # Discover image files and pair them with masks + captions.
        records = []
        for img_path in sorted(img_dir.glob("ISIC_*.jpg")):
            image_id = img_path.stem  # 'ISIC_0000001'
            mask_path = mask_dir / f"{image_id}_segmentation.png"
            if not mask_path.exists():
                continue
            if require_caption and image_id not in captions:
                continue
            if text_mode == "features" and image_id not in (text_features or {}):
                continue
            records.append((image_id, img_path, mask_path))
        if not records:
            raise RuntimeError(
                f"No usable (image, mask, caption) triples found in {img_dir}. "
                "Check require_caption setting and caption/feature coverage."
            )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, path: Path) -> np.ndarray:
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        return mask_to_binary(m)

    def __getitem__(self, idx: int) -> dict:
        image_id, img_path, mask_path = self.records[idx]
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        out = self.transform(image=image, mask=mask)
        image_t = out["image"]                          # FloatTensor (3, H, W)
        mask_t = out["mask"].unsqueeze(0).float()        # FloatTensor (1, H, W)

        caption = self.captions.get(image_id, "")
        item = {"image": image_t, "mask": mask_t, "image_id": image_id, "caption": caption}

        if self.text_mode == "tokens":
            enc = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.text_max_length,
                return_tensors="pt",
            )
            item["input_ids"] = enc["input_ids"].squeeze(0)
            item["attention_mask"] = enc["attention_mask"].squeeze(0)
        else:  # features
            feats = self.text_features[image_id]
            item["text_pooled"] = torch.as_tensor(feats["pooled"], dtype=torch.float)
            item["text_tokens"] = torch.as_tensor(feats["tokens"], dtype=torch.float)
            item["text_attention_mask"] = torch.as_tensor(
                feats["attention_mask"], dtype=torch.long
            )
        return item


def build_isic_dataset(
    *,
    root: str | Path,
    split: str,
    captions_jsonl: str | Path,
    transform: Callable,
    text_mode: str = "tokens",
    tokenizer=None,
    text_max_length: int = 64,
    text_features_cache: Optional[str | Path] = None,
    require_caption: bool = True,
) -> ISICDataset:
    """Convenience builder used by train.py."""
    captions = load_captions(captions_jsonl)
    text_features = None
    if text_mode == "features":
        if text_features_cache is None:
            raise ValueError("text_features_cache must be provided when text_mode='features'")
        text_features = torch.load(text_features_cache, map_location="cpu")
    return ISICDataset(
        root=root,
        split=split,
        captions=captions,
        transform=transform,
        text_mode=text_mode,
        tokenizer=tokenizer,
        text_max_length=text_max_length,
        text_features=text_features,
        require_caption=require_caption,
    )
