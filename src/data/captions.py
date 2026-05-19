"""Caption JSONL loading + tokenization helpers.

The captions file is JSONL with one record per line; required fields are `image_id`
and `caption`. Extra fields (cost, latency, model name, etc.) are ignored.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable


def load_captions(jsonl_path: str | Path) -> Dict[str, str]:
    """Load JSONL captions into a {image_id: caption} dict.

    Skips records missing `image_id` or `caption`, and drops empty captions.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Captions file not found: {path}")
    out: Dict[str, str] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_id = rec.get("image_id")
            caption = rec.get("caption")
            if not image_id or not caption:
                continue
            out[image_id] = caption.strip()
    return out


def coverage_report(image_ids: Iterable[str], captions: Dict[str, str]) -> dict:
    """Return a small dict describing caption coverage over an image-id list."""
    ids = list(image_ids)
    present = [i for i in ids if i in captions]
    missing = [i for i in ids if i not in captions]
    return {
        "total_images": len(ids),
        "with_caption": len(present),
        "missing": len(missing),
        "missing_examples": missing[:10],
    }
