"""Generate a dummy captions JSONL for smoke-testing the training pipeline.

Scans the ISIC dataset directory for all ISIC_*.jpg images and writes one
JSONL record per image with a fixed placeholder caption.  Run this once, then
feed the output to precompute_text_features.py before launching train.py.

Usage:
    python scripts/make_dummy_captions.py \
        --isic_root /path/to/isic2017 \
        --out cache/dummy_captions.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PLACEHOLDER = "A dermoscopy image of a skin lesion."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isic_root", type=Path, required=True,
                        help="Root dir containing train/ and val/ subdirectories")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSONL path")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    seen: set[str] = set()
    records: list[dict] = []

    for split in args.splits:
        img_dir = args.isic_root / split / "images"
        if not img_dir.exists():
            print(f"[warn] {img_dir} not found, skipping")
            continue
        ids = sorted(p.stem for p in img_dir.glob("ISIC_*.jpg"))
        new = [i for i in ids if i not in seen]
        seen.update(new)
        records.extend({"image_id": i, "caption": PLACEHOLDER} for i in new)
        print(f"  {split}: {len(new)} images")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
