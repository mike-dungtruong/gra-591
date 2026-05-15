"""Quick sanity check: do all ISIC train/val images have captions?

Usage:
    python scripts/verify_caption_coverage.py \
        --isic_root /path/to/isic2017 \
        --captions /path/to/captions.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.captions import coverage_report, load_captions  # noqa: E402


def list_image_ids(root: Path, split: str) -> list[str]:
    img_dir = root / split / "images"
    if not img_dir.is_dir():
        return []
    return sorted(p.stem for p in img_dir.glob("ISIC_*.jpg"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isic_root", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    args = parser.parse_args()

    caps = load_captions(args.captions)
    print(f"captions loaded: {len(caps)}")
    for split in ("train", "val"):
        ids = list_image_ids(args.isic_root, split)
        rep = coverage_report(ids, caps)
        print(f"\n[{split}] {rep['with_caption']}/{rep['total_images']} have captions "
              f"({rep['missing']} missing)")
        if rep["missing"]:
            print("  missing examples:", rep["missing_examples"])


if __name__ == "__main__":
    main()
