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
from src.data.captions import coverage_report, load_captions  # noqa: E402


def list_image_ids(root: Path, split: str, image_glob: str) -> list[str]:
    img_dir = root / split / "images"
    if not img_dir.is_dir():
        print(f"ERROR: image dir not found: {img_dir}", file=sys.stderr)
        sys.exit(1)
    return sorted(p.stem for p in img_dir.glob(image_glob))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isic_root", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--image_glob", default="ISIC_*.jpg",
                        help="Glob for image files (default: ISIC_*.jpg; ISIC 2018: *.png)")
    args = parser.parse_args()

    caps = load_captions(args.captions)
    print(f"captions loaded: {len(caps)}")
    for split in ("train", "val"):
        ids = list_image_ids(args.isic_root, split, args.image_glob)
        rep = coverage_report(ids, caps)
        print(f"\n[{split}] {rep['with_caption']}/{rep['total_images']} have captions "
              f"({rep['missing']} missing)")
        if rep["missing"]:
            print("  missing examples:", rep["missing_examples"])


if __name__ == "__main__":
    main()
