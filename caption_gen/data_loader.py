"""
ISIC 2017 data loading + stratified sampling for caption generation.

Expected layout (already present in your GRA591/datasets-1/isic2017/):
    train/images/ISIC_xxxxxxx.jpg
    train/masks/ISIC_xxxxxxx_segmentation.png
    val/images/ISIC_xxxxxxx.jpg
    val/masks/ISIC_xxxxxxx_segmentation.png

Optional metadata for stratified sampling (download from challenge.isic-archive.com/data/#2017):
    metadata.csv with columns:
        image_id, diagnosis, age_approximate, sex, anatomic_site_general
    Not required — use --no-stratify for random sampling without metadata.
"""
from __future__ import annotations

import base64
import csv
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Dataset location. Override with the ISIC_ROOT environment variable, e.g.:
#   ISIC_ROOT=/path/to/isic2017 python caption_gen/generate.py ...
DATASET_ROOT = Path(
    os.environ.get(
        "ISIC_ROOT",
        "/Users/mike/Documents/GRA591/datasets/ISIC17_18/isic2017",
    )
)
METADATA_CSV = DATASET_ROOT / "metadata.csv"


@dataclass
class IsicSample:
    image_id: str
    image_path: Path
    mask_path: Path
    diagnosis: str  # "melanoma" | "nevus" | "seborrheic keratosis"
    age: int | None = None
    sex: str | None = None
    anatomic_site: str | None = None
    features: list[str] = field(default_factory=list)

    def image_data_url(self) -> str:
        """Encode image as base64 data URL for OpenAI vision API."""
        with open(self.image_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        ext = self.image_path.suffix.lstrip(".").lower()
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{b64}"


def load_metadata(csv_path: Path = METADATA_CSV) -> dict[str, dict]:
    """Load metadata.csv into {image_id: {diagnosis, age, sex, site}}.

    Returns empty dict if file is missing — caller should handle the fallback.
    """
    if not csv_path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row["image_id"].strip()
            out[iid] = {
                "diagnosis": (row.get("diagnosis") or "").strip().lower(),
                "age": _safe_int(row.get("age_approximate")),
                "sex": _normalize_sex(row.get("sex")),
                "anatomic_site": _normalize_site(row.get("anatomic_site_general")),
            }
    return out


def _safe_int(v) -> int | None:
    try:
        return int(float(v)) if v not in (None, "", "unknown") else None
    except (TypeError, ValueError):
        return None


def _normalize_sex(v) -> str | None:
    if not v:
        return None
    s = str(v).strip().lower()
    if s in {"m", "male"}:
        return "male"
    if s in {"f", "female"}:
        return "female"
    return None


def _normalize_site(v) -> str | None:
    if not v or str(v).strip().lower() in {"", "unknown"}:
        return None
    return str(v).strip().lower().replace("_", " ")


def discover_images(split: str = "train") -> list[Path]:
    """List all .jpg files under {split}/images/."""
    img_dir = DATASET_ROOT / split / "images"
    return sorted(img_dir.glob("*.jpg"))


def build_samples(split: str = "train",
                  metadata: dict[str, dict] | None = None) -> list[IsicSample]:
    """Build IsicSample objects, joining images with metadata if available."""
    if metadata is None:
        metadata = load_metadata()

    samples: list[IsicSample] = []
    for img_path in discover_images(split):
        iid = img_path.stem  # "ISIC_0000000"
        mask_path = DATASET_ROOT / split / "masks" / f"{iid}_segmentation.png"
        if not mask_path.exists():
            # Skip images without masks (not useful for seg pipeline anyway).
            continue

        meta = metadata.get(iid, {})
        diagnosis = meta.get("diagnosis") or "unknown"
        # Normalize: ISIC GT uses "seborrheic_keratosis" -> "seborrheic keratosis"
        diagnosis = diagnosis.replace("_", " ")

        samples.append(IsicSample(
            image_id=iid,
            image_path=img_path,
            mask_path=mask_path,
            diagnosis=diagnosis,
            age=meta.get("age"),
            sex=meta.get("sex"),
            anatomic_site=meta.get("anatomic_site"),
        ))
    return samples


def stratified_sample(samples: list[IsicSample],
                      n_per_class: dict[str, int],
                      seed: int = 42) -> list[IsicSample]:
    """Stratified sample by diagnosis class.

    Args:
        samples: full list from build_samples().
        n_per_class: e.g. {"melanoma": 15, "nevus": 15, "seborrheic keratosis": 20}.
        seed: random seed for reproducibility.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[IsicSample]] = {}
    for s in samples:
        by_class.setdefault(s.diagnosis, []).append(s)

    picked: list[IsicSample] = []
    for cls, n in n_per_class.items():
        pool = by_class.get(cls, [])
        if len(pool) < n:
            print(f"[warn] class '{cls}' has only {len(pool)} samples, "
                  f"requested {n}. Taking all.")
            picked.extend(pool)
        else:
            picked.extend(rng.sample(pool, n))
    rng.shuffle(picked)
    return picked


def class_distribution(samples: list[IsicSample]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for s in samples:
        dist[s.diagnosis] = dist.get(s.diagnosis, 0) + 1
    return dist
