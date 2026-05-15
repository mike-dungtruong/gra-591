"""Image / mask augmentations for ISIC 2017.

Uses Albumentations because it handles synchronized image+mask transforms cleanly.
Train: light geometric + photometric augmentation suited for dermoscopy.
Val:   resize + ImageNet normalization only.
"""
from __future__ import annotations

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(image_size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                border_mode=0, p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def val_transform(image_size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def mask_to_binary(mask_uint8: np.ndarray) -> np.ndarray:
    """Convert a 0-255 PNG mask to a binary {0, 1} mask."""
    return (mask_uint8 > 127).astype(np.uint8)
