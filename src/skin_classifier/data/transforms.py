"""
Augmentation pipeline with 4 graduated levels.

Level assignment is based on the class imbalance ratio (max_count / class_count):
  level 0 — no augmentation         (ratio < 2)
  level 1 — flips + light rotation   (ratio 2–6)
  level 2 — + affine transforms      (ratio 6–15)
  level 3 — + noise and blur         (ratio > 15)

Validation and test sets always use level 0 for reproducible evaluation.
TTA uses only geometric transforms (no colour changes) because the clinical
threshold calibrator was fitted on natural colours — altering them would
shift the probability distributions and invalidate the calibration.
"""

import albumentations as A
try:
    from albumentations.pytorch import ToTensorV2
except ImportError:
    from albumentations import ToTensorV2  # albumentations >= 2.0

# ImageNet normalisation constants
_NORM = [
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
]

LEVEL_NAMES = {
    0: "no augmentation",
    1: "light",
    2: "medium",
    3: "heavy",
}


def build_transforms(img_size: int) -> dict:
    """
    Build all augmentation transforms for a given input resolution.

    Returns a dict with keys:
        'aug': {0: transform, 1: transform, 2: transform, 3: transform}
        'val': transform   (level 0, for validation and test)
        'tta': transform   (geometric only, for Test-Time Augmentation)
    """
    aug0 = A.Compose([A.Resize(img_size, img_size), *_NORM])

    aug1 = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.6),
        *_NORM,
    ])

    aug2 = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.7),
        A.Affine(
            translate_percent={"x": 0.05, "y": 0.05},
            scale=(0.9, 1.1),
            p=0.4,
        ),
        *_NORM,
    ])

    aug3 = A.Compose([
        A.Resize(img_size, img_size),
        A.Affine(
            translate_percent={"x": 0.1, "y": 0.1},
            scale=(0.9, 1.1),
            rotate=(-20, 20),
            p=0.5,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.7),
        A.GaussNoise(std_range=(0.02, 0.10), p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.ImageCompression(quality_range=(75, 100), p=0.3),
        *_NORM,
    ])

    tta = A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        *_NORM,
    ])

    return {
        'aug': {0: aug0, 1: aug1, 2: aug2, 3: aug3},
        'val': aug0,
        'tta': tta,
    }
