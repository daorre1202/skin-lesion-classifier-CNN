from __future__ import annotations
"""
PyTorch Dataset for dermoscopy images with graduated augmentation.
"""

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from skin_classifier.data.transforms import build_transforms, LEVEL_NAMES

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# Augmentation ratio thresholds (can be overridden via config)
AUG_THR_LIGHT  = 2.0
AUG_THR_MEDIUM = 6.0
AUG_THR_HEAVY  = 15.0


class SkinDataset(Dataset):
    """
    Dataset with per-class graduated augmentation.

    Transforms are stored as instance attributes rather than globals.
    When three models with different input resolutions (224px and 300px)
    are trained sequentially, using global transform variables would cause
    the second model to overwrite the first's transforms, silently producing
    incorrect evaluation metrics.

    Args:
        root_dir:        path to folder with one subdirectory per class
        classes:         list of class names in fixed order
        is_train:        if True, assigns augmentation levels per class
        transforms:      pre-built transform dict (from build_transforms);
                         if None, defaults to 224px transforms
        clinical_boost:  {class_name: min_level} — override the automatic
                         level for clinically important classes
    """

    def __init__(
        self,
        root_dir:       str,
        classes:        list,
        is_train:       bool = False,
        transforms:     dict = None,
        clinical_boost: Optional[dict] = None,
    ):
        self.classes  = classes
        self.cls2idx  = {c: i for i, c in enumerate(classes)}
        self.is_train = is_train
        self.samples  = []

        # Load transforms
        t = transforms if transforms is not None else build_transforms(224)
        self._aug_transforms = t['aug']
        self._val_transform  = t['val']
        self._tta_transform  = t['tta']

        # Collect image paths
        for cls in classes:
            cls_dir = Path(root_dir) / cls
            if cls_dir.exists():
                for p in sorted(cls_dir.iterdir()):
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                        self.samples.append((str(p), cls))

        # Assign augmentation levels based on class imbalance
        if is_train:
            counts    = Counter(cls for _, cls in self.samples)
            max_count = max(counts.values())
            self.class_aug_level = {}

            for cls, n in counts.items():
                ratio = max_count / n
                if ratio < AUG_THR_LIGHT:
                    level = 0
                elif ratio < AUG_THR_MEDIUM:
                    level = 1
                elif ratio < AUG_THR_HEAVY:
                    level = 2
                else:
                    level = 3
                self.class_aug_level[cls] = level

            # Apply clinical boost (e.g. force MEL to at least level 2)
            if clinical_boost:
                for cls, min_level in clinical_boost.items():
                    if cls in self.class_aug_level:
                        self.class_aug_level[cls] = max(
                            self.class_aug_level[cls], min_level
                        )

            print(f"\n  Graduated augmentation (majority class: {max_count} images):")
            for cls in sorted(counts, key=lambda c: counts[c], reverse=True):
                ratio = max_count / counts[cls]
                level = self.class_aug_level[cls]
                base  = (0 if ratio < AUG_THR_LIGHT else
                         1 if ratio < AUG_THR_MEDIUM else
                         2 if ratio < AUG_THR_HEAVY else 3)
                boost = " ★CLINICAL BOOST" if (
                    clinical_boost and cls in clinical_boost and level > base
                ) else ""
                print(
                    f"    {cls:8s}  {counts[cls]:5d} imgs  "
                    f"ratio={ratio:5.1f}  — Level {level} ({LEVEL_NAMES[level]}){boost}"
                )
        else:
            self.class_aug_level = {cls: 0 for cls in classes}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, cls = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))

        if getattr(self, '_tta_mode', False):
            img = self._tta_transform(image=img)['image']
        else:
            level = self.class_aug_level.get(cls, 0)
            img   = self._aug_transforms[level](image=img)['image']

        return img, self.cls2idx[cls]
