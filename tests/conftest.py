import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import tempfile
import numpy as np
import pandas as pd
import pytest

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
N_CLASSES = len(CLASSES)

@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)

@pytest.fixture
def synthetic_probs(rng):
    return rng.dirichlet(np.ones(N_CLASSES), size=100).astype(np.float32)

@pytest.fixture
def synthetic_labels(rng):
    return rng.integers(0, N_CLASSES, size=100)

@pytest.fixture
def synthetic_val_probs(rng):
    return rng.dirichlet(np.ones(N_CLASSES), size=200).astype(np.float32)

@pytest.fixture
def synthetic_val_labels(rng):
    return rng.integers(0, N_CLASSES, size=200)

@pytest.fixture
def label_map():
    mapping = {}
    for cls in CLASSES:
        for i in range(50):
            mapping[f"ISIC_{cls}_{i:04d}"] = cls
    return mapping

@pytest.fixture
def id2path(label_map, tmp_path):
    mapping = {}
    for img_id in label_map:
        p = tmp_path / f"{img_id}.jpg"
        p.touch()
        mapping[img_id] = str(p)
    return mapping

@pytest.fixture
def temp_csv_dir(tmp_path):
    ids, labels = [], []
    for cls in CLASSES:
        for i in range(50):
            ids.append(f"ISIC_{cls}_{i:04d}")
            labels.append(cls)
    for split_name in ['Training', 'Validation', 'Test']:
        rows = {'image': ids}
        for cls in CLASSES:
            rows[cls] = [1 if lbl == cls else 0 for lbl in labels]
        pd.DataFrame(rows).to_csv(
            tmp_path / f"ISIC2018_Task3_{split_name}_GroundTruth.csv",
            index=False,
        )
    return str(tmp_path)
