from __future__ import annotations

"""
Stratified dataset splitting and ground-truth CSV loading.
"""

import os
import json
import random
from collections import defaultdict

import pandas as pd

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


def load_groundtruth_csv(csv_path: str) -> dict:
    """Parse a single ISIC ground-truth CSV into {image_id: class_label}."""
    df      = pd.read_csv(csv_path)
    key_col = 'image' if 'image' in df.columns else df.columns[0]
    cls_cols = [c for c in df.columns if c.upper() in CLASSES]
    result = {}
    for _, row in df.iterrows():
        img_id = str(row[key_col])
        for col in cls_cols:
            if row[col] == 1 or str(row[col]).strip() == "1":
                result[img_id] = col.upper()
                break
    return result


def merge_groundtruth_csvs(csv_dir: str) -> dict:
    """
    Merge the three ISIC 2018 Task 3 CSVs (train/val/test splits from the
    original challenge) into a single {image_id: class_label} dictionary.
    This allows us to define our own reproducible 60/20/20 split.
    """
    paths = {
        "train": os.path.join(csv_dir, "ISIC2018_Task3_Training_GroundTruth.csv"),
        "val":   os.path.join(csv_dir, "ISIC2018_Task3_Validation_GroundTruth.csv"),
        "test":  os.path.join(csv_dir, "ISIC2018_Task3_Test_GroundTruth.csv"),
    }
    for split_name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ground truth CSV not found ({split_name}): {path}")

    label_map, origin, conflicts = {}, {}, []
    for split_name, path in paths.items():
        for img_id, cls in load_groundtruth_csv(path).items():
            if img_id in label_map and label_map[img_id] != cls:
                conflicts.append((img_id, label_map[img_id], cls))
            else:
                label_map[img_id] = cls
                origin[img_id]    = split_name

    if conflicts:
        raise RuntimeError(f"Label conflicts found: {conflicts[:3]}")

    print(f"✓ Ground truth loaded: {len(label_map)} images")
    return label_map


def split_dataset(
    label_map:   dict,
    id2path:     dict,
    train_ratio: float = 0.6,
    val_ratio:   float = 0.2,
    seed:        int   = 42,
    use_percent: float = 1.0,
    save_path:   str   = None,
) -> tuple[dict, dict, dict]:
    """
    Deterministic stratified split by class.

    Reproducibility is guaranteed by:
    1. Sorting image IDs lexicographically before shuffling.
    2. Iterating classes in alphabetical order.
    These two steps ensure the same split is produced across sessions and
    environments regardless of filesystem ordering.

    Args:
        label_map:   {image_id: class_label}
        id2path:     {image_id: file_path}
        train_ratio: fraction of data for training
        val_ratio:   fraction of data for validation
        seed:        random seed
        use_percent: use only a fraction of data (for quick tests)
        save_path:   if given, saves the split assignment as JSON

    Returns:
        Tuple of (prep_train, prep_val, prep_test), each a dict {class: [paths]}
    """
    assert train_ratio + val_ratio < 1.0

    per_cls: dict[str, list] = defaultdict(list)
    for iid in id2path:
        if iid in label_map:
            per_cls[label_map[iid]].append(iid)

    random.seed(seed)
    train_ids, val_ids, test_ids = set(), set(), set()

    for cls in sorted(per_cls.keys()):
        ids = sorted(per_cls[cls])
        random.shuffle(ids)
        if use_percent < 1.0:
            ids = ids[:max(1, int(len(ids) * use_percent))]
        n  = len(ids)
        nt = int(n * train_ratio)
        nv = int(n * val_ratio)
        train_ids.update(ids[:nt])
        val_ids.update(ids[nt:nt + nv])
        test_ids.update(ids[nt + nv:])

    prep: dict[str, dict] = {s: defaultdict(list) for s in ('train', 'val', 'test')}
    for iid, path in id2path.items():
        cls = label_map.get(iid)
        if cls not in CLASSES:
            continue
        if iid in train_ids:
            prep['train'][cls].append(path)
        elif iid in val_ids:
            prep['val'][cls].append(path)
        elif iid in test_ids:
            prep['test'][cls].append(path)

    n_train = sum(len(v) for v in prep['train'].values())
    n_val   = sum(len(v) for v in prep['val'].values())
    n_test  = sum(len(v) for v in prep['test'].values())
    print(f"✓ Split — Train: {n_train} | Val: {n_val} | Test: {n_test}")

    if save_path:
        assignment = {iid: 'train' for iid in train_ids}
        assignment.update({iid: 'val'  for iid in val_ids})
        assignment.update({iid: 'test' for iid in test_ids})
        with open(save_path, 'w') as f:
            json.dump(assignment, f)
        print(f"✓ Split saved: {os.path.basename(save_path)}")

    return dict(prep['train']), dict(prep['val']), dict(prep['test'])


def load_split_from_file(
    split_path: str,
    id2path:    dict,
    label_map:  dict,
) -> tuple[dict, dict, dict]:
    """
    Reconstruct prep_train/val/test from a saved split_assignment.json.
    Ensures resumed runs use the exact same partition as the original run,
    preventing data leakage from split differences between sessions.
    """
    with open(split_path) as f:
        assignment = json.load(f)

    prep: dict[str, dict] = {s: defaultdict(list) for s in ('train', 'val', 'test')}
    missing = 0
    for iid, path in id2path.items():
        cls = label_map.get(iid)
        if cls not in CLASSES:
            continue
        sp = assignment.get(iid)
        if sp in prep:
            prep[sp][cls].append(path)
        else:
            missing += 1

    if missing:
        print(f"  ⚠ {missing} images not found in split_assignment.json — skipped")

    n_train = sum(len(v) for v in prep['train'].values())
    n_val   = sum(len(v) for v in prep['val'].values())
    n_test  = sum(len(v) for v in prep['test'].values())
    print(f"✓ Split loaded from file — Train: {n_train} | Val: {n_val} | Test: {n_test}")

    return dict(prep['train']), dict(prep['val']), dict(prep['test'])
