"""
Tests for the dataset splitting logic.

Verifies:
- Correct 60/20/20 proportions per class
- Determinism: same seed + same data → same split every time
- No overlap between train, val and test sets
- Save/load round-trip produces identical splits
"""

import json
import os

import pytest

from skin_classifier.data.splits import (
    split_dataset,
    load_split_from_file,
    merge_groundtruth_csvs,
)

CLASSES    = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
N_PER_CLS  = 50
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2


class TestSplitDataset:

    def test_total_count(self, label_map, id2path):
        """All images end up in exactly one split."""
        train, val, test = split_dataset(
            label_map, id2path,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42,
        )
        n_train = sum(len(v) for v in train.values())
        n_val   = sum(len(v) for v in val.values())
        n_test  = sum(len(v) for v in test.values())
        assert n_train + n_val + n_test == len(label_map)

    def test_approximate_ratios(self, label_map, id2path):
        """Each split is within ±2 images of the target ratio per class."""
        train, val, test = split_dataset(
            label_map, id2path,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42,
        )
        for cls in CLASSES:
            n_tr = len(train.get(cls, []))
            n_va = len(val.get(cls, []))
            n_te = len(test.get(cls, []))
            total = n_tr + n_va + n_te
            assert abs(n_tr - round(total * TRAIN_RATIO)) <= 2
            assert abs(n_va - round(total * VAL_RATIO))   <= 2

    def test_no_overlap(self, label_map, id2path):
        """No image appears in more than one split."""
        train, val, test = split_dataset(
            label_map, id2path,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42,
        )
        def paths_set(d):
            return {p for paths in d.values() for p in paths}

        tr, va, te = paths_set(train), paths_set(val), paths_set(test)
        assert tr & va == set(), "Overlap between train and val"
        assert tr & te == set(), "Overlap between train and test"
        assert va & te == set(), "Overlap between val and test"

    def test_determinism(self, label_map, id2path):
        """Same seed produces identical splits regardless of how many times called."""
        train1, val1, test1 = split_dataset(
            label_map, id2path,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42,
        )
        train2, val2, test2 = split_dataset(
            label_map, id2path,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42,
        )
        for cls in CLASSES:
            assert sorted(train1.get(cls, [])) == sorted(train2.get(cls, []))
            assert sorted(val1.get(cls, []))   == sorted(val2.get(cls, []))

    def test_different_seeds_differ(self, label_map, id2path):
        """Different seeds produce different splits."""
        train42, _, _ = split_dataset(
            label_map, id2path,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42,
        )
        train7, _, _ = split_dataset(
            label_map, id2path,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=7,
        )
        all_equal = all(
            sorted(train42.get(cls, [])) == sorted(train7.get(cls, []))
            for cls in CLASSES
        )
        assert not all_equal, "Different seeds should produce different splits"

    def test_save_load_roundtrip(self, label_map, id2path, tmp_path):
        """Saving and loading a split produces the exact same partition."""
        save_path = str(tmp_path / "split_assignment.json")
        train1, val1, test1 = split_dataset(
            label_map, id2path,
            train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42,
            save_path=save_path,
        )
        assert os.path.exists(save_path)

        train2, val2, test2 = load_split_from_file(
            save_path, id2path, label_map
        )
        for cls in CLASSES:
            assert sorted(train1.get(cls, [])) == sorted(train2.get(cls, []))
            assert sorted(val1.get(cls, []))   == sorted(val2.get(cls, []))
            assert sorted(test1.get(cls, []))  == sorted(test2.get(cls, []))


class TestMergeGroundtruthCSVs:

    def test_loads_all_images(self, temp_csv_dir):
        """All 350 synthetic images are loaded."""
        label_map = merge_groundtruth_csvs(temp_csv_dir)
        assert len(label_map) == len(CLASSES) * 50

    def test_correct_labels(self, temp_csv_dir):
        """Labels match the class encoded in the image ID."""
        label_map = merge_groundtruth_csvs(temp_csv_dir)
        for img_id, cls in label_map.items():
            expected_cls = img_id.split('_')[1]
            assert cls == expected_cls, f"{img_id} expected {expected_cls}, got {cls}"

    def test_missing_csv_raises(self, tmp_path):
        """Missing CSV file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            merge_groundtruth_csvs(str(tmp_path))
