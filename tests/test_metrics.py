"""
Tests for per-class sensitivity and specificity computation.

Verifies:
- Correct values on known inputs
- Perfect classifier gives sensitivity=1 and specificity=1 for all classes
- Output DataFrame has the expected structure
"""

import numpy as np
import pytest

from skin_classifier.utils.metrics import compute_per_class_metrics

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


class TestComputePerClassMetrics:

    def test_perfect_classifier(self):
        """A perfect classifier gives sensitivity=1 and specificity=1."""
        labels = np.array([0, 1, 2, 3, 4, 5, 6] * 10)
        preds  = labels.copy()
        df     = compute_per_class_metrics(preds, labels, CLASSES)

        assert (df['Sensitivity'] == 1.0).all()
        assert (df['Specificity'] == 1.0).all()

    def test_output_columns(self, synthetic_labels, synthetic_probs):
        """Output DataFrame has all required columns."""
        preds = synthetic_probs.argmax(axis=1)
        df    = compute_per_class_metrics(preds, synthetic_labels, CLASSES)

        expected_cols = {'Class', 'TP', 'FP', 'FN', 'TN', 'Sensitivity', 'Specificity'}
        assert set(df.columns) == expected_cols

    def test_output_row_count(self, synthetic_labels, synthetic_probs):
        """One row per class."""
        preds = synthetic_probs.argmax(axis=1)
        df    = compute_per_class_metrics(preds, synthetic_labels, CLASSES)
        assert len(df) == len(CLASSES)

    def test_sensitivity_in_range(self, synthetic_labels, synthetic_probs):
        """All sensitivity values are in [0, 1]."""
        preds = synthetic_probs.argmax(axis=1)
        df    = compute_per_class_metrics(preds, synthetic_labels, CLASSES)
        assert (df['Sensitivity'] >= 0).all() and (df['Sensitivity'] <= 1).all()

    def test_specificity_in_range(self, synthetic_labels, synthetic_probs):
        """All specificity values are in [0, 1]."""
        preds = synthetic_probs.argmax(axis=1)
        df    = compute_per_class_metrics(preds, synthetic_labels, CLASSES)
        assert (df['Specificity'] >= 0).all() and (df['Specificity'] <= 1).all()

    def test_known_values(self):
        """
        Binary-like scenario: 10 MEL positives, 10 negatives.
        Model predicts 8 TP and 2 FN for MEL, and no false positives.
        Expected: sensitivity = 8/10 = 0.8, specificity = 10/10 = 1.0
        """
        labels = np.array([0] * 10 + [1] * 10)  # 10 MEL, 10 NV
        preds  = np.array([0] * 8 + [1] * 2 + [1] * 10)  # 2 MEL missed

        # Use only the first two classes for this test
        df = compute_per_class_metrics(preds, labels, ['MEL', 'NV'])
        mel_row = df[df['Class'] == 'MEL'].iloc[0]

        assert mel_row['Sensitivity'] == pytest.approx(0.8, abs=1e-4)
        assert mel_row['Specificity'] == pytest.approx(1.0, abs=1e-4)

    def test_tp_fp_fn_tn_sum_to_total(self, synthetic_labels, synthetic_probs):
        """For each class, TP + FP + FN + TN equals total number of samples."""
        preds = synthetic_probs.argmax(axis=1)
        df    = compute_per_class_metrics(preds, synthetic_labels, CLASSES)
        n     = len(synthetic_labels)

        for _, row in df.iterrows():
            total = int(row['TP']) + int(row['FP']) + int(row['FN']) + int(row['TN'])
            assert total == n
