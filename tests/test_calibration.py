"""
Tests for clinical threshold calibration.

Verifies:
- Returned threshold is always in [0.01, 0.99]
- Applying thresholds increases sensitivity for the target class
- Priority ordering: MEL overrides AKIEC when both thresholds exceeded
- Fallback to 0.5 when the target is unreachable
"""

import numpy as np
import pytest

from skin_classifier.inference.calibration import (
    calibrate_threshold,
    apply_clinical_thresholds,
)

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


class TestCalibrateThreshold:

    def test_output_in_valid_range(self, synthetic_val_probs, synthetic_val_labels):
        """Returned threshold is always in [0.01, 0.99]."""
        theta = calibrate_threshold(
            val_probs          = synthetic_val_probs,
            val_labels         = synthetic_val_labels,
            cls_idx            = 0,
            cls_name           = 'MEL',
            sensitivity_target = 0.5,
            specificity_floor  = 0.5,
        )
        assert 0.01 <= theta <= 0.99

    def test_fallback_when_unreachable(self, rng):
        """
        When sensitivity target is impossible (all images wrongly predicted),
        the function falls back to 0.5.
        """
        # Probs that never assign high probability to MEL
        probs  = np.full((50, 7), 1 / 7, dtype=np.float32)
        labels = np.zeros(50, dtype=int)  # all MEL

        theta = calibrate_threshold(
            val_probs          = probs,
            val_labels         = labels,
            cls_idx            = 0,
            cls_name           = 'MEL',
            sensitivity_target = 0.99,
            specificity_floor  = 0.99,
        )
        assert theta == 0.5

    def test_lower_threshold_for_high_sensitivity_target(self, rng):
        """
        A higher sensitivity target requires a lower threshold (more aggressive).
        """
        probs  = rng.dirichlet(np.ones(7), size=200).astype(np.float32)
        labels = rng.integers(0, 7, size=200)

        theta_low  = calibrate_threshold(probs, labels, 0, 'MEL', 0.3, 0.3)
        theta_high = calibrate_threshold(probs, labels, 0, 'MEL', 0.7, 0.3)

        # Higher sensitivity target → lower (more permissive) threshold
        assert theta_high <= theta_low


class TestApplyClinicalThresholds:

    def test_output_shape(self, synthetic_probs, synthetic_labels):
        """Output predictions have the same shape as input labels."""
        thresholds = {'MEL': 0.3, 'AKIEC': 0.4}
        preds, _   = apply_clinical_thresholds(
            synthetic_probs, synthetic_labels,
            thresholds, CLASSES, ['MEL', 'AKIEC']
        )
        assert preds.shape == synthetic_labels.shape

    def test_thresholds_increase_target_class_predictions(self):
        """
        Setting a low threshold for MEL should increase the number of
        MEL predictions compared to standard argmax.
        """
        n = 200
        rng    = np.random.default_rng(0)
        probs  = rng.dirichlet(np.ones(7), size=n).astype(np.float32)
        labels = rng.integers(0, 7, size=n)

        argmax_preds = probs.argmax(axis=1)
        n_mel_argmax = (argmax_preds == 0).sum()

        # Force a very low threshold → many more MEL predictions
        thresh_preds, _ = apply_clinical_thresholds(
            probs, labels, {'MEL': 0.05}, CLASSES, ['MEL']
        )
        n_mel_thresh = (thresh_preds == 0).sum()
        assert n_mel_thresh >= n_mel_argmax

    def test_mel_overrides_akiec(self):
        """
        When an image exceeds both MEL and AKIEC thresholds,
        MEL (higher priority) should win.
        """
        # One image with high probability for both MEL (idx 0) and AKIEC (idx 3)
        probs       = np.zeros((1, 7), dtype=np.float32)
        probs[0, 0] = 0.5   # MEL
        probs[0, 3] = 0.5   # AKIEC
        labels      = np.array([0])

        thresholds = {'MEL': 0.4, 'AKIEC': 0.4}
        preds, _   = apply_clinical_thresholds(
            probs, labels, thresholds, CLASSES, ['MEL', 'AKIEC']
        )
        assert preds[0] == 0, "MEL should override AKIEC when both thresholds exceeded"

    def test_bacc_returned_is_float(self, synthetic_probs, synthetic_labels):
        """The function returns a (predictions, float) tuple."""
        _, bacc = apply_clinical_thresholds(
            synthetic_probs, synthetic_labels,
            {'MEL': 0.3}, CLASSES, ['MEL']
        )
        assert isinstance(bacc, float)
        assert 0.0 <= bacc <= 1.0
