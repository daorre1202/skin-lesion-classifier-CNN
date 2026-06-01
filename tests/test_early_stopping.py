"""
Tests for the mixed-score early stopping mechanism.

Verifies:
- Counter increments correctly when score does not improve
- Counter resets when score improves
- Training stops after exactly `patience` non-improving epochs
- Best epoch is tracked correctly
- Best BACC and loss are updated only on genuine improvement
"""

import pytest
import torch
import torch.nn as nn

from skin_classifier.training.early_stopping import EarlyStopping


@pytest.fixture
def tiny_model():
    """Minimal linear model for checkpoint saving tests."""
    return nn.Linear(4, 2)


class TestEarlyStopping:

    def test_stops_after_patience(self, tiny_model, tmp_path):
        """Training stops after exactly `patience` non-improving epochs."""
        es      = EarlyStopping(patience=3, min_delta=0.001)
        stopped = False

        for epoch in range(1, 20):
            # Score never improves after epoch 1
            val_bacc = 0.7 if epoch == 1 else 0.6
            if es.step(val_bacc, 0.5, tiny_model, epoch):
                stopped = True
                assert epoch == 4   # 1 good + 3 non-improving = stop at epoch 4
                break

        assert stopped, "Early stopping never triggered"

    def test_counter_resets_on_improvement(self, tiny_model):
        """Counter goes back to 0 when a genuine improvement is found."""
        es = EarlyStopping(patience=5, min_delta=0.001)

        es.step(0.70, 0.5, tiny_model, epoch=1)
        es.step(0.65, 0.5, tiny_model, epoch=2)
        es.step(0.65, 0.5, tiny_model, epoch=3)
        assert es.counter == 2

        # Genuine improvement
        es.step(0.80, 0.4, tiny_model, epoch=4)
        assert es.counter == 0

    def test_best_epoch_tracked(self, tiny_model):
        """best_epoch reflects the last epoch with a genuine improvement."""
        es = EarlyStopping(patience=10, min_delta=0.001)

        es.step(0.70, 0.5, tiny_model, epoch=1)
        es.step(0.75, 0.4, tiny_model, epoch=2)   # improvement
        es.step(0.74, 0.4, tiny_model, epoch=3)   # no improvement
        es.step(0.80, 0.3, tiny_model, epoch=4)   # improvement

        assert es.best_epoch == 4

    def test_best_bacc_updated(self, tiny_model):
        """best_bacc reflects the highest BACC seen."""
        es = EarlyStopping(patience=5, min_delta=0.001)

        es.step(0.70, 0.5, tiny_model, epoch=1)
        es.step(0.82, 0.4, tiny_model, epoch=2)
        es.step(0.75, 0.4, tiny_model, epoch=3)

        assert es.best_bacc == pytest.approx(0.82, abs=1e-4)

    def test_checkpoint_saved(self, tiny_model, tmp_path):
        """Checkpoint file is created when score improves."""
        ckpt = str(tmp_path / "best.pth")
        es   = EarlyStopping(patience=5, checkpoint_path=ckpt)

        es.step(0.75, 0.5, tiny_model, epoch=1)
        assert not (tmp_path / "best.pth").exists() is False  # file created

        # Overwrite with better checkpoint
        es.step(0.85, 0.4, tiny_model, epoch=2)
        loaded = torch.load(ckpt, weights_only=False)
        assert loaded is not None

    def test_min_delta_respected(self, tiny_model):
        """An improvement smaller than min_delta does not reset the counter."""
        es = EarlyStopping(patience=3, min_delta=0.01)

        es.step(0.700, 0.5, tiny_model, epoch=1)
        es.step(0.705, 0.5, tiny_model, epoch=2)  # +0.005 < min_delta → no improvement
        assert es.counter == 1

    def test_score_formula(self):
        """Verify the mixed-score formula: w*BACC + (1-w)/(1+loss)."""
        es = EarlyStopping(bacc_weight=0.85)
        expected = 0.85 * 0.8 + 0.15 / (1 + 0.4)
        assert es.score(0.8, 0.4) == pytest.approx(expected, abs=1e-6)
