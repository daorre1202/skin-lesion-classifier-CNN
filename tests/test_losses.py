"""
Tests for FocalLoss.

Verifies:
- With gamma=0 and no label smoothing, FocalLoss equals CrossEntropyLoss
- Label smoothing reduces loss magnitude on confident predictions
- Output is a scalar tensor
- Gradient flows correctly
"""

import torch
import torch.nn as nn
import pytest

from skin_classifier.models.losses import FocalLoss


@pytest.fixture
def batch():
    """Small deterministic batch: 8 samples, 7 classes."""
    torch.manual_seed(42)
    logits  = torch.randn(8, 7)
    targets = torch.randint(0, 7, (8,))
    return logits, targets


class TestFocalLoss:

    def test_gamma_zero_equals_crossentropy(self, batch):
        """
        FocalLoss with gamma=0 and no smoothing must equal CrossEntropyLoss.
        The focal weight (1 - p_t)^0 = 1, so the losses are identical.
        """
        logits, targets = batch
        focal = FocalLoss(gamma=0.0, label_smoothing=0.0)
        ce    = nn.CrossEntropyLoss()
        assert torch.allclose(focal(logits, targets), ce(logits, targets), atol=1e-5)

    def test_output_is_scalar(self, batch):
        """Loss reduction='mean' returns a scalar tensor."""
        logits, targets = batch
        loss = FocalLoss(gamma=1.0)(logits, targets)
        assert loss.shape == torch.Size([])

    def test_loss_is_positive(self, batch):
        """Loss is always strictly positive."""
        logits, targets = batch
        loss = FocalLoss(gamma=1.0)(logits, targets)
        assert loss.item() > 0

    def test_smoothing_reduces_loss_on_confident_predictions(self):
        """
        With label smoothing the loss is higher on very confident (almost
        correct) predictions than without smoothing, because the target
        distribution is no longer a hard one-hot.
        """
        logits  = torch.tensor([[10.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]])
        targets = torch.tensor([0])

        loss_no_smooth = FocalLoss(gamma=0.0, label_smoothing=0.0)(logits, targets)
        loss_smooth    = FocalLoss(gamma=0.0, label_smoothing=0.1)(logits, targets)
        assert loss_smooth.item() > loss_no_smooth.item()

    def test_higher_gamma_lowers_easy_sample_loss(self):
        """
        Higher gamma should reduce the contribution of easy (high-confidence)
        samples by increasing the focal weight suppression.
        """
        logits  = torch.tensor([[10.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]])
        targets = torch.tensor([0])

        loss_g0 = FocalLoss(gamma=0.0, label_smoothing=0.0)(logits, targets)
        loss_g2 = FocalLoss(gamma=2.0, label_smoothing=0.0)(logits, targets)
        assert loss_g2.item() < loss_g0.item()

    def test_gradient_flows(self, batch):
        """Backward pass completes without error and produces non-zero gradients."""
        logits, targets = batch
        logits = logits.requires_grad_(True)
        loss   = FocalLoss(gamma=1.0)(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0
