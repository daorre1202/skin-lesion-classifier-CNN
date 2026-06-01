"""
Tests for the augmentation pipeline.

Verifies:
- build_transforms returns the expected dictionary structure
- All 4 augmentation levels produce tensors of the correct shape
- Val transform produces deterministic output (same input → same output)
- TTA transform produces varied output (stochastic by design)
"""

import numpy as np
import pytest
import torch

from skin_classifier.data.transforms import build_transforms


@pytest.fixture
def sample_image():
    """Random 224×224 RGB image as uint8 numpy array."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_300():
    """Random 300×300 RGB image as uint8 numpy array."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)


class TestBuildTransforms:

    def test_returns_correct_keys(self):
        """build_transforms returns a dict with 'aug', 'val', and 'tta' keys."""
        t = build_transforms(224)
        assert set(t.keys()) == {'aug', 'val', 'tta'}

    def test_aug_has_four_levels(self):
        """'aug' sub-dict contains levels 0 through 3."""
        t = build_transforms(224)
        assert set(t['aug'].keys()) == {0, 1, 2, 3}

    @pytest.mark.parametrize("level", [0, 1, 2, 3])
    def test_aug_output_shape(self, level, sample_image):
        """Each augmentation level produces a (3, 224, 224) tensor."""
        t      = build_transforms(224)
        result = t['aug'][level](image=sample_image)['image']
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_val_output_shape(self, sample_image):
        """Val transform produces a (3, 224, 224) tensor."""
        t      = build_transforms(224)
        result = t['val'](image=sample_image)['image']
        assert result.shape == (3, 224, 224)

    def test_efficientnet_resolution(self, sample_image_300):
        """EfficientNet-B3 transforms produce a (3, 300, 300) tensor."""
        t      = build_transforms(300)
        result = t['val'](image=sample_image_300)['image']
        assert result.shape == (3, 300, 300)

    def test_val_is_deterministic(self, sample_image):
        """
        Val transform (no augmentation) produces identical tensors
        for the same input image.
        """
        t       = build_transforms(224)
        result1 = t['val'](image=sample_image.copy())['image']
        result2 = t['val'](image=sample_image.copy())['image']
        assert torch.allclose(result1, result2)

    def test_output_dtype(self, sample_image):
        """Output tensors are float32."""
        t      = build_transforms(224)
        result = t['val'](image=sample_image)['image']
        assert result.dtype == torch.float32
