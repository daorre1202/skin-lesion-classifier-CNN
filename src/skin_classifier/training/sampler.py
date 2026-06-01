"""
WeightedRandomSampler for class-imbalanced datasets.
"""

from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Build a sampler that assigns higher sampling probability to images
    from under-represented classes, compensating for the heavy class
    imbalance in HAM10000 (NV accounts for ~67% of all samples).

    Args:
        dataset: SkinDataset instance with a .samples attribute

    Returns:
        WeightedRandomSampler with per-sample weights
    """
    counts  = Counter(cls for _, cls in dataset.samples)
    weights = {cls: 1.0 / n for cls, n in counts.items()}
    sample_weights = torch.tensor(
        [weights[cls] for _, cls in dataset.samples],
        dtype=torch.float,
    )
    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )
