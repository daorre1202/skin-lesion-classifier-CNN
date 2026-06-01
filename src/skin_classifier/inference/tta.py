from __future__ import annotations

"""
Test-Time Augmentation (TTA).

Applies stochastic geometric augmentations N times at inference and
averages the resulting probability distributions. This reduces prediction
variance, particularly for minority classes where a single forward pass
can be sensitive to small image variations.

Only geometric transforms (flips, rotations) are used — no colour
changes — because the clinical threshold calibrator was fitted on
unmodified colour distributions. Altering colours at inference would
shift the probability outputs and invalidate the calibrated thresholds.
"""

import numpy as np
import torch
from typing import Optional
from torch.utils.data import DataLoader


def predict_with_tta(
    model:      torch.nn.Module,
    dataset,
    n_rounds:   int = 10,
    batch_size: int = 16,
    num_workers: int = 0,
    device:     Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run N rounds of TTA and return averaged probabilities.

    The dataset's _tta_mode flag is set before inference and restored
    in a finally block, ensuring the dataset is not left in TTA mode
    if an exception occurs.

    Args:
        model:       trained nn.Module in eval mode
        dataset:     SkinDataset instance (must have _tta_transform)
        n_rounds:    number of augmented forward passes
        batch_size:  inference batch size
        num_workers: DataLoader workers
        device:      if None, uses model's current device

    Returns:
        (avg_probs, labels)
        avg_probs: (N, C) array of mean softmax probabilities
        labels:    (N,) array of true class indices
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    sum_probs  = None
    all_labels = []
    dataset._tta_mode = True

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
    )

    try:
        for _ in range(n_rounds):
            round_probs, round_labels = [], []
            with torch.no_grad():
                for x, y in loader:
                    probs = torch.softmax(model(x.to(device)), dim=1)
                    round_probs.append(probs.cpu().numpy())
                    round_labels.extend(y.numpy())

            batch = np.vstack(round_probs)
            if sum_probs is None:
                sum_probs  = batch
                all_labels = round_labels
            else:
                sum_probs += batch
    finally:
        dataset._tta_mode = False

    return sum_probs / n_rounds, np.array(all_labels)
