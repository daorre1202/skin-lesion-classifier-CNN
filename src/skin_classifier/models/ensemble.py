from __future__ import annotations

"""
Weighted ensemble inference over multiple CNN models.

Each model's contribution is proportional to its validation BACC.
Validation BACC (not test BACC) is used for weighting to ensure the
test set remains completely unseen during all design decisions.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


def compute_ensemble_probs(
    models:      dict,
    val_baccs:   dict,
    data_loader: DataLoader,
    device:      torch.device,
) -> tuple[np.ndarray, list]:
    """
    Run weighted ensemble inference over a dataset.

    Args:
        models:      {model_name: nn.Module}
        val_baccs:   {model_name: float} — used to compute weights
        data_loader: DataLoader (shuffle=False, same for all models)
        device:      target device

    Returns:
        (sum_probs, labels)
        sum_probs: (N, C) array of weighted averaged probabilities
        labels:    list of true class indices
    """
    total_bacc   = sum(val_baccs.values())
    sum_probs    = None
    all_labels   = None

    for name, model in models.items():
        weight = val_baccs[name] / total_bacc
        model.eval()
        batch_probs, batch_labels = [], []

        with torch.no_grad():
            for x, y in data_loader:
                probs = torch.softmax(model(x.to(device)), dim=1)
                batch_probs.append(probs.cpu().numpy())
                batch_labels.extend(y.numpy())

        model_probs = np.vstack(batch_probs) * weight

        if sum_probs is None:
            sum_probs  = model_probs
            all_labels = batch_labels
        else:
            sum_probs += model_probs

    return sum_probs, all_labels


def get_ensemble_weights(val_baccs: dict) -> dict:
    """Return normalised weights proportional to validation BACC."""
    total = sum(val_baccs.values())
    return {name: bacc / total for name, bacc in val_baccs.items()}
