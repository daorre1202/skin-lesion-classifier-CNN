"""
Mixed-score Early Stopping.

Uses a combined score of validation BACC and validation loss:
    score = w * BACC + (1 - w) * 1 / (1 + loss)

The mixed score is more stable than monitoring BACC alone, as the loss
component smooths out epoch-to-epoch fluctuations common in imbalanced
multi-class datasets.
"""

import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Stop training when the mixed validation score stops improving.

    Args:
        patience:         epochs without improvement before stopping
        min_delta:        minimum improvement to count as progress
        checkpoint_path:  if given, saves the best model state dict here
        bacc_weight:      weight of BACC in the mixed score (0–1)
    """

    def __init__(
        self,
        patience:         int   = 10,
        min_delta:        float = 0.001,
        checkpoint_path:  str   = None,
        bacc_weight:      float = 0.85,
    ):
        self.patience         = patience
        self.min_delta        = min_delta
        self.checkpoint_path  = checkpoint_path
        self.bacc_weight      = bacc_weight

        self.best_score = -np.inf
        self.best_bacc  = 0.0
        self.best_loss  = np.inf
        self.best_epoch = 0
        self.counter    = 0

    def score(self, val_bacc: float, val_loss: float) -> float:
        """Compute the mixed score for a given epoch."""
        return (
            self.bacc_weight * val_bacc
            + (1 - self.bacc_weight) / (1 + val_loss)
        )

    def step(
        self,
        val_bacc: float,
        val_loss: float,
        model:    nn.Module,
        epoch:    int,
    ) -> bool:
        """
        Update state with the current epoch's metrics.

        Returns True if training should stop.
        """
        s = self.score(val_bacc, val_loss)
        if s > self.best_score + self.min_delta:
            self.best_score = s
            self.best_bacc  = val_bacc
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            if self.checkpoint_path:
                torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1

        return self.counter >= self.patience


class CheckpointMeta:
    """
    Lightweight substitute for EarlyStopping when loading from checkpoint.
    Exposes the same interface so downstream code works without changes.
    """

    def __init__(self, meta: dict, bacc_weight: float):
        self.best_epoch  = meta['best_epoch']
        self.best_bacc   = meta['val_bacc']
        self.best_loss   = meta.get('val_loss', 0.0)
        self.bacc_weight = bacc_weight

    def score(self, val_bacc: float, val_loss: float) -> float:
        return (
            self.bacc_weight * val_bacc
            + (1 - self.bacc_weight) / (1 + val_loss)
        )
