"""
Focal Loss with optional label smoothing.

Label smoothing is applied during training only. In validation, standard
CrossEntropyLoss is used so the loss value is directly interpretable and
the LR scheduler does not fire prematurely due to inflated loss values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) with label smoothing.

    Focuses training on hard examples by down-weighting easy ones with the
    modulating factor (1 - p_t)^gamma. With gamma=0 this reduces to standard
    cross-entropy. Label smoothing prevents overconfident predictions on
    minority classes with few training samples.

    Args:
        gamma:           focusing parameter (0 = standard CE)
        label_smoothing: fraction to redistribute from target to other classes
        reduction:       'mean' or 'sum'
    """

    def __init__(
        self,
        gamma:           float = 1.0,
        label_smoothing: float = 0.1,
        reduction:       str   = 'mean',
    ):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing
        self.reduction       = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.size(1)

        # Build smooth target distribution
        with torch.no_grad():
            smooth = torch.full_like(logits, self.label_smoothing / (C - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs     = F.log_softmax(logits, dim=1)
        ce_per_sample = -(smooth * log_probs).sum(dim=1)

        # Focal weight: down-weight easy examples
        with torch.no_grad():
            p_t = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
            fw  = (1.0 - p_t) ** self.gamma

        loss = fw * ce_per_sample
        return loss.mean() if self.reduction == 'mean' else loss.sum()
