from __future__ import annotations

"""
Training and evaluation loops.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> tuple[float, float, float]:
    """
    One full training pass over the dataset.

    Gradient clipping (max_norm=1.0) prevents large gradient spikes
    when the model misclassifies minority-class samples early in training.

    Returns:
        (avg_loss, accuracy, balanced_accuracy)
    """
    model.train()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    bacc     = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, bacc


def evaluate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray, list]:
    """
    Evaluation pass (no gradient computation).

    Returns:
        (avg_loss, accuracy, balanced_accuracy, confusion_matrix, probs, labels)
        probs:  (N, C) softmax probability array
        labels: list of true class indices
    """
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            probs  = torch.softmax(logits, dim=1)

            total_loss += loss.item() * x.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y).sum().item()
            total      += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_probs_arr = np.vstack(all_probs)
    avg_loss  = total_loss / total
    accuracy  = correct / total
    bacc      = balanced_accuracy_score(all_labels, all_preds)
    cm        = confusion_matrix(all_labels, all_preds)
    return avg_loss, accuracy, bacc, cm, all_probs_arr, all_labels
