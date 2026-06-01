"""
Clinical evaluation metrics: per-class sensitivity and specificity.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_per_class_metrics(
    preds:       np.ndarray,
    labels:      np.ndarray,
    classes_used: list,
) -> pd.DataFrame:
    """
    Compute TP, FP, FN, TN, sensitivity, and specificity for each class.

    Args:
        preds:        (N,) predicted class indices
        labels:       (N,) true class indices
        classes_used: ordered list of class names

    Returns:
        DataFrame with one row per class
    """
    cm = confusion_matrix(labels, preds)
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    return pd.DataFrame({
        'Class':       classes_used,
        'TP':          TP,
        'FP':          FP,
        'FN':          FN,
        'TN':          TN,
        'Sensitivity': np.round(TP / np.maximum(TP + FN, 1), 4),
        'Specificity': np.round(TN / np.maximum(TN + FP, 1), 4),
    })
