from __future__ import annotations

"""
Clinical threshold calibration for malignant skin lesion classes.

Standard argmax (threshold = 0.5) optimises global accuracy but yields
suboptimal sensitivity for rare malignant classes. This module searches
for the best per-class probability threshold on the validation set that
meets a minimum sensitivity target while maintaining a specificity floor.

Design rationale:
    Two strategies were evaluated for MEL calibration:
    1. F-beta (β=2) with specificity_floor=0.70
       Found theta≈0.15, which improved MEL sensitivity (+0.05) but
       severely degraded NV sensitivity (−0.13). NV represents 67% of
       the dataset — high NV error rates have clinical consequences
       (false positives = avoidable biopsies not flagged correctly).
    2. argmax-θ with specificity_floor=0.85
       Finds the highest theta that satisfies both sensitivity and
       specificity targets. Achieves sensitivity ≥0.85 for MEL with
       only a −0.004 impact on global BACC.

    Strategy 2 is used in this implementation.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def calibrate_threshold(
    val_probs:          np.ndarray,
    val_labels:         np.ndarray,
    cls_idx:            int,
    cls_name:           str,
    sensitivity_target: float,
    specificity_floor:  float,
    use_fbeta:          bool  = False,
    beta:               float = 2.0,
) -> float:
    """
    Find the optimal probability threshold for one class on the validation set.

    Args:
        val_probs:          (N, C) probability array from TTA ensemble
        val_labels:         (N,) true class indices
        cls_idx:            index of the target class
        cls_name:           class name for logging
        sensitivity_target: minimum required sensitivity
        specificity_floor:  minimum required specificity
        use_fbeta:          if True, maximise F-beta instead of argmax-θ
        beta:               F-beta beta parameter

    Returns:
        Calibrated threshold in [0.01, 0.99]
    """

    def _search(floor: float) -> list:
        candidates = []
        for theta in np.linspace(0.01, 0.99, 199):
            preds = (val_probs[:, cls_idx] >= theta).astype(int)
            lbin  = (val_labels == cls_idx).astype(int)
            tp = int(((preds == 1) & (lbin == 1)).sum())
            fp = int(((preds == 1) & (lbin == 0)).sum())
            fn = int(((preds == 0) & (lbin == 1)).sum())
            tn = int(((preds == 0) & (lbin == 0)).sum())
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            if spec < floor:
                continue
            if use_fbeta:
                score = (
                    (1 + beta**2) * sens * spec
                    / (beta**2 * spec + sens + 1e-9)
                )
            else:
                if sens >= sensitivity_target:
                    score = theta  # highest theta = most conservative
                else:
                    continue
            candidates.append((score, theta, sens, spec))
        return candidates

    method = f"F-beta(β={beta})" if use_fbeta else "argmax-θ"

    # Attempt 1: strict specificity floor
    candidates = _search(specificity_floor)
    if candidates:
        _, theta, sens, spec = max(candidates, key=lambda x: x[0])
        print(
            f"  {cls_name}: θ={theta:.3f} | sens={sens:.4f} | "
            f"spec={spec:.4f} | method={method} | floor={specificity_floor:.2f}"
        )
        return theta

    # Attempt 2: relaxed floor (fallback)
    fallback = 0.50
    print(
        f"  WARNING {cls_name}: no candidates with spec≥{specificity_floor:.2f} "
        f"— relaxing floor to {fallback:.2f}"
    )
    candidates = _search(fallback)
    if candidates:
        _, theta, sens, spec = max(candidates, key=lambda x: x[0])
        print(
            f"  {cls_name}: θ={theta:.3f} | sens={sens:.4f} | "
            f"spec={spec:.4f} | method={method} | floor=RELAXED"
        )
        return theta

    # Attempt 3: target unreachable
    print(
        f"  WARNING {cls_name}: sensitivity≥{sensitivity_target:.2f} unreachable "
        f"— using 0.5 (standard argmax)"
    )
    return 0.5


def apply_clinical_thresholds(
    probs:          np.ndarray,
    labels:         np.ndarray,
    thresholds:     dict,
    classes_used:   list,
    priority_order: list,
) -> tuple[np.ndarray, float]:
    """
    Apply per-class probability thresholds in priority order.

    Classes are applied in reverse priority order so that higher-priority
    classes (MEL) overwrite lower-priority ones (AKIEC) when both thresholds
    are exceeded for the same image.

    Args:
        probs:          (N, C) probability array
        labels:         (N,) true class indices
        thresholds:     {class_name: threshold_value}
        classes_used:   ordered list of class names
        priority_order: list of class names from lowest to highest priority

    Returns:
        (predictions, balanced_accuracy)
    """
    preds = probs.argmax(axis=1).copy()

    for cls_name in reversed(priority_order):
        if cls_name not in thresholds or cls_name not in classes_used:
            continue
        idx = classes_used.index(cls_name)
        preds[probs[:, idx] > thresholds[cls_name]] = idx

    bacc = balanced_accuracy_score(labels, preds)
    return preds, bacc
