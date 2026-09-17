"""Sensitivity of the pipeline to the melanoma sensitivity target.

For each candidate target, the MEL threshold is recalibrated on validation with the same
argmax-theta rule used in the pipeline (largest theta meeting sensitivity >= target and
specificity >= 0.85); the AKIEC threshold stays at its calibrated value. The resulting
operating point is then scored on validation and, for reference only, on test.

The target used throughout the pipeline (0.85) is fixed by the clinical requirement, not
selected from this sweep.

Inputs (results/seed_42/): tta_val_sum.csv, tta_val_labels.csv, tta_sum_probs.csv, tta_labels.csv
Output: results/seed_42/threshold_sweep.json

Usage: python scripts/threshold_sweep.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
MALIGNANT = ['MEL', 'BCC', 'AKIEC']
THRESHOLD_PRIORITY = ['MEL', 'AKIEC']
MEL_TARGETS = [0.75, 0.80, 0.85, 0.90, 0.95]
MEL_SPECIFICITY_FLOOR = 0.85
AKIEC_TARGET, AKIEC_SPECIFICITY_FLOOR = 0.75, 0.70

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'seed_42'


def calibrate(probs, labels, cls, sensitivity_target, specificity_floor):
    # largest theta meeting both constraints, as in the pipeline
    idx = CLASSES.index(cls)
    positive = (labels == idx).astype(int)
    best = None
    for theta in np.linspace(0.01, 0.99, 199):
        pred = (probs[:, idx] >= theta).astype(int)
        tp = int(((pred == 1) & (positive == 1)).sum())
        fp = int(((pred == 1) & (positive == 0)).sum())
        fn = int(((pred == 0) & (positive == 1)).sum())
        tn = int(((pred == 0) & (positive == 0)).sum())
        sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        if spec < specificity_floor or sens < sensitivity_target:
            continue
        if best is None or theta > best['theta']:
            best = {'theta': float(theta), 'val_sensitivity': sens, 'val_specificity': spec}
    return best


def apply_thresholds(probs, thresholds):
    pred = probs.argmax(axis=1)
    for cls in reversed(THRESHOLD_PRIORITY):
        idx = CLASSES.index(cls)
        pred[probs[:, idx] >= thresholds[cls]] = idx
    return pred


def malignant_bacc(y, pred):
    return float(np.mean([np.mean(pred[y == CLASSES.index(c)] == CLASSES.index(c)) for c in MALIGNANT]))


def main():
    val = np.loadtxt(RESULTS_DIR / 'tta_val_sum.csv', delimiter=',')
    y_val = np.loadtxt(RESULTS_DIR / 'tta_val_labels.csv').astype(int)
    test = np.loadtxt(RESULTS_DIR / 'tta_sum_probs.csv', delimiter=',')
    y_test = np.loadtxt(RESULTS_DIR / 'tta_labels.csv').astype(int)

    akiec = calibrate(val, y_val, 'AKIEC', AKIEC_TARGET, AKIEC_SPECIFICITY_FLOOR)
    assert akiec is not None, 'no AKIEC threshold meets the constraints'

    rows = []
    for target in MEL_TARGETS:
        mel = calibrate(val, y_val, 'MEL', target, MEL_SPECIFICITY_FLOOR)
        if mel is None:
            rows.append({'mel_target': target, 'feasible': False})
            continue
        thresholds = {'MEL': mel['theta'], 'AKIEC': akiec['theta']}
        pv, pt = apply_thresholds(val, thresholds), apply_thresholds(test, thresholds)
        rows.append({
            'mel_target': target, 'feasible': True, 'thresholds': thresholds,
            'mel_val_sensitivity': mel['val_sensitivity'], 'mel_val_specificity': mel['val_specificity'],
            'val_bacc': float(balanced_accuracy_score(y_val, pv)), 'val_malignant_bacc': malignant_bacc(y_val, pv),
            'test_bacc': float(balanced_accuracy_score(y_test, pt)), 'test_malignant_bacc': malignant_bacc(y_test, pt),
        })

    out = {'akiec': akiec, 'mel_specificity_floor': MEL_SPECIFICITY_FLOOR, 'sweep': rows}
    with open(RESULTS_DIR / 'threshold_sweep.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f"AKIEC theta = {akiec['theta']:.3f} (val sens {akiec['val_sensitivity']:.3f}, "
          f"spec {akiec['val_specificity']:.3f})")
    print(f"{'MEL target':>11} {'theta':>6} {'val BACC':>9} {'val malig':>10} {'test BACC':>10} {'test malig':>11}")
    for r in rows:
        if not r['feasible']:
            print(f"{r['mel_target']:>11.2f}  not reachable under the specificity floor")
            continue
        print(f"{r['mel_target']:>11.2f} {r['thresholds']['MEL']:>6.3f} {r['val_bacc']:>9.4f} "
              f"{r['val_malignant_bacc']:>10.4f} {r['test_bacc']:>10.4f} {r['test_malignant_bacc']:>11.4f}")


if __name__ == '__main__':
    main()
