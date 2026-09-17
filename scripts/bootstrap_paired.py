"""Paired bootstrap: clinical thresholds vs argmax on the seed-42 test set.

Both decision rules are applied to the same TTA ensemble probabilities and to the same
resampled images in every replicate; the thresholds stay fixed at their validation values.

Inputs (results/seed_42/):
    tta_sum_probs.csv          TTA ensemble probabilities, 2348 x 7, class order CLASSES
    tta_labels.csv             ground-truth class index per image
    calibrated_thresholds.json thresholds calibrated on validation

Output: results/seed_42/bootstrap_summary.json

Usage: python scripts/bootstrap_paired.py [--n-boot 10000] [--seed 42]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
MALIGNANT = ['MEL', 'BCC', 'AKIEC']
THRESHOLD_PRIORITY = ['MEL', 'AKIEC']   # same order as the pipeline: MEL wins over AKIEC

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results' / 'seed_42'


def apply_clinical_thresholds(probs, thresholds):
    # argmax-theta: a prioritised class overrides argmax when its probability exceeds its threshold
    pred = probs.argmax(axis=1)
    for cls in reversed(THRESHOLD_PRIORITY):
        idx = CLASSES.index(cls)
        pred[probs[:, idx] > thresholds[cls]] = idx
    return pred


def malignant_bacc(y, pred):
    return float(np.mean([np.mean(pred[y == CLASSES.index(c)] == CLASSES.index(c)) for c in MALIGNANT]))


def summarise(deltas):
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {'ci95': [float(lo), float(hi)], 'mean': float(deltas.mean()),
            'p_delta_gt_0': float((deltas > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    probs = np.loadtxt(RESULTS_DIR / 'tta_sum_probs.csv', delimiter=',')
    y = np.loadtxt(RESULTS_DIR / 'tta_labels.csv').astype(int)
    with open(RESULTS_DIR / 'calibrated_thresholds.json') as f:
        thresholds = json.load(f)
    assert probs.shape == (len(y), len(CLASSES)), 'probability/label shape mismatch'

    pred_argmax = probs.argmax(axis=1)
    pred_clin = apply_clinical_thresholds(probs, thresholds)

    point = {
        'global_bacc_argmax': float(balanced_accuracy_score(y, pred_argmax)),
        'global_bacc_thresholds': float(balanced_accuracy_score(y, pred_clin)),
        'malignant_bacc_argmax': malignant_bacc(y, pred_argmax),
        'malignant_bacc_thresholds': malignant_bacc(y, pred_clin),
    }

    rng = np.random.default_rng(args.seed)
    n = len(y)
    d_mal = np.empty(args.n_boot)
    d_glob = np.empty(args.n_boot)
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        yb, pa, pc = y[idx], pred_argmax[idx], pred_clin[idx]
        d_mal[b] = malignant_bacc(yb, pc) - malignant_bacc(yb, pa)
        d_glob[b] = balanced_accuracy_score(yb, pc) - balanced_accuracy_score(yb, pa)

    out = {
        'n_images': int(n), 'n_boot': args.n_boot, 'seed': args.seed,
        'rng': 'numpy.random.default_rng', 'thresholds': thresholds,
        'point_estimates': point,
        'delta_malignant_bacc': summarise(d_mal),
        'delta_global_bacc': summarise(d_glob),
    }
    with open(RESULTS_DIR / 'bootstrap_summary.json', 'w') as f:
        json.dump(out, f, indent=2)

    print(f"malignant BACC {point['malignant_bacc_argmax']:.4f} -> {point['malignant_bacc_thresholds']:.4f}, "
          f"95% CI [{out['delta_malignant_bacc']['ci95'][0]:+.3f}, {out['delta_malignant_bacc']['ci95'][1]:+.3f}], "
          f"P(delta>0) = {out['delta_malignant_bacc']['p_delta_gt_0']:.1%}")
    print(f"global BACC    {point['global_bacc_argmax']:.4f} -> {point['global_bacc_thresholds']:.4f}, "
          f"95% CI [{out['delta_global_bacc']['ci95'][0]:+.4f}, {out['delta_global_bacc']['ci95'][1]:+.4f}]")


if __name__ == '__main__':
    main()
