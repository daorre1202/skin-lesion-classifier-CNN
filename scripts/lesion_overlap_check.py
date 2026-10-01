"""Seed-42 results on the test images with and without a lesion-mate in train or validation.

The split is by image, and HAM10000 holds several images of some lesions. The per-image flags
come from the repository skin-lesion-foundation-models (results/lesion_overlap/flags.csv),
which uses the same split, so they apply row by row to the arrays of this repository.

    python scripts/lesion_overlap_check.py

Writes results/seed_42/lesion_overlap_summary.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED42 = ROOT / "results" / "seed_42"
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
MALIGNANT = [0, 2, 3]
PRIORITY = ["MEL", "AKIEC"]


def thresholded(probs, thresholds):
    pred = probs.argmax(axis=1).copy()
    for name in reversed(PRIORITY):
        pred[probs[:, CLASSES.index(name)] > thresholds[name]] = CLASSES.index(name)
    return pred


def recall(y, pred, c):
    return (pred[y == c] == c).mean()


def bacc(y, pred, classes=range(7)):
    return float(np.mean([recall(y, pred, c) for c in classes]))


def boot(y, preds, classes, n_boot, seed):
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_boot) for k in preds}
    for i in range(n_boot):
        s = rng.integers(0, len(y), len(y))
        for k, p in preds.items():
            out[k][i] = np.mean([recall(y[s], p[s], c) for c in classes if (y[s] == c).any()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default=SEED42 / "tta_sum_probs.csv")
    ap.add_argument("--labels", default=SEED42 / "tta_labels.csv")
    ap.add_argument("--thresholds", default=SEED42 / "calibrated_thresholds.json")
    ap.add_argument("--flags", default=ROOT / "results" / "lesion_overlap" / "flags.csv")
    ap.add_argument("--out", default=SEED42 / "lesion_overlap_summary.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    probs = np.loadtxt(a.probs, delimiter=",")
    y = np.loadtxt(a.labels, dtype=int)
    thr = json.loads(Path(a.thresholds).read_text())
    t = pd.read_csv(a.flags)
    t = t[t["split"] == "test"].sort_values("position")
    assert len(t) == len(y) == len(probs)
    known, over = t["has_lesion_id"].to_numpy() == 1, t["overlap"].to_numpy() == 1
    groups = {"all": np.ones(len(y), bool), "no_lesion_mate": known & ~over,
              "lesion_mate": over, "no_lesion_id": ~known}
    arg, thr_pred = probs.argmax(axis=1), thresholded(probs, thr)
    summary = {"test_with_lesion_id": int(known.sum()), "test_with_lesion_mate": int(over.sum()), "groups": {}}
    for name, m in groups.items():
        g = {"n": int(m.sum()), "bacc_argmax": bacc(y[m], arg[m]), "bacc_thresholds": bacc(y[m], thr_pred[m]),
             "malignant_argmax": bacc(y[m], arg[m], MALIGNANT), "malignant_thresholds": bacc(y[m], thr_pred[m], MALIGNANT)}
        if name in ("all", "no_lesion_mate"):
            b = boot(y[m], {"a": arg[m], "t": thr_pred[m]}, range(7), a.n_boot, 42)
            mb = boot(y[m], {"a": arg[m], "t": thr_pred[m]}, MALIGNANT, a.n_boot, 42)
            d = mb["t"] - mb["a"]
            g["bacc_argmax_ci"] = [float(np.percentile(b["a"], q)) for q in (2.5, 97.5)]
            g["bacc_thresholds_ci"] = [float(np.percentile(b["t"], q)) for q in (2.5, 97.5)]
            g["malignant_gain_ci"] = [float(np.percentile(d, q)) for q in (2.5, 97.5)]
        summary["groups"][name] = g
        print(f"{name:15s} n={g['n']:5d}  BACC {g['bacc_argmax']:.4f} / {g['bacc_thresholds']:.4f}  "
              f"malignant {g['malignant_argmax']:.4f} -> {g['malignant_thresholds']:.4f}")
    Path(a.out).write_text(json.dumps(summary, indent=2))
    print(f"lesion-mate in train/val: {over.sum()} of {known.sum()} ({100 * over.sum() / known.sum():.0f}%); saved {a.out}")


if __name__ == "__main__":
    main()
