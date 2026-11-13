"""Effect sizes and robustness checks for the attention audit (seed 42).

Reads the per-image records of the Grad-CAM audit, the TTA ensemble test
probabilities, the calibrated thresholds and the lesion-overlap flags.

    python scripts/attention_stats.py

Writes results/attention/attention_stats.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
MALIGNANT = {"MEL", "BCC", "AKIEC"}
PRIORITY = ["MEL", "AKIEC"]


def rank_biserial(a, b):
    # positive when a tends to be larger than b
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def compare(a, b, rng, n_boot):
    a, b = np.asarray(a), np.asarray(b)
    boots = [rank_biserial(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)]
    return {
        "n": [len(a), len(b)],
        "mean": [round(float(a.mean()), 3), round(float(b.mean()), 3)],
        "median": [round(float(np.median(a)), 3), round(float(np.median(b)), 3)],
        "r_rb": round(float(rank_biserial(a, b)), 3),
        "r_rb_ci95": [round(float(x), 3) for x in np.percentile(boots, [2.5, 97.5])],
        "p_mannwhitney": float(f"{mannwhitneyu(a, b, alternative='two-sided').pvalue:.2g}"),
    }


def analyse(df, rng, n_boot):
    mal = df.true.isin(MALIGNANT)
    out = {
        "n": len(df),
        "spearman_ep_maskfrac": round(float(spearmanr(df.energy_prop, df.mask_frac).statistic), 3),
        "malignant_vs_benign": {
            "energy_prop": compare(df.energy_prop[mal], df.energy_prop[~mal], rng, n_boot),
            "ep_over_maskfrac": compare(df.norm[mal], df.norm[~mal], rng, n_boot),
            "mask_frac": compare(df.mask_frac[mal], df.mask_frac[~mal], rng, n_boot),
        },
    }
    q = pd.qcut(df.mask_frac, 5, labels=False)
    out["malignant_vs_benign"]["energy_prop_by_maskfrac_quintile"] = [
        compare(df.energy_prop[(q == k) & mal], df.energy_prop[(q == k) & ~mal], rng, n_boot)
        for k in range(5)
    ]
    # error vs correct; first group is the errors
    for col, name in [("correct", "resnet50"), ("ens_correct", "calibrated_ensemble")]:
        res = {"pooled": compare(df.energy_prop[~df[col]], df.energy_prop[df[col]], rng, n_boot)}
        for c in CLASSES:
            s = df[df.true == c]
            if (~s[col]).sum() >= 5 and s[col].sum() >= 5:
                res[c] = compare(s.energy_prop[~s[col]], s.energy_prop[s[col]], rng, n_boot)
        out["error_vs_correct_" + name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=ROOT / "results" / "attention" / "gradcam_segmentation_overlap.json")
    ap.add_argument("--probs", default=ROOT / "results" / "seed_42" / "tta_sum_probs.csv")
    ap.add_argument("--labels", default=ROOT / "results" / "seed_42" / "tta_labels.csv")
    ap.add_argument("--thresholds", default=ROOT / "results" / "seed_42" / "calibrated_thresholds.json")
    ap.add_argument("--flags", default=ROOT / "results" / "lesion_overlap" / "flags.csv")
    ap.add_argument("--out", default=ROOT / "results" / "attention" / "attention_stats.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    rec = pd.DataFrame(json.loads(Path(a.records).read_text())["records"])
    probs = np.loadtxt(a.probs, delimiter=",")
    y = np.loadtxt(a.labels, dtype=int)
    thr = json.loads(Path(a.thresholds).read_text())
    pred = probs.argmax(axis=1).copy()
    for name in reversed(PRIORITY):
        pred[probs[:, CLASSES.index(name)] > thr[name]] = CLASSES.index(name)

    flags = pd.read_csv(a.flags)
    test = flags[flags.split == "test"].copy()
    test["label"] = [CLASSES[i] for i in y[test.position]]
    test["ens_pred"] = [CLASSES[i] for i in pred[test.position]]
    df = rec.merge(test[["image_id", "overlap", "label", "ens_pred"]], left_on="id", right_on="image_id")
    # alignment check: every audited image found, labels agree
    assert len(df) == len(rec) and (df.true == df.label).all()
    df["ens_correct"] = df.ens_pred == df.true
    df["norm"] = df.energy_prop / df.mask_frac

    rng = np.random.default_rng(42)
    out = {
        "note": "errors in the per-image records are ResNet-50 (seed 42) argmax; "
                "calibrated_ensemble = TTA ensemble with clinical thresholds",
        "all_masked": analyse(df, rng, a.n_boot),
        "no_lesion_mate": analyse(df[df.overlap == 0], rng, a.n_boot),
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
