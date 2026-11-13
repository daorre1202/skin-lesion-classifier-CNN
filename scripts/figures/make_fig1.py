"""Figure 1 of the BIOIMAGING 2027 paper: energy proportion by class.

Drawn at the final column width (7.5 cm) so the text lands at its nominal
size on the page (7-7.5 pt). Reads the per-image records of the Grad-CAM audit.

    python scripts/figures/make_fig1.py --out fig_gradcam_seg_energy.pdf

Needs matplotlib and the Liberation Serif font (metric-compatible with Times).
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
MALIGNANT = {"MEL", "BCC", "AKIEC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=ROOT / "results/attention/gradcam_segmentation_overlap.json")
    ap.add_argument("--out", default="fig_gradcam_seg_energy.pdf")
    ap.add_argument("--font", default="/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf")
    args = ap.parse_args()

    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update({
        "font.family": "Liberation Serif", "font.size": 7.5, "pdf.fonttype": 42,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    })
    rec = json.load(open(args.records))["records"]
    data = [[r["energy_prop"] for r in rec if r["true"] == c] for c in CLASSES]
    overall = float(np.mean([r["energy_prop"] for r in rec]))

    fig, ax = plt.subplots(figsize=(7.5 / 2.54, 2.25))
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=2.2, markeredgewidth=0.5,
                                    markerfacecolor="none", markeredgecolor="0.25"),
                    medianprops=dict(color="black", linewidth=0.9),
                    whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6),
                    boxprops=dict(linewidth=0.6))
    for patch, c in zip(bp["boxes"], CLASSES):
        patch.set_facecolor("#d62728" if c in MALIGNANT else "#1f77b4")
        patch.set_alpha(0.55)
    ax.axhline(overall, color="0.35", linestyle="--", linewidth=0.7)
    ax.set_xticks(range(1, 8), CLASSES)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Energy proportion inside lesion", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout(pad=0.15)
    fig.savefig(args.out)
    print(f"overall mean {overall:.3f}, n = {len(rec)}, wrote {args.out}")


if __name__ == "__main__":
    main()
