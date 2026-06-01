#!/usr/bin/env python3
"""
Ensemble output visualisation.

Loads the TTA probability arrays saved during training and generates
figures that show what the ensemble actually produces: for each image,
a probability distribution over the 7 classes, not a single hard label.

This visualisation is the honest representation of the model output —
the final class assignment (argmax or clinical threshold) is derived
FROM these distributions, not produced directly.

Usage:
    python scripts/visualize_ensemble.py \\
        --checkpoint outputs/2026-05-20_13-15-53

    # More images per class
    python scripts/visualize_ensemble.py \\
        --checkpoint outputs/2026-05-20_13-15-53 --n-per-class 3

    # Save figures to a specific folder
    python scripts/visualize_ensemble.py \\
        --checkpoint outputs/2026-05-20_13-15-53 --out-dir figures/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
CLASS_NAMES = {
    'MEL':   'Melanoma',
    'NV':    'Melanocytic Nevus',
    'BCC':   'Basal Cell Carcinoma',
    'AKIEC': 'Actinic Keratoses',
    'BKL':   'Benign Keratosis',
    'DF':    'Dermatofibroma',
    'VASC':  'Vascular Lesion',
}
MALIGNANT = {'MEL', 'BCC', 'AKIEC'}

# Colours consistent with the pipeline diagram
COLOR_MALIGNANT = '#c0392b'
COLOR_BENIGN    = '#2980b9'
COLOR_ARGMAX    = '#27ae60'
COLOR_THRESHOLD = '#e67e22'


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualise TTA ensemble probability distributions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--checkpoint',  required=True,
                        help='Training output directory (must contain tta_sum_probs.npy)')
    parser.add_argument('--n-per-class', type=int, default=2,
                        help='Images to show per class (default: 2)')
    parser.add_argument('--out-dir',     default=None,
                        help='Output folder for figures (default: <checkpoint>/figures/)')
    return parser.parse_args()


# ── Data loading ─────────────────────────────────────────────────────────────

def load_arrays(checkpoint_dir: str) -> dict:
    """Load all numpy arrays and metadata from a checkpoint directory."""
    required = ['tta_sum_probs.npy', 'tta_labels.npy', 'classes_used.json']
    for fname in required:
        path = os.path.join(checkpoint_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run a full training first to generate the TTA arrays."
            )

    data = {
        'probs':        np.load(os.path.join(checkpoint_dir, 'tta_sum_probs.npy')),
        'labels':       np.load(os.path.join(checkpoint_dir, 'tta_labels.npy')),
        'classes_used': json.load(open(os.path.join(checkpoint_dir, 'classes_used.json'))),
    }

    thresh_path = os.path.join(checkpoint_dir, 'calibrated_thresholds.json')
    data['thresholds'] = (json.load(open(thresh_path))
                          if os.path.exists(thresh_path) else {})

    split_path = os.path.join(checkpoint_dir, 'split_assignment.json')
    data['split'] = (json.load(open(split_path))
                     if os.path.exists(split_path) else {})

    print(f"✓ Loaded: {len(data['labels'])} test images, "
          f"{len(data['classes_used'])} classes")
    if data['thresholds']:
        print(f"  Clinical thresholds: " +
              "  ".join(f"{k}=θ{v:.3f}" for k, v in data['thresholds'].items()))
    return data


def find_image_paths(checkpoint_dir: str, split: dict) -> dict:
    """
    Try to locate test image files from the split assignment JSON.
    Returns {image_id: path} for found images, empty dict if not found.
    """
    test_ids = {iid for iid, sp in split.items() if sp == 'test'}
    if not test_ids:
        return {}

    # Look for images relative to the checkpoint dir or common locations
    search_dirs = [
        os.path.join(checkpoint_dir, '..', '..', 'ISIC2018'),   # Drive layout
        '/content/data/isic2018_task3_prepared/test',            # Colab prep dir
        '/kaggle/working/isic2018_task3_prepared/test',          # Kaggle prep dir
    ]

    found = {}
    for search_dir in search_dirs:
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            import glob
            for path in glob.glob(os.path.join(search_dir, '**', ext), recursive=True):
                iid = os.path.splitext(os.path.basename(path))[0]
                if iid in test_ids:
                    found[iid] = path
        if found:
            print(f"  Found {len(found)} image files in {search_dir}")
            break

    return found


def get_predicted_class(probs_row: np.ndarray, classes_used: list,
                        thresholds: dict) -> tuple:
    """
    Return (argmax_class, threshold_class).
    threshold_class differs from argmax_class only when a clinical
    threshold is exceeded and overrides the argmax decision.
    """
    argmax_idx   = int(probs_row.argmax())
    argmax_class = classes_used[argmax_idx]

    thresh_class = argmax_class
    # Apply in reverse priority (MEL overrides AKIEC)
    for cls in ['AKIEC', 'MEL']:
        if cls in thresholds and cls in classes_used:
            idx = classes_used.index(cls)
            if probs_row[idx] > thresholds[cls]:
                thresh_class = cls

    return argmax_class, thresh_class


# ── Figure 1: probability distributions per image ────────────────────────────

def plot_probability_distributions(
    probs:        np.ndarray,
    labels:       np.ndarray,
    classes_used: list,
    thresholds:   dict,
    id2path:      dict,
    split:        dict,
    n_per_class:  int,
    out_path:     str,
) -> None:
    """
    For a sample of test images (n_per_class per true class), show:
      - The dermoscopy image (if available)
      - Bar chart of the 7 class probabilities as percentages
      - True label highlighted in green
      - Argmax prediction marked
      - Clinical threshold line (where applicable)
    """
    # Build {class_idx: [sample_indices]} — prefer correctly classified
    preds_argmax = probs.argmax(axis=1)
    samples_by_class = {i: [] for i in range(len(classes_used))}
    for idx in range(len(labels)):
        samples_by_class[labels[idx]].append(idx)

    selected = []
    for cls_idx in range(len(classes_used)):
        candidates = samples_by_class[cls_idx]
        # Prefer correct predictions, then fall back to any
        correct   = [i for i in candidates if preds_argmax[i] == cls_idx]
        pool      = correct if correct else candidates
        selected += pool[:n_per_class]

    n_imgs  = len(selected)
    has_img = bool(id2path)

    cols  = n_per_class * len(classes_used)
    rows  = 3 if has_img else 2   # row 0: image (optional), row 1: probs, row 2: label
    fig   = plt.figure(figsize=(max(cols * 2.5, 14), rows * 3.5 + 1))
    fig.suptitle(
        'TTA Ensemble — Probability distribution per test image\n'
        'Each bar = P(class | image) averaged over 10 augmentation rounds',
        fontsize=11, fontweight='bold', y=0.98,
    )

    # Build test_id → sample_idx mapping if images are available
    test_ids_ordered = []
    if split:
        for iid, sp in split.items():
            if sp == 'test':
                test_ids_ordered.append(iid)

    n_classes = len(classes_used)
    bar_colors = [COLOR_MALIGNANT if c in MALIGNANT else COLOR_BENIGN
                  for c in classes_used]

    for col_idx, sample_idx in enumerate(selected):
        true_cls  = classes_used[labels[sample_idx]]
        row_probs = probs[sample_idx]
        argmax_c, thresh_c = get_predicted_class(row_probs, classes_used, thresholds)
        correct = (thresh_c == true_cls)

        # ── Image panel ──────────────────────────────────────────────────
        img_row = 0
        if has_img:
            ax_img = fig.add_subplot(rows, cols, col_idx + 1)
            iid    = (test_ids_ordered[sample_idx]
                      if sample_idx < len(test_ids_ordered) else None)
            if iid and iid in id2path:
                img = np.array(Image.open(id2path[iid]).convert('RGB').resize((150, 150)))
                ax_img.imshow(img)
                border_color = '#27ae60' if correct else '#c0392b'
                for spine in ax_img.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
            else:
                ax_img.set_facecolor('#f0f0f0')
                ax_img.text(0.5, 0.5, 'image\nnot found',
                            ha='center', va='center', fontsize=7, color='gray',
                            transform=ax_img.transAxes)
            ax_img.set_xticks([]); ax_img.set_yticks([])
            verdict = '✓ correct' if correct else '✗ wrong'
            ax_img.set_title(f'True: {true_cls}\n{verdict}',
                             fontsize=7.5, color='#27ae60' if correct else '#c0392b',
                             fontweight='bold')

        # ── Probability bar chart ─────────────────────────────────────────
        prob_row_offset = cols if has_img else 0
        ax_bar = fig.add_subplot(rows, cols, prob_row_offset + col_idx + 1)

        bars = ax_bar.bar(
            range(n_classes),
            row_probs * 100,
            color=bar_colors,
            edgecolor='white',
            linewidth=0.5,
            width=0.7,
        )

        # Mark true class (green border)
        true_idx = classes_used.index(true_cls)
        bars[true_idx].set_edgecolor('#27ae60')
        bars[true_idx].set_linewidth(2.5)

        # Mark argmax (star)
        argmax_idx = classes_used.index(argmax_c)
        ax_bar.text(argmax_idx, row_probs[argmax_idx] * 100 + 1.5,
                    '▲', ha='center', fontsize=7, color='#27ae60')

        # Clinical threshold lines
        for cls_t, theta in thresholds.items():
            if cls_t in classes_used:
                t_idx = classes_used.index(cls_t)
                ax_bar.axhline(
                    y=theta * 100, xmin=t_idx / n_classes,
                    xmax=(t_idx + 1) / n_classes,
                    color=COLOR_THRESHOLD, linewidth=1.5, linestyle='--',
                )

        # Probability values on top of bars
        for i, (bar, p) in enumerate(zip(bars, row_probs)):
            if p > 0.03:
                ax_bar.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.5,
                            f'{p*100:.0f}%',
                            ha='center', va='bottom', fontsize=6.5)

        ax_bar.set_ylim(0, 105)
        ax_bar.set_xticks(range(n_classes))
        ax_bar.set_xticklabels(classes_used, fontsize=7)
        ax_bar.set_ylabel('Probability (%)', fontsize=7)
        ax_bar.tick_params(axis='y', labelsize=6)
        ax_bar.grid(axis='y', alpha=0.25, linewidth=0.5)

        # Threshold override label
        if thresh_c != argmax_c:
            ax_bar.set_title(
                f'→ θ override: {argmax_c}→{thresh_c}',
                fontsize=7, color=COLOR_THRESHOLD
            )

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLOR_MALIGNANT, label='Malignant / pre-malignant'),
        mpatches.Patch(color=COLOR_BENIGN,    label='Benign'),
        plt.Line2D([0], [0], color=COLOR_THRESHOLD, linestyle='--',
                   label='Clinical threshold'),
        plt.Line2D([0], [0], marker='^', color='#27ae60', linestyle='None',
                   markersize=8, label='Argmax prediction'),
        mpatches.Patch(facecolor='white', edgecolor='#27ae60',
                       linewidth=2, label='True label'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 1 saved: {out_path}")


# ── Figure 2: threshold effect ───────────────────────────────────────────────

def plot_threshold_effect(
    probs:        np.ndarray,
    labels:       np.ndarray,
    classes_used: list,
    thresholds:   dict,
    out_path:     str,
) -> None:
    """
    Side-by-side comparison of the probability distribution for images
    where the clinical threshold changed the final prediction.
    Shows exactly why the threshold matters clinically.
    """
    if not thresholds:
        print("  No thresholds found — skipping threshold effect figure")
        return

    # Find images where threshold changes prediction
    changed = []
    for idx in range(len(labels)):
        row         = probs[idx]
        argmax_c, thresh_c = get_predicted_class(row, classes_used, thresholds)
        if thresh_c != argmax_c:
            changed.append({
                'idx':        idx,
                'true_cls':   classes_used[labels[idx]],
                'argmax_cls': argmax_c,
                'thresh_cls': thresh_c,
                'probs':      row,
            })

    if not changed:
        print("  No threshold overrides found in this run")
        return

    n_show = min(len(changed), 6)
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 3.5, 5))
    if n_show == 1:
        axes = [axes]
    fig.suptitle(
        f'Clinical threshold effect — {len(changed)} images where '
        f'threshold overrode argmax\n'
        'These are cases where the model assigned the highest probability '
        'to a non-malignant class,\n'
        'but the malignant class exceeded its clinical threshold.',
        fontsize=9, fontweight='bold',
    )

    n_classes  = len(classes_used)
    bar_colors = [COLOR_MALIGNANT if c in MALIGNANT else COLOR_BENIGN
                  for c in classes_used]

    for col, case in enumerate(changed[:n_show]):
        ax   = axes[col]
        row  = case['probs']
        bars = ax.bar(range(n_classes), row * 100,
                      color=bar_colors, edgecolor='white', linewidth=0.5)

        # Mark true class
        true_idx = classes_used.index(case['true_cls'])
        bars[true_idx].set_edgecolor('#27ae60'); bars[true_idx].set_linewidth(2.5)

        # Mark argmax (what the model would predict without threshold)
        am_idx = classes_used.index(case['argmax_cls'])
        ax.text(am_idx, row[am_idx] * 100 + 1.5, '▲',
                ha='center', fontsize=8, color='#27ae60')

        # Mark threshold-overridden class
        th_idx = classes_used.index(case['thresh_cls'])
        for cls_t, theta in thresholds.items():
            if cls_t in classes_used:
                t_i = classes_used.index(cls_t)
                ax.axhline(
                    y=theta * 100, xmin=t_i / n_classes,
                    xmax=(t_i + 1) / n_classes,
                    color=COLOR_THRESHOLD, linewidth=2, linestyle='--',
                )
        ax.text(th_idx, row[th_idx] * 100 + 1.5, '◆',
                ha='center', fontsize=8, color=COLOR_THRESHOLD)

        # Values
        for i, (bar, p) in enumerate(zip(bars, row)):
            if p > 0.04:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f'{p*100:.0f}%', ha='center', va='bottom', fontsize=7)

        ax.set_ylim(0, 108)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(classes_used, fontsize=7)
        ax.set_ylabel('Probability (%)', fontsize=7)
        ax.tick_params(axis='y', labelsize=6)
        ax.grid(axis='y', alpha=0.25)
        ax.set_title(
            f'True: {case["true_cls"]}\n'
            f'Argmax ▲: {case["argmax_cls"]}\n'
            f'Threshold ◆: {case["thresh_cls"]}',
            fontsize=8,
            color='#27ae60' if case['thresh_cls'] == case['true_cls'] else '#c0392b',
        )

    legend_handles = [
        mpatches.Patch(color=COLOR_MALIGNANT, label='Malignant'),
        mpatches.Patch(color=COLOR_BENIGN,    label='Benign'),
        plt.Line2D([0], [0], color=COLOR_THRESHOLD, linestyle='--',
                   label='θ threshold'),
        plt.Line2D([0], [0], marker='^', color='#27ae60', linestyle='None',
                   markersize=9, label='Argmax'),
        plt.Line2D([0], [0], marker='D', color=COLOR_THRESHOLD, linestyle='None',
                   markersize=9, label='Threshold override'),
        mpatches.Patch(facecolor='white', edgecolor='#27ae60',
                       linewidth=2, label='True label'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=6, fontsize=8, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 2 saved: {out_path}")


# ── Figure 3: summary distribution heatmap ───────────────────────────────────

def plot_class_confidence_summary(
    probs:        np.ndarray,
    labels:       np.ndarray,
    classes_used: list,
    out_path:     str,
) -> None:
    """
    Heatmap of mean probability assigned to each class, grouped by true label.
    Shows the model's confidence patterns: high diagonal = good discrimination.
    """
    import seaborn as sns
    n = len(classes_used)
    matrix = np.zeros((n, n))

    for true_idx in range(n):
        mask = labels == true_idx
        if mask.sum() > 0:
            matrix[true_idx] = probs[mask].mean(axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=classes_used, yticklabels=classes_used,
        ax=ax, linewidths=0.5, cbar_kws={'label': 'Mean probability (%)'},
    )
    ax.set_xlabel('Probability assigned to class', fontsize=10)
    ax.set_ylabel('True class', fontsize=10)
    ax.set_title(
        'Mean probability distribution by true class\n'
        'Diagonal = correct class confidence  |  '
        'Off-diagonal = confusion patterns',
        fontsize=10, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 3 saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    checkpoint = args.checkpoint

    if not os.path.isdir(checkpoint):
        print(f"Error: directory not found — {checkpoint}")
        sys.exit(1)

    out_dir = args.out_dir or os.path.join(checkpoint, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nCheckpoint : {checkpoint}")
    print(f"Output dir : {out_dir}\n")

    # Load data
    data = load_arrays(checkpoint)
    probs        = data['probs']
    labels       = data['labels']
    classes_used = data['classes_used']
    thresholds   = data['thresholds']
    split        = data['split']

    # Try to find image files
    id2path = find_image_paths(checkpoint, split)
    if not id2path:
        print("  ⚠ Image files not found — figures will show bars only (no photos)")

    # Figure 1 — probability distributions
    plot_probability_distributions(
        probs, labels, classes_used, thresholds, id2path, split,
        n_per_class = args.n_per_class,
        out_path    = os.path.join(out_dir, '1_probability_distributions.png'),
    )

    # Figure 2 — threshold override cases
    plot_threshold_effect(
        probs, labels, classes_used, thresholds,
        out_path = os.path.join(out_dir, '2_threshold_effect.png'),
    )

    # Figure 3 — confidence heatmap
    plot_class_confidence_summary(
        probs, labels, classes_used,
        out_path = os.path.join(out_dir, '3_confidence_heatmap.png'),
    )

    print(f"\n✓ All figures saved to: {out_dir}")
    print(f"  Run with --n-per-class N to show more examples per class")


if __name__ == '__main__':
    main()
