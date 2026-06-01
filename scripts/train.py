#!/usr/bin/env python3
"""
Main training entry point.

Usage:
    # Full training from scratch
    python scripts/train.py --config configs/config.yaml --seed 42

    # Resume from existing checkpoints
    python scripts/train.py --config configs/config.yaml \\
        --resume outputs/2026-05-20_13-15-53 --seed 42

    # Force-retrain a specific model
    python scripts/train.py --config configs/config.yaml \\
        --resume outputs/2026-05-20_13-15-53 \\
        --force-retrain efficientnet_b3 --seed 42

    # Recalibrate thresholds only (no retraining)
    python scripts/train.py --config configs/config.yaml \\
        --load-tta outputs/2026-05-20_13-15-53 --seed 42
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

# Add src to path (allows running from the project root without installing)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from skin_classifier.data       import (SkinDataset, merge_groundtruth_csvs,
                                         split_dataset, load_split_from_file,
                                         build_transforms)
from skin_classifier.models     import (build_model, FocalLoss,
                                         compute_ensemble_probs)
from skin_classifier.training   import (train_one_epoch, evaluate,
                                         EarlyStopping, CheckpointMeta,
                                         build_weighted_sampler)
from skin_classifier.inference  import (predict_with_tta, calibrate_threshold,
                                         apply_clinical_thresholds)
from skin_classifier.utils      import (log, Tee, index_images,
                                         prepare_directories,
                                         compute_per_class_metrics)

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN ensemble for skin lesion classification"
    )
    parser.add_argument('--config',        required=True,  help="Path to config.yaml")
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--resume',        default=None,   help="Checkpoint directory to resume from")
    parser.add_argument('--force-retrain', nargs='+', default=[],
                        help="Model names to retrain even if checkpoint exists")
    parser.add_argument('--load-tta',      default=None,
                        help="Directory with saved TTA arrays (skip training)")
    parser.add_argument('--output-dir',    default=None,
                        help="Override output directory")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    cfg    = yaml.safe_load(open(args.config))
    seed   = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    print(f"✓ Device: {device} | Seed: {seed}")

    # ── Output directory ──────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.load_tta:
        out_dir = args.output_dir or os.path.join(cfg['paths']['outputs'], ts + "_thresh")
    elif args.resume:
        out_dir = args.output_dir or os.path.join(cfg['paths']['outputs'], ts + "_r")
    else:
        out_dir = args.output_dir or os.path.join(cfg['paths']['outputs'], ts)

    os.makedirs(out_dir, exist_ok=True)

    # Copy checkpoints when resuming
    if args.resume and os.path.isdir(args.resume):
        for fname in os.listdir(args.resume):
            if fname.endswith(('.pth', '_meta.json', '_history.json')) \
                    or fname == 'split_assignment.json':
                src = os.path.join(args.resume, fname)
                dst = os.path.join(out_dir, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    # ── Logging ───────────────────────────────────────────────────────────
    tee = Tee(os.path.join(out_dir, "training_log.txt"))
    sys.stdout = tee

    try:
        _run_pipeline(args, cfg, seed, device, out_dir, ts)
    except Exception as e:
        log(f"\n✗ ERROR {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
    finally:
        sys.stdout = tee.terminal
        tee.close()


def _run_pipeline(args, cfg, seed, device, out_dir, ts):
    """Core training/inference pipeline."""

    # ── Data ─────────────────────────────────────────────────────────────
    label_map = merge_groundtruth_csvs(cfg['paths']['csv_dir'])
    id2path   = index_images(cfg['paths']['source_dir'])
    id2path   = {k: v for k, v in id2path.items() if k in label_map}

    split_file = os.path.join(out_dir, 'split_assignment.json')
    if os.path.exists(split_file):
        log("✓ Loading split from split_assignment.json")
        prep_train, prep_val, prep_test = load_split_from_file(
            split_file, id2path, label_map
        )
    else:
        prep_train, prep_val, prep_test = split_dataset(
            label_map, id2path,
            train_ratio  = cfg['dataset']['train_ratio'],
            val_ratio    = cfg['dataset']['val_ratio'],
            seed         = seed,
            use_percent  = cfg['dataset']['use_percent'],
            save_path    = split_file,
        )

    classes_used = [
        c for c in CLASSES
        if len(prep_train.get(c, [])) >= cfg['dataset']['min_samples']
        and len(prep_val.get(c,   [])) >= cfg['dataset']['min_samples']
        and len(prep_test.get(c,  [])) >= cfg['dataset']['min_samples']
    ]
    log(f"✓ Classes used: {classes_used}")

    prepare_directories(
        prep_train, prep_val, prep_test,
        classes_used, cfg['paths']['prep_dir']
    )

    # ── Training ──────────────────────────────────────────────────────────
    model_names      = cfg['models']['architectures']
    trained_models   = {}
    val_baccs        = {}
    test_baccs       = {}
    model_times      = {}
    es_trackers      = {}
    test_datasets    = {}
    val_datasets     = {}
    test_loaders     = {}
    eval_criterion   = nn.CrossEntropyLoss()
    num_workers      = cfg['training'].get('num_workers', 0)

    for name in model_names:
        img_size   = cfg['models']['img_size'][name]
        batch_size = cfg['models']['batch_size'][name]
        transforms = build_transforms(img_size)

        val_ds  = SkinDataset(
            Path(cfg['paths']['prep_dir']) / 'val',
            classes_used, is_train=False, transforms=transforms
        )
        test_ds = SkinDataset(
            Path(cfg['paths']['prep_dir']) / 'test',
            classes_used, is_train=False, transforms=transforms
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )
        test_datasets[name] = test_ds
        val_datasets[name]  = val_ds
        test_loaders[name]  = test_loader

        ckpt_path = os.path.join(out_dir, f"{name}_best.pth")
        meta_path = os.path.join(out_dir, f"{name}_meta.json")
        hist_path = os.path.join(out_dir, f"{name}_history.json")

        # Load checkpoint if available and not in force-retrain list
        if (os.path.exists(ckpt_path) and os.path.exists(meta_path)
                and name not in args.force_retrain):
            log(f"\n{'='*50}\n  Loading checkpoint: {name}\n{'='*50}")
            model = build_model(name, len(classes_used), device)
            model.load_state_dict(
                torch.load(ckpt_path, weights_only=False, map_location=device)
            )
            trained_models[name] = model
            with open(meta_path) as f:
                meta = json.load(f)
            val_baccs[name]   = meta['val_bacc']
            test_baccs[name]  = meta.get('test_bacc', 0.0)
            model_times[name] = meta['tiempo_segundos']
            es_trackers[name] = CheckpointMeta(meta, cfg['early_stopping']['bacc_weight'][name])
            log(f"  ✓ {name} — Val BACC: {meta['val_bacc']:.4f} | Epoch: {meta['best_epoch']}")
            continue

        # Train from scratch
        log(f"\n{'='*50}\n  Training: {name}\n{'='*50}")
        model   = build_model(name, len(classes_used), device)
        t_start = time.time()

        train_ds = SkinDataset(
            Path(cfg['paths']['prep_dir']) / 'train',
            classes_used, is_train=True, transforms=transforms,
            clinical_boost=cfg['augmentation'].get('clinical_boost', {})
        )
        sampler      = build_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers
        )

        gamma      = cfg['loss']['focal_gamma'][name]
        smoothing  = cfg['loss']['label_smoothing']
        train_crit = FocalLoss(gamma=gamma, label_smoothing=smoothing)

        es = EarlyStopping(
            patience        = cfg['early_stopping']['patience'][name],
            min_delta       = cfg['early_stopping']['min_delta'][name],
            checkpoint_path = ckpt_path,
            bacc_weight     = cfg['early_stopping']['bacc_weight'][name],
        )
        optimizer = optim.Adam(
            model.parameters(),
            lr           = cfg['models']['lr'][name],
            weight_decay = cfg['models']['weight_decay'][name],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor   = cfg['scheduler']['factor'],
            patience = cfg['scheduler']['patience'][name],
        )
        history = {k: [] for k in
                   ('train_loss', 'train_acc', 'train_bacc',
                    'val_loss',   'val_acc',   'val_bacc')}

        for epoch in range(1, cfg['training']['max_epochs'] + 1):
            tr_loss, tr_acc, tr_bacc = train_one_epoch(
                model, train_loader, train_crit, optimizer, device
            )
            va_loss, va_acc, va_bacc, _, _, _ = evaluate(
                model, val_loader, eval_criterion, device
            )
            scheduler.step(-es.score(va_bacc, va_loss))
            for k, v in zip(
                ['train_loss','train_acc','train_bacc','val_loss','val_acc','val_bacc'],
                [tr_loss, tr_acc, tr_bacc, va_loss, va_acc, va_bacc]
            ):
                history[k].append(v)
            log(
                f"  Epoch {epoch:3d}/{cfg['training']['max_epochs']} | "
                f"Train Loss: {tr_loss:.4f}  BACC: {tr_bacc:.4f} | "
                f"Val Loss: {va_loss:.4f}  BACC: {va_bacc:.4f}"
            )
            if es.step(va_bacc, va_loss, model, epoch):
                log(f"  ⏹ Early stopping — best epoch: {es.best_epoch}")
                break

        model.load_state_dict(
            torch.load(ckpt_path, weights_only=False, map_location=device)
        )
        trained_models[name] = model
        val_baccs[name]      = es.best_bacc
        es_trackers[name]    = es

        elapsed            = time.time() - t_start
        model_times[name]  = elapsed
        mins, secs         = divmod(int(elapsed), 60)
        log(
            f"  ✓ Best checkpoint — epoch {es.best_epoch} | "
            f"Val BACC: {es.best_bacc:.4f} | Time: {mins}m {secs}s"
        )

        _, _, test_bacc, _, _, _ = evaluate(
            model, test_loader, eval_criterion, device
        )
        test_baccs[name] = test_bacc
        log(f"  — Test BACC: {test_bacc:.4f}")

        with open(meta_path, 'w') as f:
            json.dump({
                'val_bacc':        es.best_bacc,
                'val_loss':        es.best_loss,
                'test_bacc':       test_bacc,
                'best_epoch':      es.best_epoch,
                'tiempo_segundos': elapsed,
            }, f, indent=2)
        with open(hist_path, 'w') as f:
            json.dump(history, f)

    # ── Ensemble + TTA ────────────────────────────────────────────────────
    log("\n===== WEIGHTED ENSEMBLE (test) =====")
    ens_probs, ens_labels = compute_ensemble_probs(
        trained_models, val_baccs,
        DataLoader(test_datasets[model_names[0]], batch_size=32,
                   shuffle=False, num_workers=num_workers),
        device
    )
    ens_bacc = balanced_accuracy_score(ens_labels, ens_probs.argmax(axis=1))
    log(f"✓ Ensemble Test BACC: {ens_bacc:.4f}")

    log(f"\n===== TTA ENSEMBLE (test) — {cfg['inference']['tta_rounds']} rounds =====")
    tta_rounds    = cfg['inference']['tta_rounds']
    tta_sum       = None
    tta_labels    = None
    tta_val_sum   = None
    tta_val_labels = None

    for name, model in trained_models.items():
        w = val_baccs[name] / sum(val_baccs.values())
        log(f"  {name} (weight {w*100:.1f}%)...")
        tp, tl = predict_with_tta(
            model, test_datasets[name],
            n_rounds=tta_rounds, device=device, num_workers=num_workers
        )
        vp, vl = predict_with_tta(
            model, val_datasets[name],
            n_rounds=tta_rounds, device=device, num_workers=num_workers
        )
        if tta_sum is None:
            tta_sum = tp * w; tta_labels = tl
            tta_val_sum = vp * w; tta_val_labels = vl
        else:
            tta_sum += tp * w
            tta_val_sum += vp * w

    tta_bacc = balanced_accuracy_score(tta_labels, tta_sum.argmax(axis=1))
    log(f"✓ TTA Ensemble Test BACC: {tta_bacc:.4f}")

    # ── Clinical threshold calibration ────────────────────────────────────
    log("\n===== CLINICAL THRESHOLD CALIBRATION (val) =====")
    thresholds = {}
    for cls_name, tgt in cfg['clinical']['thresholds'].items():
        if cls_name not in classes_used:
            continue
        theta = calibrate_threshold(
            val_probs          = tta_val_sum,
            val_labels         = tta_val_labels,
            cls_idx            = classes_used.index(cls_name),
            cls_name           = cls_name,
            sensitivity_target = tgt['sensitivity_target'],
            specificity_floor  = tgt['specificity_floor'],
            use_fbeta          = tgt['use_fbeta'],
            beta               = tgt['beta'],
        )
        thresholds[cls_name] = theta

    thresh_preds, thresh_bacc = apply_clinical_thresholds(
        probs          = tta_sum,
        labels         = tta_labels,
        thresholds     = thresholds,
        classes_used   = classes_used,
        priority_order = cfg['clinical']['priority_order'],
    )
    metrics_df = compute_per_class_metrics(thresh_preds, tta_labels, classes_used)
    log(f"\n===== FINAL RESULTS (test) =====")
    log(f"  TTA Ensemble (argmax):    BACC = {tta_bacc:.4f}")
    log(f"  TTA + clinical thresholds: BACC = {thresh_bacc:.4f}")
    for _, row in metrics_df.iterrows():
        flag = " ← target" if row['Class'] in thresholds else ""
        log(f"    {row['Class']:6s}  sens={row['Sensitivity']:.4f}  spec={row['Specificity']:.4f}{flag}")

    # ── Save outputs ──────────────────────────────────────────────────────
    np.save(os.path.join(out_dir, 'tta_sum_probs.npy'),  tta_sum)
    np.save(os.path.join(out_dir, 'tta_labels.npy'),     tta_labels)
    np.save(os.path.join(out_dir, 'tta_val_sum.npy'),    tta_val_sum)
    np.save(os.path.join(out_dir, 'tta_val_labels.npy'), tta_val_labels)

    with open(os.path.join(out_dir, 'calibrated_thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=2)
    with open(os.path.join(out_dir, 'classes_used.json'), 'w') as f:
        json.dump(classes_used, f)

    results = {
        "run":          {"seed": seed, "timestamp": ts},
        "ensemble":     {"test_bacc": round(ens_bacc, 4)},
        "tta_ensemble": {"rounds": tta_rounds, "test_bacc": round(tta_bacc, 4)},
        "clinical": {
            "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
            "test_bacc":  round(thresh_bacc, 4),
            "per_class": {
                row["Class"]: {
                    "sensitivity": float(row["Sensitivity"]),
                    "specificity": float(row["Specificity"]),
                }
                for _, row in metrics_df.iterrows()
            },
        },
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\n✓ Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
