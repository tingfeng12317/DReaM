import os
import sys
import json
import csv
import argparse
import torch
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vit import ViTWrapper
from src.data.dataset_registry import get_loaders, get_adversarial_loader
from src.dream.detector import DReaMDetector
from src.evaluation.metrics import compute_auroc, compute_aupr, compute_fpr_at_tpr
from src.utils.config import load_config


def evaluate_detector(detector, val_loader, fgsm_loader, pgd_loader, device, desc=""):
    clean_preds, clean_scores = [], []
    for images, _, _ in tqdm(val_loader, desc=f"{desc} Clean", leave=False):
        images = images.to(device)
        is_adv, W, _ = detector.detect(images, early_stop=True)
        clean_preds.extend(is_adv.cpu().numpy().tolist())
        clean_scores.extend(W.cpu().numpy().tolist())
    fpr = np.mean(np.array(clean_preds) == 1)

    fgsm_preds, fgsm_scores = [], []
    for images, _, _ in tqdm(fgsm_loader, desc=f"{desc} FGSM", leave=False):
        images = images.to(device)
        is_adv, W, _ = detector.detect(images, early_stop=True)
        fgsm_preds.extend(is_adv.cpu().numpy().tolist())
        fgsm_scores.extend(W.cpu().numpy().tolist())
    fgsm_tpr = np.mean(np.array(fgsm_preds) == 1)

    pgd_preds, pgd_scores = [], []
    for images, _, _ in tqdm(pgd_loader, desc=f"{desc} PGD20", leave=False):
        images = images.to(device)
        is_adv, W, _ = detector.detect(images, early_stop=True)
        pgd_preds.extend(is_adv.cpu().numpy().tolist())
        pgd_scores.extend(W.cpu().numpy().tolist())
    pgd_tpr = np.mean(np.array(pgd_preds) == 1)

    clean_scores = np.array(clean_scores)
    fgsm_scores = np.array(fgsm_scores)
    pgd_scores = np.array(pgd_scores)

    fgsm_auroc = compute_auroc(clean_scores, fgsm_scores)
    fgsm_fpr95, _ = compute_fpr_at_tpr(clean_scores, fgsm_scores, 0.95)
    pgd_auroc = compute_auroc(clean_scores, pgd_scores)
    pgd_fpr95, _ = compute_fpr_at_tpr(clean_scores, pgd_scores, 0.95)

    return {
        'fpr': float(fpr),
        'fgsm_tpr': float(fgsm_tpr),
        'fgsm_auroc': float(fgsm_auroc),
        'fgsm_fpr95': float(fgsm_fpr95),
        'pgd_tpr': float(pgd_tpr),
        'pgd_auroc': float(pgd_auroc),
        'pgd_fpr95': float(pgd_fpr95),
        'B_max': float(detector.B_max)
    }


def tune_parameters(cfg):
    dataset_name = cfg.dataset_name
    num_classes = cfg.num_classes
    K = cfg.get('dream.K')
    device = cfg.device
    weights_path = cfg.weights_path
    batch_size = cfg.get('training_free.batch_size_detection')

    dist_path = cfg.paths['distribution']
    topk_path = cfg.paths['topk']

    print("=" * 60)
    print(f"STAGED PARAMETER TUNING (Validation Set Only)")
    print(f"Dataset: {dataset_name}")
    print("=" * 60)
    sys.stdout.flush()

    print("\nLoading model and data...")
    sys.stdout.flush()
    if weights_path and os.path.exists(weights_path):
        model = ViTWrapper(num_classes=num_classes, weights_path=weights_path).to(device)
    else:
        model = ViTWrapper(num_classes=num_classes, weights='imagenet').to(device)

    val_loader = get_loaders(dataset_name, split='val', batch_size=batch_size, num_workers=0)
    fgsm_val = get_adversarial_loader(dataset_name, 'fgsm', 'val', batch_size, 0)
    pgd20_val = get_adversarial_loader(dataset_name, 'pgd20', 'val', batch_size, 0)

    all_results = []

    # ============================================================
    # STAGE 1: Tune a
    # ============================================================
    a_candidates = cfg.get('tuning.stage1_a')
    print(f"\n{'='*60}")
    print(f"STAGE 1: Tuning a")
    print(f"Search space: {a_candidates}")
    print(f"{'='*60}")
    sys.stdout.flush()

    stage1_results = []
    for a in a_candidates:
        print(f"\n--- a = {a} ---")
        sys.stdout.flush()
        detector = DReaMDetector(model, dist_path, topk_path, a=a, lambda_weight=0.5, lambda_b=3.0, K=K, device=device)
        print("Calibrating B_max...")
        sys.stdout.flush()
        detector.calibrate_B_max(val_loader)
        print("Evaluating...")
        sys.stdout.flush()
        metrics = evaluate_detector(detector, val_loader, fgsm_val, pgd20_val, device, desc=f"a={a}")
        metrics['stage'] = 1
        metrics['a'] = a
        metrics['lambda_b'] = 3.0
        metrics['lambda_weight'] = 0.5
        stage1_results.append(metrics)
        all_results.append(metrics)
        print(f"  FPR={metrics['fpr']:.4f}, FGSM_TPR={metrics['fgsm_tpr']:.4f}, FGSM_AUROC={metrics['fgsm_auroc']:.4f}")
        print(f"  PGD_TPR={metrics['pgd_tpr']:.4f}, PGD_AUROC={metrics['pgd_auroc']:.4f}")
        sys.stdout.flush()

    best_a = max(stage1_results, key=lambda x: (x['fgsm_auroc'] + x['pgd_auroc']) / 2.0)['a']
    print(f"\n>>> Best a = {best_a} (by average AUROC)")
    sys.stdout.flush()

    # ============================================================
    # STAGE 2: Tune lambda_b
    # ============================================================
    lb_candidates = cfg.get('tuning.stage2_lambda_b')
    print(f"\n{'='*60}")
    print(f"STAGE 2: Tuning lambda_b (a={best_a}, lambda_weight=0.5)")
    print(f"Search space: {lb_candidates}")
    print(f"{'='*60}")
    sys.stdout.flush()

    stage2_results = []
    for lb in lb_candidates:
        print(f"\n--- lambda_b = {lb} ---")
        sys.stdout.flush()
        detector = DReaMDetector(model, dist_path, topk_path, a=best_a, lambda_weight=0.5, lambda_b=lb, K=K, device=device)
        print("Calibrating B_max...")
        sys.stdout.flush()
        detector.calibrate_B_max(val_loader)
        print("Evaluating...")
        sys.stdout.flush()
        metrics = evaluate_detector(detector, val_loader, fgsm_val, pgd20_val, device, desc=f"lb={lb}")
        metrics['stage'] = 2
        metrics['a'] = best_a
        metrics['lambda_b'] = lb
        metrics['lambda_weight'] = 0.5
        stage2_results.append(metrics)
        all_results.append(metrics)
        print(f"  FPR={metrics['fpr']:.4f}, FGSM_TPR={metrics['fgsm_tpr']:.4f}, FGSM_FPR95={metrics['fgsm_fpr95']:.4f}")
        print(f"  PGD_TPR={metrics['pgd_tpr']:.4f}, PGD_FPR95={metrics['pgd_fpr95']:.4f}")
        sys.stdout.flush()

    valid = [r for r in stage2_results if r['fpr'] <= 0.05]
    if not valid:
        valid = stage2_results
    best_lb = min(valid, key=lambda x: (x['fgsm_fpr95'] + x['pgd_fpr95']) / 2.0)['lambda_b']
    print(f"\n>>> Best lambda_b = {best_lb} (FPR<5% preferred, lowest avg FPR@95%TPR)")
    sys.stdout.flush()

    # ============================================================
    # STAGE 3: Tune lambda_w
    # ============================================================
    lw_candidates = cfg.get('tuning.stage3_lambda_w')
    print(f"\n{'='*60}")
    print(f"STAGE 3: Tuning lambda_weight (a={best_a}, lambda_b={best_lb})")
    print(f"Search space: {lw_candidates}")
    print(f"{'='*60}")
    sys.stdout.flush()

    stage3_results = []
    for lw in lw_candidates:
        print(f"\n--- lambda_weight = {lw} ---")
        sys.stdout.flush()
        detector = DReaMDetector(model, dist_path, topk_path, a=best_a, lambda_weight=lw, lambda_b=best_lb, K=K, device=device)
        print("Calibrating B_max...")
        sys.stdout.flush()
        detector.calibrate_B_max(val_loader)
        print("Evaluating...")
        sys.stdout.flush()
        metrics = evaluate_detector(detector, val_loader, fgsm_val, pgd20_val, device, desc=f"lw={lw}")
        metrics['stage'] = 3
        metrics['a'] = best_a
        metrics['lambda_b'] = best_lb
        metrics['lambda_weight'] = lw
        stage3_results.append(metrics)
        all_results.append(metrics)
        print(f"  FPR={metrics['fpr']:.4f}, FGSM_TPR={metrics['fgsm_tpr']:.4f}, FGSM_FPR95={metrics['fgsm_fpr95']:.4f}")
        print(f"  PGD_TPR={metrics['pgd_tpr']:.4f}, PGD_FPR95={metrics['pgd_fpr95']:.4f}")
        sys.stdout.flush()

    best_lw = min(stage3_results, key=lambda x: (x['fgsm_fpr95'] + x['pgd_fpr95']) / 2.0)['lambda_weight']
    print(f"\n>>> Best lambda_weight = {best_lw} (lowest avg FPR@95%TPR)")
    sys.stdout.flush()

    # ============================================================
    # Final Summary & Auto-write B_max back to config
    # ============================================================
    print(f"\n{'='*60}")
    print("FINAL RECOMMENDED PARAMETERS")
    print(f"{'='*60}")
    print(f"  a*            = {best_a}")
    print(f"  lambda_b*     = {best_lb}")
    print(f"  lambda_weight*= {best_lw}")

    # Re-run with best params to get final B_max
    detector = DReaMDetector(model, dist_path, topk_path, a=best_a, lambda_weight=best_lw, lambda_b=best_lb, K=K, device=device)
    detector.calibrate_B_max(val_loader)
    best_B_max = detector.B_max
    print(f"  B_max         = {best_B_max:.4f}")
    sys.stdout.flush()

    # Write back to config
    cfg.set('dream.a', best_a)
    cfg.set('dream.lambda_b', best_lb)
    cfg.set('dream.lambda_w', best_lw)
    cfg.set('dream.B_max', float(best_B_max))
    cfg.save()
    print(f"\n>>> Parameters written back to {cfg._yaml_path}")
    sys.stdout.flush()

    # Save results
    save_dir = cfg.paths['grid_search']
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'staged_tuning_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    keys = list(all_results[0].keys())
    with open(os.path.join(save_dir, 'staged_tuning_results.csv'), 'w', newline='') as f:
        import csv
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_results)

    print(f"\nSaved results to: {save_dir}")
    return best_a, best_lb, best_lw, best_B_max


def main():
    parser = argparse.ArgumentParser(description='DReaM staged hyperparameter tuning')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = cfg.device
    tune_parameters(cfg)

    print(f"\n{'='*60}")
    print("Tuning complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()