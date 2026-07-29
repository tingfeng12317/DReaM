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


def evaluate_detector(detector, val_loader, fgsm_loader, pgd_loader, device):
    clean_preds, clean_scores = [], []
    for images, _, _ in tqdm(val_loader, desc="Clean", leave=False):
        images = images.to(device)
        is_adv, W, _ = detector.detect(images, early_stop=True)
        clean_preds.extend(is_adv.cpu().numpy().tolist())
        clean_scores.extend(W.cpu().numpy().tolist())
    fpr = np.mean(np.array(clean_preds) == 1)

    fgsm_preds, fgsm_scores = [], []
    for images, _, _ in tqdm(fgsm_loader, desc="FGSM", leave=False):
        images = images.to(device)
        is_adv, W, _ = detector.detect(images, early_stop=True)
        fgsm_preds.extend(is_adv.cpu().numpy().tolist())
        fgsm_scores.extend(W.cpu().numpy().tolist())
    fgsm_tpr = np.mean(np.array(fgsm_preds) == 1)

    pgd_preds, pgd_scores = [], []
    for images, _, _ in tqdm(pgd_loader, desc="PGD20", leave=False):
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


def run_ablation(cfg):
    dataset_name = cfg.dataset_name
    num_classes = cfg.num_classes
    K = cfg.get('dream.K')
    device = cfg.device
    weights_path = cfg.weights_path
    batch_size = cfg.get('training_free.batch_size_detection')

    dist_path = cfg.paths['distribution']
    topk_path = cfg.paths['topk']

    print("=" * 60)
    print(f"ABLATION STUDIES (Validation Set) - {dataset_name.upper()}")
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
    # Ablation 1: a sensitivity
    # ============================================================
    print(f"\n{'='*60}")
    print("ABLATION 1: a Sensitivity (K=3, lambda_b=4.0, lambda_w=0.3)")
    print(f"{'='*60}")
    sys.stdout.flush()
    a_values = [1.0, 1.2, 1.4]
    for a in a_values:
        print(f"\n--- a = {a} ---")
        sys.stdout.flush()
        detector = DReaMDetector(model, dist_path, topk_path, a=a, lambda_weight=0.3, lambda_b=4.0, K=K, device=device)
        detector.calibrate_B_max(val_loader)
        metrics = evaluate_detector(detector, val_loader, fgsm_val, pgd20_val, device)
        metrics['ablation_type'] = 'a'
        metrics['param_value'] = str(a)
        all_results.append(metrics)
        print(f"  FPR={metrics['fpr']:.4f}, FGSM_TPR={metrics['fgsm_tpr']:.4f}, FGSM_AUROC={metrics['fgsm_auroc']:.4f}")
        print(f"  PGD_TPR={metrics['pgd_tpr']:.4f}, PGD_AUROC={metrics['pgd_auroc']:.4f}")
        sys.stdout.flush()

    # ============================================================
    # Ablation 2: lambda_weight sensitivity
    # ============================================================
    print(f"\n{'='*60}")
    print("ABLATION 2: lambda_weight Sensitivity (K=3, a=1.0, lambda_b=4.0)")
    print(f"{'='*60}")
    sys.stdout.flush()
    lw_values = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    for lw in lw_values:
        print(f"\n--- lambda_weight = {lw} ---")
        sys.stdout.flush()
        detector = DReaMDetector(model, dist_path, topk_path, a=1.0, lambda_weight=lw, lambda_b=4.0, K=K, device=device)
        detector.calibrate_B_max(val_loader)
        metrics = evaluate_detector(detector, val_loader, fgsm_val, pgd20_val, device)
        metrics['ablation_type'] = 'lambda_w'
        metrics['param_value'] = str(lw)
        all_results.append(metrics)
        print(f"  FPR={metrics['fpr']:.4f}, FGSM_TPR={metrics['fgsm_tpr']:.4f}, FGSM_AUROC={metrics['fgsm_auroc']:.4f}")
        print(f"  PGD_TPR={metrics['pgd_tpr']:.4f}, PGD_AUROC={metrics['pgd_auroc']:.4f}")
        sys.stdout.flush()

    # ============================================================
    # Ablation 3: Component ablation
    # ============================================================
    print(f"\n{'='*60}")
    print("ABLATION 3: Component Ablation (K=3, a=1.0, lambda_b=4.0)")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Residual only
    print(f"\n--- Residual Only (lambda_w=1.0) ---")
    sys.stdout.flush()
    detector = DReaMDetector(model, dist_path, topk_path, a=1.0, lambda_weight=1.0, lambda_b=4.0, K=K, device=device)
    detector.calibrate_B_max(val_loader)
    metrics = evaluate_detector(detector, val_loader, fgsm_val, pgd20_val, device)
    metrics['ablation_type'] = 'component'
    metrics['param_value'] = 'residual_only'
    all_results.append(metrics)
    print(f"  FPR={metrics['fpr']:.4f}, FGSM_TPR={metrics['fgsm_tpr']:.4f}, FGSM_AUROC={metrics['fgsm_auroc']:.4f}")
    print(f"  PGD_TPR={metrics['pgd_tpr']:.4f}, PGD_AUROC={metrics['pgd_auroc']:.4f}")
    sys.stdout.flush()

    # Activation only
    print(f"\n--- Activation Only (lambda_w=0.0) ---")
    sys.stdout.flush()
    detector = DReaMDetector(model, dist_path, topk_path, a=1.0, lambda_weight=0.0, lambda_b=4.0, K=K, device=device)
    detector.calibrate_B_max(val_loader)
    metrics = evaluate_detector(detector, val_loader, fgsm_val, pgd20_val, device)
    metrics['ablation_type'] = 'component'
    metrics['param_value'] = 'activation_only'
    all_results.append(metrics)
    print(f"  FPR={metrics['fpr']:.4f}, FGSM_TPR={metrics['fgsm_tpr']:.4f}, FGSM_AUROC={metrics['fgsm_auroc']:.4f}")
    print(f"  PGD_TPR={metrics['pgd_tpr']:.4f}, PGD_AUROC={metrics['pgd_auroc']:.4f}")
    sys.stdout.flush()

    # Save to dataset-specific output dir
    save_dir = os.path.join(cfg.dataset_output_dir, 'ablation')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'ablation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    keys = list(all_results[0].keys())
    with open(os.path.join(save_dir, 'ablation_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_results)

    print(f"\n{'='*60}")
    print(f"Ablation complete! Saved to: {save_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='DReaM ablation studies')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_ablation(cfg)

    print(f"\n{'='*60}")
    print("Ablation studies complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()