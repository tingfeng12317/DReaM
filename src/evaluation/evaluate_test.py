import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vit import ViTWrapper
from src.data.dataset_registry import get_loaders, get_adversarial_loader
from src.dream.detector import DReaMDetector
from src.evaluation.metrics import summarize_metrics
from src.utils.config import load_config


def evaluate_test_set(cfg):
    dataset_name = cfg.dataset_name
    num_classes = cfg.num_classes
    K = cfg.get('dream.K')
    a = cfg.get('dream.a')
    lambda_b = cfg.get('dream.lambda_b')
    lambda_w = cfg.get('dream.lambda_w')
    B_max = cfg.get('dream.B_max')
    batch_size = cfg.get('training_free.batch_size_detection')
    num_workers = cfg.get('training_free.num_workers')
    device = cfg.device
    weights_path = cfg.weights_path

    if B_max is None:
        raise RuntimeError(
            f"B_max not found in config. "
            f"Please run tune_params first: "
            f"python src/evaluation/tune_params.py --config configs/{dataset_name}.yaml"
        )

    print("Loading ViT-B/16...")
    if weights_path and os.path.exists(weights_path):
        model = ViTWrapper(num_classes=num_classes, weights_path=weights_path).to(device)
        print(f"Loaded weights from: {weights_path}")
    else:
        model = ViTWrapper(num_classes=num_classes, weights='imagenet').to(device)

    dist_path = cfg.paths['distribution']
    topk_path = cfg.paths['topk']

    detector = DReaMDetector(
        model=model,
        distributions_path=dist_path,
        topk_path=topk_path,
        a=a,
        lambda_weight=lambda_w,
        lambda_b=lambda_b,
        K=K,
        device=device
    )

    # LOCKED: use validation-calibrated B_max, do NOT recompute
    detector.B_max = B_max
    print(f"\n{'='*60}")
    print(f"LOCKED PARAMETERS (from validation tuning):")
    print(f"  a = {a}, lambda_b = {lambda_b}, lambda_weight = {lambda_w}")
    print(f"  B_max = {B_max:.4f}")
    print(f"{'='*60}")

    test_loader = get_loaders(dataset_name, split='test', batch_size=batch_size, num_workers=num_workers)
    fgsm_test_loader = get_adversarial_loader(dataset_name, 'fgsm', 'test', batch_size, num_workers)
    pgd20_test_loader = get_adversarial_loader(dataset_name, 'pgd20', 'test', batch_size, num_workers)

    # 1. Clean test FPR
    print(f"\n{'='*60}")
    print(f"Evaluating on CLEAN test set ({dataset_name.upper()})...")
    print(f"{'='*60}")
    clean_result = detector.evaluate(test_loader, is_adversarial=False)

    # 2. FGSM test TPR
    print(f"\n{'='*60}")
    print(f"Evaluating on FGSM test set ({dataset_name.upper()})...")
    print(f"{'='*60}")
    fgsm_result = detector.evaluate(fgsm_test_loader, is_adversarial=True)

    # 3. PGD-20 test TPR
    print(f"\n{'='*60}")
    print(f"Evaluating on PGD-20 test set ({dataset_name.upper()})...")
    print(f"{'='*60}")
    pgd20_result = detector.evaluate(pgd20_test_loader, is_adversarial=True)

    # 4. Metrics
    print(f"\n{'='*60}")
    print("Computing FGSM test metrics...")
    print(f"{'='*60}")
    fgsm_summary = summarize_metrics(
        clean_scores=clean_result['scores'],
        adv_scores=fgsm_result['scores'],
        save_path=cfg.metrics_path('test_fgsm_tuned.json')
    )

    print(f"\n{'='*60}")
    print("Computing PGD-20 test metrics...")
    print(f"{'='*60}")
    pgd20_summary = summarize_metrics(
        clean_scores=clean_result['scores'],
        adv_scores=pgd20_result['scores'],
        save_path=cfg.metrics_path('test_pgd20_tuned.json')
    )

    # 5. Final report
    print(f"\n{'='*60}")
    print(f"TEST SET FINAL REPORT ({dataset_name.upper()}, Tuned Parameters)")
    print(f"{'='*60}")
    print(f"Parameters: a={a}, lambda_b={lambda_b}, lambda_weight={lambda_w}, B_max={B_max:.4f}")
    print(f"{'Attack':<12} {'TPR':>8} {'FPR':>8} {'AUROC':>8} {'AUPR':>8} {'FPR@95TPR':>10}")
    print("-" * 60)
    print(f"{'FGSM':<12} {fgsm_result['tpr']:>8.4f} {clean_result['fpr']:>8.4f} "
          f"{fgsm_summary['AUROC']:>8.4f} {fgsm_summary['AUPR']:>8.4f} {fgsm_summary['FPR@95TPR']:>10.4f}")
    print(f"{'PGD-20':<12} {pgd20_result['tpr']:>8.4f} {clean_result['fpr']:>8.4f} "
          f"{pgd20_summary['AUROC']:>8.4f} {pgd20_summary['AUPR']:>8.4f} {pgd20_summary['FPR@95TPR']:>10.4f}")
    print(f"{'='*60}")

    report = {
        'params': {'a': a, 'lambda_b': lambda_b, 'lambda_weight': lambda_w, 'B_max': float(B_max)},
        'clean_test_fpr': float(clean_result['fpr']),
        'fgsm_test': {
            'tpr': float(fgsm_result['tpr']),
            'auroc': float(fgsm_summary['AUROC']),
            'aupr': float(fgsm_summary['AUPR']),
            'fpr95': float(fgsm_summary['FPR@95TPR'])
        },
        'pgd20_test': {
            'tpr': float(pgd20_result['tpr']),
            'auroc': float(pgd20_summary['AUROC']),
            'aupr': float(pgd20_summary['AUPR']),
            'fpr95': float(pgd20_summary['FPR@95TPR'])
        }
    }
    report_path = cfg.metrics_path('test_final_tuned_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved final report to: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description='DReaM test set evaluation with locked parameters')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"Configuration: {cfg.dataset_name}")
    print(f"  num_classes: {cfg.num_classes}")
    print(f"  K: {cfg.get('dream.K')}")
    print(f"  a: {cfg.get('dream.a')}")
    print(f"  lambda_b: {cfg.get('dream.lambda_b')}")
    print(f"  lambda_w: {cfg.get('dream.lambda_w')}")
    print(f"  B_max: {cfg.get('dream.B_max')}")
    print(f"  device: {cfg.device}")
    print(f"{'='*60}")

    report = evaluate_test_set(cfg)

    print(f"\n{'='*60}")
    print("Test evaluation complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
