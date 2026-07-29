# -*- coding: utf-8 -*-
"""
msp_energy_detector.py
MSP / Energy 两个 training-free 基线（CIFAR-10, ViT-B/16）。
与 Mahalanobis / ReAct 相同评估协议：无显式阈值，仅报告 AUROC / AUPR / FPR@95TPR。
运行：python src/baselines/msp_energy/msp_energy_detector.py
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vit import ViTWrapper
from src.data.cifar_loader import get_cifar10_loaders, get_adversarial_loader

WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'weights', 'vit_b_16_cifar10.pth')
SAVE_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'metrics')
BATCH_SIZE = 256
NUM_WORKERS = 0
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def collect_logits(model, loader, device):
    """返回 (logits, labels)。你们的 loader 返回 (img, label, idx) 三元组。"""
    logits_all, labels_all = [], []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        logits_all.append(model(images).cpu())
        labels_all.append(labels)
    return torch.cat(logits_all), torch.cat(labels_all)


def msp_score(logits):
    """负的最大 softmax 概率，越大越可疑。"""
    return -F.softmax(logits, dim=1).max(dim=1).values


def energy_score(logits):
    """Energy = -logsumexp(logits)，越大越可疑（T=1，与 ReAct 原文一致）。"""
    return -torch.logsumexp(logits, dim=1)


def fpr_at_95tpr(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(fpr[int(np.argmax(tpr >= 0.95))])


def evaluate(clean_scores, adv_scores, method, attack):
    y = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(adv_scores))])
    s = np.concatenate([clean_scores.numpy(), adv_scores.numpy()])
    return {
        'Method': method, 'Attack': attack,
        'AUROC': round(roc_auc_score(y, s), 4),
        'AUPR': round(average_precision_score(y, s), 4),
        'FPR@95TPR': round(fpr_at_95tpr(y, s), 4),
        'clean_mean': round(float(clean_scores.mean()), 4),
        'adv_mean': round(float(adv_scores.mean()), 4),
    }


def main():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 模型：与 Mahalanobis / ReAct 完全相同的加载方式 ----
    model = ViTWrapper(num_classes=10, weights_path=WEIGHTS_PATH).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # ---- 干净测试集（10,000 张）----
    _, test_loader = get_cifar10_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    logits_clean, labels_clean = collect_logits(model, test_loader, device)

    # ----  Sanity check：干净准确率（必须先确认这一项！）----
    clean_acc = (logits_clean.argmax(dim=1) == labels_clean).float().mean().item()
    print(f"\n[Sanity Check] Clean test accuracy: {clean_acc * 100:.2f}%")
    if clean_acc < 0.5:
        print("[警告] 干净准确率异常低，分类头可能未训练。"
              "MSP/Energy 依赖有意义的 logits，请先确认模型与主实验一致！\n")

    results = []
    for attack_name, tag in [('fgsm', 'FGSM'), ('pgd20', 'PGD-20')]:
        adv_loader = get_adversarial_loader(attack_name=attack_name, split='test',
                                            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        logits_adv, _ = collect_logits(model, adv_loader, device)
        adv_acc = (logits_adv.argmax(dim=1) == _).float().mean().item()
        print(f"[{tag}] adv samples: {len(logits_adv)}, attack success (1-acc): {1 - adv_acc:.4f}")

        for method, fn in [('MSP', msp_score), ('Energy', energy_score)]:
            results.append(evaluate(fn(logits_clean), fn(logits_adv), method, tag))

    df = pd.DataFrame(results)
    pd.set_option('display.width', 120)
    print("\n===== MSP / Energy Baseline Results (CIFAR-10 test, threshold-free) =====")
    print(df[['Method', 'Attack', 'AUROC', 'AUPR', 'FPR@95TPR']].to_string(index=False))
    print("\n分数均值（用于核对 7.3 机理分析）：")
    print(df[['Method', 'Attack', 'clean_mean', 'adv_mean']].to_string(index=False))

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_csv = os.path.join(SAVE_DIR, 'baseline_msp_energy_cifar10.csv')
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n[Saved] {out_csv}")


if __name__ == '__main__':
    main()