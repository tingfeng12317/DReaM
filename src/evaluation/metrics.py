import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.stats import wilcoxon
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def compute_auroc(clean_scores, adv_scores):
    y_true = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(adv_scores))
    ])
    y_score = np.concatenate([clean_scores, adv_scores])
    return roc_auc_score(y_true, y_score)


def compute_aupr(clean_scores, adv_scores):
    y_true = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(adv_scores))
    ])
    y_score = np.concatenate([clean_scores, adv_scores])
    return average_precision_score(y_true, y_score)


def compute_fpr_at_tpr(clean_scores, adv_scores, target_tpr=0.95):
    y_true = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(adv_scores))
    ])
    y_score = np.concatenate([clean_scores, adv_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(tpr - target_tpr))
    return fpr[idx], thresholds[idx]


def compute_roc_data(clean_scores, adv_scores):
    y_true = np.concatenate([
        np.zeros(len(clean_scores)),
        np.ones(len(adv_scores))
    ])
    y_score = np.concatenate([clean_scores, adv_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }


def compute_fpr_tpr(clean_scores, adv_scores, target_fpr=None):
    """
    统一阈值：从 clean_scores 中找 percentile，使得 FPR ≈ target_fpr
    由于离散性，实际 FPR 可能不完全等于 target_fpr，返回实际值
    """
    threshold = np.percentile(clean_scores, (1 - target_fpr) * 100)
    actual_fpr = np.mean(clean_scores > threshold)
    tpr = np.mean(adv_scores > threshold)
    return float(actual_fpr), float(tpr), float(threshold)


def wilcoxon_test(clean_scores, adv_scores):
    statistic, p_value = wilcoxon(clean_scores, adv_scores)
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }


def summarize_metrics(clean_scores, adv_scores, save_path=None, target_fpr=None):
    """
    target_fpr: 如果为 None，不计算固定 FPR 下的 TPR（适用于无显式阈值的方法）
                如果传入具体值（如 0.04），才计算并保存 FPR/TPR
    """
    auroc = compute_auroc(clean_scores, adv_scores)
    aupr = compute_aupr(clean_scores, adv_scores)
    fpr95, threshold95 = compute_fpr_at_tpr(clean_scores, adv_scores, target_tpr=0.95)
    roc_data = compute_roc_data(clean_scores, adv_scores)
    wilcoxon_result = wilcoxon_test(clean_scores, adv_scores)

    summary = {
        'AUROC': float(auroc),
        'AUPR': float(aupr),
        'FPR@95TPR': float(fpr95),
        'threshold@95TPR': float(threshold95),
        'Wilcoxon_p_value': wilcoxon_result['p_value'],
        'Wilcoxon_significant': wilcoxon_result['significant'],
        'num_clean': len(clean_scores),
        'num_adversarial': len(adv_scores),
        'roc_data': roc_data
    }

    # 只有显式传入 target_fpr 时才计算固定操作点下的 FPR/TPR
    if target_fpr is not None:
        actual_fpr, tpr_at_fpr, threshold = compute_fpr_tpr(clean_scores, adv_scores, target_fpr=target_fpr)
        summary['FPR'] = actual_fpr
        summary['TPR'] = tpr_at_fpr
        summary['threshold'] = threshold
        summary['target_FPR'] = target_fpr
        print(f"FPR (actual at ~{target_fpr*100:.0f}% target): {actual_fpr:.4f}")
        print(f"TPR (actual):   {tpr_at_fpr:.4f}")

    summary['clean_scores'] = clean_scores.tolist()
    summary['adv_scores'] = adv_scores.tolist()

    print(f"\n{'='*60}")
    print("METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"AUROC:          {auroc:.4f}")
    print(f"AUPR:           {aupr:.4f}")
    print(f"FPR@95%TPR:     {fpr95:.4f}")
    print(f"Wilcoxon p:     {wilcoxon_result['p_value']:.2e} ({'significant' if wilcoxon_result['significant'] else 'not significant'})")
    print(f"{'='*60}")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to: {save_path}")

    return summary


if __name__ == '__main__':
    np.random.seed(42)
    clean_scores = np.random.normal(5, 2, 1000)
    adv_scores = np.random.normal(15, 3, 1000)

    summary = summarize_metrics(
        clean_scores, adv_scores,
        save_path=os.path.join(PROJECT_ROOT, 'outputs', 'metrics', 'test_metrics.json')
    )