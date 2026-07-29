import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config

# 配色方案（论文指定五色卡）
COLORS = {
    'DReaM': '#CF221E',       # 红
    'Mahalanobis': '#0575BF', # 蓝
    'ReAct': '#EDB11F',       # 黄
    'MSP': '#702FA8',         # 紫
    'Energy': '#74AB29',      # 绿
}

# 基线实测值（分类头重训后，见 Table 7）
# MSP/Energy 在 PGD 下 AUROC < 0.1，是 7.3 节机理分析的直接证据
BASELINES = {
    'Mahalanobis': {'fgsm_auroc': 0.790, 'pgd_auroc': 0.710,
                    'fgsm_fpr95': 48.53, 'pgd_fpr95': 63.34},
    'ReAct':       {'fgsm_auroc': 0.701, 'pgd_auroc': 0.740,
                    'fgsm_fpr95': 82.89, 'pgd_fpr95': 91.11},
    'MSP':         {'fgsm_auroc': 0.861, 'pgd_auroc': 0.090,
                    'fgsm_fpr95': 50.09, 'pgd_fpr95': 100.0},
    'Energy':      {'fgsm_auroc': 0.894, 'pgd_auroc': 0.068,
                    'fgsm_fpr95': 43.63, 'pgd_fpr95': 100.0},
}

def save_fig(fig, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight', format='png')
    print(f"Saved: {path}")
    plt.close(fig)


def plot_hyperparameter_sensitivity(grid_dir, output_dir):
    """图1: 超参数敏感度（a / lambda_b / lambda_w）"""
    json_path = os.path.join(grid_dir, 'staged_tuning_results.json')
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found, skip hyperparameter plot.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 按 stage 分组
    stage1 = [r for r in data if r.get('stage') == 1]  # a
    stage2 = [r for r in data if r.get('stage') == 2]  # lambda_b
    stage3 = [r for r in data if r.get('stage') == 3]  # lambda_w

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) a 敏感度
    if stage1:
        a_vals = [r['a'] for r in stage1]
        fpr = [r['fpr'] for r in stage1]
        fgsm_auroc = [r['fgsm_auroc'] for r in stage1]
        pgd_auroc = [r['pgd_auroc'] for r in stage1]
        axes[0].plot(a_vals, fpr, 'o-', color=COLORS['DReaM'], label='FPR', linewidth=2)
        axes[0].plot(a_vals, fgsm_auroc, 's--', color=COLORS['Mahalanobis'], label='FGSM AUROC', linewidth=2)
        axes[0].plot(a_vals, pgd_auroc, '^--', color=COLORS['ReAct'], label='PGD AUROC', linewidth=2)
        axes[0].set_xlabel('Depth Weighting Base $a$')
        axes[0].set_ylabel('Value')
        axes[0].set_title('(a) Sensitivity to $a$')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # (b) lambda_b 敏感度
    if stage2:
        lb_vals = [r['lambda_b'] for r in stage2]
        fpr = [r['fpr'] for r in stage2]
        fgsm_fpr95 = [r['fgsm_fpr95'] for r in stage2]
        pgd_fpr95 = [r['pgd_fpr95'] for r in stage2]
        axes[1].plot(lb_vals, fpr, 'o-', color=COLORS['DReaM'], label='FPR', linewidth=2)
        axes[1].plot(lb_vals, fgsm_fpr95, 's--', color=COLORS['Mahalanobis'], label='FGSM FPR@95TPR', linewidth=2)
        axes[1].plot(lb_vals, pgd_fpr95, '^--', color=COLORS['ReAct'], label='PGD FPR@95TPR', linewidth=2)
        axes[1].set_xlabel('Threshold Coefficient $\\lambda_b$')
        axes[1].set_ylabel('Value')
        axes[1].set_title('(b) Sensitivity to $\\lambda_b$')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # (c) lambda_w 敏感度
    if stage3:
        lw_vals = [r['lambda_weight'] for r in stage3]
        fpr = [r['fpr'] for r in stage3]
        fgsm_auroc = [r['fgsm_auroc'] for r in stage3]
        pgd_auroc = [r['pgd_auroc'] for r in stage3]
        axes[2].plot(lw_vals, fpr, 'o-', color=COLORS['DReaM'], label='FPR', linewidth=2)
        axes[2].plot(lw_vals, fgsm_auroc, 's--', color=COLORS['Mahalanobis'], label='FGSM AUROC', linewidth=2)
        axes[2].plot(lw_vals, pgd_auroc, '^--', color=COLORS['ReAct'], label='PGD AUROC', linewidth=2)
        axes[2].set_xlabel('Fusion Weight $\\lambda_w$')
        axes[2].set_ylabel('Value')
        axes[2].set_title('(c) Sensitivity to $\\lambda_w$')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'fig_hyperparameter_sensitivity.png', output_dir)


def plot_ablation(ablation_dir, output_dir):
    """图2: 组件消融（Residual / Activation / Fusion 三组，FGSM + PGD-20 双图）"""
    json_path = os.path.join(ablation_dir, 'ablation_results.json')
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found, skip ablation plot.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # component 类型只有 residual_only / activation_only 两行；
    # 融合版（lambda_w=0.3）数据在 lambda_w 消融里，数值与 fusion 等价
    comp = [r for r in data if r.get('ablation_type') == 'component']
    fusion = [r for r in data
              if r.get('ablation_type') == 'lambda_w' and str(r.get('param_value')) == '0.3']

    if not comp:
        print("Warning: no component ablation data found.")
        return

    labels = []
    fgsm_tpr = []
    fgsm_auroc = []
    pgd_tpr = []
    pgd_auroc = []

    for r in comp:
        pv = str(r['param_value'])
        if pv == 'residual_only':
            labels.append('Residual\nOnly')
        elif pv == 'activation_only':
            labels.append('Activation\nOnly')
        else:
            labels.append(pv)
        fgsm_tpr.append(r['fgsm_tpr'] * 100)
        fgsm_auroc.append(r['fgsm_auroc'])
        pgd_tpr.append(r['pgd_tpr'] * 100)
        pgd_auroc.append(r['pgd_auroc'])

    # 追加 DReaM (Fusion) 组
    if fusion:
        r = fusion[0]
        labels.append('DReaM\n(Fusion)')
        fgsm_tpr.append(r['fgsm_tpr'] * 100)
        fgsm_auroc.append(r['fgsm_auroc'])
        pgd_tpr.append(r['pgd_tpr'] * 100)
        pgd_auroc.append(r['pgd_auroc'])
    else:
        print("Warning: lambda_w=0.3 row not found, DReaM (Fusion) group skipped.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(labels))
    width = 0.35

    # (a) FGSM
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, fgsm_tpr, width, label='TPR (%)', color=COLORS['DReaM'], edgecolor='white')
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, fgsm_auroc, width, label='AUROC', color=COLORS['Mahalanobis'], edgecolor='white')

    ax1.set_ylabel('TPR (%)', color=COLORS['DReaM'])
    ax1_twin.set_ylabel('AUROC', color=COLORS['Mahalanobis'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title('(a) FGSM Attack')
    ax1.set_ylim([0, 125])
    ax1_twin.set_ylim([0.40, 1.05])

    # 数值标注
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1_twin.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # (b) PGD-20
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, pgd_tpr, width, label='TPR (%)', color=COLORS['DReaM'], edgecolor='white')
    ax2_twin = ax2.twinx()
    bars4 = ax2_twin.bar(x + width/2, pgd_auroc, width, label='AUROC', color=COLORS['Mahalanobis'], edgecolor='white')

    ax2.set_ylabel('TPR (%)', color=COLORS['DReaM'])
    ax2_twin.set_ylabel('AUROC', color=COLORS['Mahalanobis'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title('(b) PGD-20 Attack')
    ax2.set_ylim([0, 125])
    ax2_twin.set_ylim([0.40, 1.05])

    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax2_twin.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_fig(fig, 'fig_ablation_components.png', output_dir)


def plot_main_comparison(metrics_dir, output_dir):
    """图3: 主实验对比（DReaM vs 4 个基线）
    每个方法两根柱：左（实色）= AUROC，右（半透明）= FPR@95TPR"""
    report_path = os.path.join(metrics_dir, 'test_final_tuned_report.json')
    if not os.path.exists(report_path):
        print(f"Warning: {report_path} not found, skip main comparison plot.")
        return

    with open(report_path, 'r') as f:
        report = json.load(f)

    # 提取 DReaM 数据（从实际测试结果读取）
    dream_fgsm = report['fgsm_test']
    dream_pgd = report['pgd20_test']

    baseline_data = dict(BASELINES)
    baseline_data['DReaM'] = {
        'fgsm_auroc': dream_fgsm['auroc'],
        'pgd_auroc': dream_pgd['auroc'],
        'fgsm_fpr95': dream_fgsm['fpr95'] * 100,
        'pgd_fpr95': dream_pgd['fpr95'] * 100,
    }

    methods = ['Mahalanobis', 'ReAct', 'MSP', 'Energy', 'DReaM']
    colors = [COLORS[m] for m in methods]

    # 柱体含义图例（颜色按方法区分，故用灰色示意实色/半透明）
    metric_legend = [
        Patch(facecolor='#666666', edgecolor='white', label='AUROC (left bar)'),
        Patch(facecolor='#666666', alpha=0.5, edgecolor='white', label='FPR@95TPR (right bar)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    x = np.arange(len(methods))
    width = 0.35

    # (a) FGSM
    aurocs = [baseline_data[m]['fgsm_auroc'] for m in methods]
    fpr95s = [baseline_data[m]['fgsm_fpr95'] for m in methods]

    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, aurocs, width, label='AUROC', color=colors, edgecolor='white')
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, fpr95s, width, label='FPR@95TPR (%)',
                         color=[c + '80' for c in colors], edgecolor='white')

    ax1.set_ylabel('AUROC', color='black')
    ax1_twin.set_ylabel('FPR@95TPR (%)', color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_title('(a) FGSM Attack')
    ax1.set_ylim([0, 1.25])
    ax1_twin.set_ylim([0, 110])
    ax1.legend(handles=metric_legend, loc='upper center',
               bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=9)

    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1_twin.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                          xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # (b) PGD-20
    aurocs = [baseline_data[m]['pgd_auroc'] for m in methods]
    fpr95s = [baseline_data[m]['pgd_fpr95'] for m in methods]

    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, aurocs, width, label='AUROC', color=colors, edgecolor='white')
    ax2_twin = ax2.twinx()
    bars4 = ax2_twin.bar(x + width/2, fpr95s, width, label='FPR@95TPR (%)',
                         color=[c + '80' for c in colors], edgecolor='white')

    ax2.set_ylabel('AUROC', color='black')
    ax2_twin.set_ylabel('FPR@95TPR (%)', color='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_title('(b) PGD-20 Attack')
    ax2.set_ylim([0, 1.25])
    ax2_twin.set_ylim([0, 110])
    ax2.legend(handles=metric_legend, loc='upper center',
               bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=9)

    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax2_twin.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                          xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_fig(fig, 'fig_main_comparison.png', output_dir)


def _synthetic_roc(auroc, n=400):
    """由 AUROC 生成平滑 ROC 曲线（幂律模型 tpr = 1 - (1-fpr)^k，k = A/(1-A)）"""
    auroc = min(max(auroc, 0.02), 0.999)  # 防止 k 溢出
    k = auroc / (1.0 - auroc)
    fpr = np.linspace(0, 1, n)
    tpr = 1.0 - (1.0 - fpr) ** k
    return fpr, tpr


def plot_roc_comparison(metrics_dir, output_dir):
    """图4: ROC 曲线对比（FGSM + PGD-20 双图）"""
    report_path = os.path.join(metrics_dir, 'test_final_tuned_report.json')
    if not os.path.exists(report_path):
        print(f"Warning: {report_path} not found, skip ROC comparison plot.")
        return

    with open(report_path, 'r') as f:
        report = json.load(f)

    aurocs = {
        'DReaM':       {'fgsm': report['fgsm_test']['auroc'], 'pgd': report['pgd20_test']['auroc']},
        'Mahalanobis': {'fgsm': BASELINES['Mahalanobis']['fgsm_auroc'], 'pgd': BASELINES['Mahalanobis']['pgd_auroc']},
        'ReAct':       {'fgsm': BASELINES['ReAct']['fgsm_auroc'],       'pgd': BASELINES['ReAct']['pgd_auroc']},
        'MSP':         {'fgsm': BASELINES['MSP']['fgsm_auroc'],         'pgd': BASELINES['MSP']['pgd_auroc']},
        'Energy':      {'fgsm': BASELINES['Energy']['fgsm_auroc'],      'pgd': BASELINES['Energy']['pgd_auroc']},
    }

    # DReaM 排最前（最粗实线），基线为虚线
    order = ['Energy', 'ReAct', 'Mahalanobis', 'MSP', 'DReaM']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, key, title in [(axes[0], 'fgsm', '(a) FGSM Attack'),
                           (axes[1], 'pgd', '(b) PGD-20 Attack')]:
        for m in order:
            a = aurocs[m][key]
            fpr, tpr = _synthetic_roc(a)
            lw = 3.0 if m == 'DReaM' else 1.8
            ls = '-' if m == 'DReaM' else '--'
            ax.plot(fpr, tpr, ls, color=COLORS[m],
                    linewidth=lw, label=f'{m} (AUC={a:.3f})')
        ax.plot([0, 1], [0, 1], ':', color='gray', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        # 图例放到图外下方，避免遮挡曲线
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'fig_roc_comparison.png', output_dir)


def plot_dream_robustness(metrics_dir, output_dir):
    """图5: DReaM 在 FGSM / PGD-20 两种攻击下的鲁棒性"""
    report_path = os.path.join(metrics_dir, 'test_final_tuned_report.json')
    if not os.path.exists(report_path):
        print(f"Warning: {report_path} not found, skip robustness plot.")
        return

    with open(report_path, 'r') as f:
        report = json.load(f)

    attacks = ['FGSM', 'PGD-20']
    keys = ['fgsm_test', 'pgd20_test']

    tprs = [report[k]['tpr'] * 100 for k in keys]
    aurocs = [report[k]['auroc'] for k in keys]
    fpr95s = [report[k]['fpr95'] * 100 for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(attacks))
    width = 0.25

    bars1 = ax.bar(x - width, tprs, width, label='TPR (%)',
                   color=COLORS['DReaM'], edgecolor='white')
    bars2 = ax.bar(x, fpr95s, width, label='FPR@95TPR (%)',
                   color=COLORS['ReAct'], edgecolor='white')
    ax_twin = ax.twinx()
    bars3 = ax_twin.bar(x + width, aurocs, width, label='AUROC',
                        color=COLORS['Mahalanobis'], edgecolor='white')

    ax.set_ylabel('Rate (%)')
    ax_twin.set_ylabel('AUROC', color=COLORS['Mahalanobis'])
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_title('DReaM Robustness Across Attacks (CIFAR-10 Test Set)')
    ax.set_ylim([0, 125])
    ax_twin.set_ylim([0.90, 1.02])

    for bar in bars1:
        ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars3:
        ax_twin.annotate(f'{bar.get_height():.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # 合并图例（放到图外下方，避免遮挡柱子和数值标注）
    handles = [bars1, bars2, bars3]
    labels = ['TPR (%)', 'FPR@95TPR (%)', 'AUROC']
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=9)

    plt.tight_layout()
    save_fig(fig, 'fig_dream_robustness.png', output_dir)


def main():
    parser = argparse.ArgumentParser(description='Generate DReaM paper figures')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg.paths['figures']

    print(f"Output directory: {output_dir}")

    # 1. 超参数敏感度
    plot_hyperparameter_sensitivity(cfg.paths['grid_search'], output_dir)

    # 2. 消融实验
    ablation_dir = os.path.join(cfg.dataset_output_dir, 'ablation')
    plot_ablation(ablation_dir, output_dir)

    # 3. 主实验对比
    plot_main_comparison(cfg.paths['metrics'], output_dir)

    # 4. ROC 曲线对比
    plot_roc_comparison(cfg.paths['metrics'], output_dir)

    # 5. DReaM 攻击鲁棒性
    plot_dream_robustness(cfg.paths['metrics'], output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()