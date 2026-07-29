import os
import sys
import json
import csv
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config


def extract_tuning_results(cfg):
    grid_path = cfg.grid_search_path('staged_tuning_results.json')

    if not os.path.exists(grid_path):
        raise FileNotFoundError(
            f"Tuning results not found: {grid_path}\n"
            f"Please run: python src/evaluation/tune_params.py --config configs/{cfg.dataset_name}.yaml"
        )

    with open(grid_path, 'r') as f:
        data = json.load(f)

    # 按 stage 分组
    stage1 = [r for r in data if r.get('stage') == 1]  # a
    stage2 = [r for r in data if r.get('stage') == 2]  # lambda_b
    stage3 = [r for r in data if r.get('stage') == 3]  # lambda_w

    out_dir = cfg.paths['grid_search']
    os.makedirs(out_dir, exist_ok=True)

    # 收集所有 dict 中出现的全部 key（避免不同阶段键不一致导致报错）
    all_keys = sorted(set().union(*(d.keys() for d in data)))

    # 统一 CSV（所有结果）
    if data:
        with open(os.path.join(out_dir, 'staged_tuning_results.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(data)
        print(f"Saved: {os.path.join(out_dir, 'staged_tuning_results.csv')}")

    # 分阶段 CSV
    for stage_data, name in [(stage1, 'a'), (stage2, 'lambda_b'), (stage3, 'lambda_w')]:
        if stage_data:
            stage_keys = sorted(set().union(*(d.keys() for d in stage_data)))
            with open(os.path.join(out_dir, f'{name}_tuning.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=stage_keys)
                w.writeheader()
                w.writerows(stage_data)
            print(f"Saved: {os.path.join(out_dir, f'{name}_tuning.csv')}")

    print(f"\nAll files extracted to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract staged tuning results to CSV')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    cfg = load_config(args.config)
    extract_tuning_results(cfg)


if __name__ == '__main__':
    main()