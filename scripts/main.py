import os
import sys
import shutil
import argparse
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config


# ============================================
# 自动解析 Python 解释器（通用版，无硬编码路径）
# ============================================
def resolve_python():
    """
    1. 优先用当前运行的 Python（如果已经是 conda 环境）
    2. 尝试从 CONDA_PREFIX 环境变量推断
    3. 尝试用 shutil.which 找 conda 环境的 python
    4. 兜底：返回 sys.executable 并打印警告
    """
    exe = sys.executable

    # 情况 1：当前已经是 conda 环境（路径含 envs 或 conda）
    if 'envs' in exe or 'conda' in exe.lower():
        return exe

    # 情况 2：从 CONDA_PREFIX 推断（conda activate 后运行）
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidate = os.path.join(conda_prefix, 'python.exe')
        if os.path.exists(candidate):
            return candidate

    # 情况 3：尝试用 conda 命令查找默认环境
    conda_exe = shutil.which('conda')
    if conda_exe:
        try:
            # 获取当前激活的 conda 环境路径
            result = subprocess.run(
                [conda_exe, 'run', '-n', 'base', 'python', '-c',
                 'import sys; print(sys.executable)'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                p = result.stdout.strip()
                if os.path.exists(p):
                    return p
        except Exception:
            pass

    # 情况 4：兜底
    print(f"[WARNING] Not running in a conda environment.")
    print(f"[WARNING] Using: {exe}")
    print("[HINT] For CUDA support, please run with:")
    print("       conda activate your_env")
    print("       python scripts/main.py --config configs/xxx.yaml")
    print("       OR specify explicitly:")
    print("       your_conda_python scripts/main.py --config configs/xxx.yaml")
    return exe


PYTHON_EXE = resolve_python()
print(f"[INFO] Using Python: {PYTHON_EXE}")


# ============================================
# 步骤定义与检查函数
# ============================================
def _check_profiler(cfg):
    return os.path.exists(cfg.paths['topk'])


def _check_distribution(cfg):
    return os.path.exists(cfg.paths['distribution'])


def _check_generate_adv(cfg):
    attacks = cfg.get('attack.attacks_to_generate', [])
    splits = cfg.get('attack.splits_to_generate', [])
    for attack in attacks:
        for split in splits:
            if not os.path.exists(cfg.adv_path(attack, split)):
                return False
    return True


def _check_tune_params(cfg):
    return os.path.exists(cfg.grid_search_path('staged_tuning_results.json'))


def _check_evaluate_test(cfg):
    return os.path.exists(cfg.metrics_path('test_final_tuned_report.json'))


STEPS = [
    ('profiler',      'src/dream/profiler.py',       _check_profiler),
    ('distribution',  'src/dream/distribution.py',   _check_distribution),
    ('generate_adv',  'src/attacks/generate_adv.py', _check_generate_adv),
    ('tune_params',   'src/evaluation/tune_params.py', _check_tune_params),
    ('evaluate_test', 'src/evaluation/evaluate_test.py', _check_evaluate_test),
]

STEP_MAP = {s[0]: s for s in STEPS}


def run_step(step_name, script_path, cfg, force=False):
    script_abs = os.path.join(PROJECT_ROOT, script_path)
    check_fn = STEP_MAP[step_name][2] if step_name in STEP_MAP else None

    if not force and check_fn and check_fn(cfg):
        print(f"\n{'=' * 60}")
        print(f"[SKIP] {step_name}: output already exists.")
        print(f"{'=' * 60}")
        return True

    print(f"\n{'=' * 60}")
    print(f"[RUN] {step_name}  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"{'=' * 60}")

    cmd = [PYTHON_EXE, script_abs, '--config', cfg._yaml_path]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\n[ERROR] {step_name} failed with exit code {result.returncode}")
        return False

    print(f"\n[DONE] {step_name} completed.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='DReaM Pipeline: one-click execution of core experiments'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML (e.g., configs/cifar10.yaml)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run all steps even if outputs exist')
    parser.add_argument('--step', type=str, default=None,
                        choices=[s[0] for s in STEPS],
                        help='Run only one specific step (for debugging)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"\n{'#' * 60}")
    print(f"# DReaM Pipeline")
    print(f"# Dataset: {cfg.dataset_name}")
    print(f"# Config:  {cfg._yaml_path}")
    print(f"# Python:  {PYTHON_EXE}")
    print(f"# Force:   {args.force}")
    print(f"# Step:    {args.step or 'ALL'}")
    print(f"{'#' * 60}")

    steps_to_run = [(name, path) for name, path, _ in STEPS]
    if args.step:
        steps_to_run = [(name, path) for name, path, _ in STEPS if name == args.step]
        if not steps_to_run:
            print(f"[ERROR] Unknown step: {args.step}")
            sys.exit(1)

    for step_name, script_path in steps_to_run:
        if not run_step(step_name, script_path, cfg, force=args.force):
            sys.exit(1)

    print(f"\n{'#' * 60}")
    print(f"# ALL STEPS COMPLETED SUCCESSFULLY")
    print(f"{'#' * 60}")
    print(f"\nFinal outputs:")
    print(f"  Top-K:        {cfg.paths['topk']}")
    print(f"  Distribution: {cfg.paths['distribution']}")
    print(f"  Grid search:  {cfg.grid_search_path('staged_tuning_results.json')}")
    print(f"  Test report:  {cfg.metrics_path('test_final_tuned_report.json')}")


if __name__ == '__main__':
    main()