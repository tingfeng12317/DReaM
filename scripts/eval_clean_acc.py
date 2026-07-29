# -*- coding: utf-8 -*-
"""
eval_clean_acc.py — 只评估 clean test accuracy，不训练、不覆盖任何权重。
用法：
  python scripts/eval_clean_acc.py --config configs/cifar10.yaml
  python scripts/eval_clean_acc.py --config configs/cifar100.yaml
  python scripts/eval_clean_acc.py --config configs/isic2018.yaml
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vit import ViTWrapper

BATCH_SIZE = 512


def read_info(config_path):
    """从 YAML 取 dataset.name / num_classes / model.weights_path / num_workers。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    ds = raw.get('dataset', {}) if isinstance(raw, dict) else {}
    name = ds.get('name') or raw.get('name')
    num_classes = ds.get('num_classes') or raw.get('num_classes')
    weights_path = (raw.get('model', {}) or {}).get('weights_path')
    num_workers = (raw.get('training_free', {}) or {}).get('num_workers', 0)
    if not name or not num_classes or not weights_path:
        raise KeyError(f"YAML 字段缺失: name={name}, num_classes={num_classes}, weights_path={weights_path}")
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(PROJECT_ROOT, weights_path)
    return str(name), int(num_classes), weights_path, int(num_workers)


def get_test_loader(dataset_name, batch_size, num_workers):
    """只取 test loader，与 train_head.py v3 的取数方式保持一致。"""
    if dataset_name == 'cifar10':
        from src.data.cifar_loader import get_cifar10_loaders
        return get_cifar10_loaders(batch_size=batch_size, num_workers=num_workers)[1]

    if dataset_name == 'cifar100':
        from src.data.cifar_loader import CIFAR100Dataset
        test_set = CIFAR100Dataset(split='test')
        return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    # 其他数据集（isic2018 等）：registry 统一入口
    from src.data.dataset_registry import get_loaders
    return get_loaders(dataset_name, split='test', batch_size=batch_size,
                       num_workers=num_workers, pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    dataset_name, num_classes, weights_path, num_workers = read_info(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Dataset: {dataset_name} | Classes: {num_classes}")
    print(f"Weights: {weights_path}")

    model = ViTWrapper(num_classes=num_classes, weights_path=weights_path).to(device)
    model.eval()

    test_loader = get_test_loader(dataset_name, BATCH_SIZE, num_workers)

    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            images, y = batch[0].to(device), batch[1].to(device)
            correct += (model(images).argmax(dim=1) == y).sum().item()
            total += y.size(0)

    print(f"\n[Eval] {dataset_name} clean test accuracy: {correct / total * 100:.2f}%  ({correct}/{total})")


if __name__ == '__main__':
    main()