# -*- coding: utf-8 -*-
"""
train_head.py（通用版 v3，直接读 YAML + registry 统一入口）
冻结 ImageNet 预训练主干，只训练指定数据集的分类头（linear probe）。
每个数据集训一次，产出 weights/vit_b_16_{dataset}.pth。

用法：
  python src/models/train_head.py --config configs/cifar10.yaml
  python src/models/train_head.py --config configs/cifar100.yaml
  python src/models/train_head.py --config configs/isic2018.yaml
"""

import os
import sys
import argparse
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vit import ViTWrapper

MOTHER_WEIGHTS = os.path.join(PROJECT_ROOT, 'weights', 'vit_b_16-c867db91.pth')
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3
SEED = 42


def read_dataset_info(config_path):
    """直接读 YAML 原文，取 dataset.name / dataset.num_classes / num_workers。"""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config 不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    ds = raw.get('dataset', {}) if isinstance(raw, dict) else {}
    name = ds.get('name') or raw.get('name')
    num_classes = ds.get('num_classes') or raw.get('num_classes')
    num_workers = (raw.get('training_free', {}) or {}).get('num_workers', 0)
    if not name or not num_classes:
        raise KeyError(
            f"在 {config_path} 中找不到 dataset.name / dataset.num_classes。\n"
            f"该 YAML 的顶层字段有: {list(raw.keys())}\n"
            f"请把该文件发给我，我来适配字段名。"
        )
    return str(name), int(num_classes), int(num_workers)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_train_test_loaders(dataset_name, batch_size, num_workers):
    """按数据集名返回 (train_loader, test_loader)。
    loader 的 batch 可以是 (img, label) 或 (img, label, idx)，下游只取前两个。"""
    if dataset_name == 'cifar10':
        from src.data.cifar_loader import get_cifar10_loaders
        return get_cifar10_loaders(batch_size=batch_size, num_workers=num_workers)

    if dataset_name == 'cifar100':
        from src.data.cifar_loader import CIFAR100Dataset
        train_set = CIFAR100Dataset(split='train')
        test_set = CIFAR100Dataset(split='test')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
        return train_loader, test_loader

    # 其他数据集（isic2018 等）：走 dataset_registry 统一入口
    try:
        from src.data.dataset_registry import get_loaders
        train_loader = get_loaders(dataset_name, split='train', batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=True)
        test_loader = get_loaders(dataset_name, split='test', batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=True)
        print(f"[train_head] 已通过 dataset_registry.get_loaders 获取 {dataset_name} (train/test)")
        return train_loader, test_loader
    except Exception as e:
        raise NotImplementedError(
            f"通过 dataset_registry 获取 '{dataset_name}' 失败: {e}\n"
            f"请把 src/data/isic_loader.py 和 src/data/dataset_registry.py 发给我，"
            f"我来适配 split 名称。"
        )


@torch.no_grad()
def extract_features(model, loader, device):
    feats, labels = [], []
    for batch in loader:
        images, y = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        feats.append(model.forward_features(images).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def evaluate(model, loader, device):
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            images, y = batch[0].to(device), batch[1].to(device)
            correct += (model(images).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--weights-in', type=str, default=MOTHER_WEIGHTS,
                        help='预训练母本权重（默认官方 ImageNet 文件）')
    parser.add_argument('--weights-out', type=str, default='',
                        help='输出路径（默认 weights/vit_b_16_{dataset}.pth）')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()

    dataset_name, num_classes, num_workers = read_dataset_info(args.config)
    weights_out = args.weights_out or os.path.join(
        PROJECT_ROOT, 'weights', f'vit_b_16_{dataset_name}.pth')

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Dataset: {dataset_name} | Classes: {num_classes}")

    model = ViTWrapper(num_classes=num_classes, weights_path=args.weights_in).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    train_loader, test_loader = get_train_test_loaders(dataset_name, BATCH_SIZE, num_workers)

    acc_before = evaluate(model, test_loader, device)
    print(f"[Before] clean test accuracy (random head): {acc_before * 100:.2f}%")

    print("\n[1/2] 提取训练集特征 ...")
    train_feats, train_labels = extract_features(model, train_loader, device)
    print(f"  train features: {train_feats.shape}")
    print("[1/2] 提取测试集特征 ...")
    test_feats, test_labels = extract_features(model, test_loader, device)

    print("\n[2/2] 训练分类头 (linear probe) ...")
    head = nn.Linear(768, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    train_feats_gpu = train_feats.to(device)
    train_labels_gpu = train_labels.to(device)
    n = train_feats_gpu.size(0)

    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss, nb = 0.0, 0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            loss = criterion(head(train_feats_gpu[idx]), train_labels_gpu[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        scheduler.step()
        with torch.no_grad():
            pred = head(test_feats.to(device)).argmax(dim=1)
            acc = (pred == test_labels.to(device)).float().mean().item()
        print(f"  epoch {epoch + 1:2d}/{args.epochs}  loss={epoch_loss / nb:.4f}  "
              f"test_acc={acc * 100:.2f}%")

    model.model.heads.head.load_state_dict(head.state_dict())
    acc_after = evaluate(model, test_loader, device)
    print(f"\n[After] clean test accuracy (trained head): {acc_after * 100:.2f}%")

    torch.save(model.model.state_dict(), weights_out)
    print(f"[Saved] {weights_out}")
    print(f"\n下一步：把 configs/{dataset_name}.yaml 里的 model.weights_path 改为：")
    print(f"  ./weights/vit_b_16_{dataset_name}.pth")


if __name__ == '__main__':
    main()