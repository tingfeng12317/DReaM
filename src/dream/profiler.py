import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vit import ViTWrapper
from src.models.hooks import GradientCollector
from src.data.dataset_registry import get_loaders
from src.utils.config import load_config


def profile_topk(K, dataset_name, num_classes, batch_size, num_workers, device, weights_path, save_path):
    print(f"Loading {dataset_name.upper()}...")
    train_loader = get_loaders(dataset_name, split='train', batch_size=batch_size, num_workers=num_workers)
    num_batches = len(train_loader)
    print(f"Total batches: {num_batches}, Batch size: {batch_size}")

    print("Loading ViT-B/16...")
    if weights_path and os.path.exists(weights_path):
        model = ViTWrapper(num_classes=num_classes, weights_path=weights_path).to(device)
        print(f"Loaded weights from: {weights_path}")
    else:
        model = ViTWrapper(num_classes=num_classes, weights='imagenet').to(device)
    model.eval()

    collector = GradientCollector(model)
    criterion = nn.CrossEntropyLoss()

    num_layers = model.num_layers
    num_heads = model.num_heads
    fisher_accum = [torch.zeros(num_heads, device=device) for _ in range(num_layers)]
    total_samples = 0

    print(f"\nStarting first-pass profiling (K={K})...")
    print("Mode: eval() with gradient enabled (training-free)")

    pbar = tqdm(train_loader, desc="Profiling")
    for images, labels, indices in pbar:
        images = images.to(device)
        labels = labels.to(device)
        B = images.size(0)

        images.requires_grad = True

        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        fisher_per_layer = collector.compute_fisher()

        for l in range(num_layers):
            fisher_accum[l] += fisher_per_layer[l] * B

        total_samples += B

        images.requires_grad = False
        model.zero_grad()

        pbar.set_postfix({"samples": total_samples, "loss": f"{loss.item():.4f}"})

    fisher_avg = [f / total_samples for f in fisher_accum]

    topk_indices = {}
    print(f"\n{'='*50}")
    print("Top-K Selection (layer-wise independent sorting):")
    print(f"{'='*50}")

    for l in range(num_layers):
        values = fisher_avg[l].cpu().numpy()
        topk_idx = torch.topk(fisher_avg[l], k=K).indices.cpu().tolist()
        topk_indices[l] = topk_idx

        topk_values = [f"{values[i]:.6f}" for i in topk_idx]
        print(f"Layer {l+1:2d}: Top-{K} heads = {topk_idx}, Fisher = {topk_values}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(topk_indices, f, indent=2)
    print(f"\nSaved Top-K indices to: {save_path}")

    print(f"\n{'='*50}")
    print("Layer-wise Fisher Magnitude (mean over heads):")
    print(f"{'='*50}")
    for l in range(num_layers):
        mean_fisher = fisher_avg[l].mean().item()
        print(f"Layer {l+1:2d}: mean Fisher = {mean_fisher:.6f}")

    return topk_indices


def main():
    parser = argparse.ArgumentParser(description='DReaM Profiler - First-pass Fisher-based head screening')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML (e.g., configs/cifar100.yaml)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 从 config 读取所有参数
    dataset_name = cfg.dataset_name
    num_classes = cfg.num_classes
    K = cfg.get('dream.K')
    batch_size = cfg.get('training_free.batch_size_profiler')
    num_workers = cfg.get('training_free.num_workers')
    device = cfg.device
    weights_path = cfg.weights_path

    # 自动生成保存路径
    save_path = cfg.paths['topk']

    print(f"\n{'='*60}")
    print(f"Configuration: {dataset_name}")
    print(f"  num_classes: {num_classes}")
    print(f"  K: {K}")
    print(f"  batch_size: {batch_size}")
    print(f"  device: {device}")
    print(f"  weights: {weights_path}")
    print(f"  output: {save_path}")
    print(f"{'='*60}")

    topk = profile_topk(
        K=K,
        dataset_name=dataset_name,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        weights_path=weights_path,
        save_path=save_path
    )

    print(f"\n{'='*60}")
    print("Profiling complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()