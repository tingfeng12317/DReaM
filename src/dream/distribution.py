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
from src.data.dataset_registry import get_loaders
from src.utils.config import load_config


def compute_median_mad(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-8:
        mad = 1e-8
    return float(median), float(mad)


def build_distributions(K, dataset_name, num_classes, batch_size, num_workers, device,
                        weights_path, topk_path, save_path):
    if not os.path.exists(topk_path):
        raise FileNotFoundError(f"Top-K 文件未找到: {topk_path}，请先运行 profiler.py")

    with open(topk_path, 'r') as f:
        topk_indices = json.load(f)
    topk_indices = {int(k): v for k, v in topk_indices.items()}

    print(f"Loaded Top-K indices from: {topk_path}")
    for l, heads in topk_indices.items():
        print(f"  Layer {l+1:2d}: heads {heads}")

    print(f"\nLoading {dataset_name.upper()} training set...")
    train_loader = get_loaders(dataset_name, split='train', batch_size=batch_size, num_workers=num_workers)
    print(f"Total batches: {len(train_loader)}, Batch size: {batch_size}")

    print("Loading ViT-B/16...")
    if weights_path and os.path.exists(weights_path):
        model = ViTWrapper(num_classes=num_classes, weights_path=weights_path).to(device)
        print(f"Loaded weights from: {weights_path}")
    else:
        model = ViTWrapper(num_classes=num_classes, weights='imagenet').to(device)
    model.eval()

    num_layers = model.num_layers

    residual_values = [[] for _ in range(num_layers)]
    activation_values = [{} for _ in range(num_layers)]

    for l in range(num_layers):
        for h in topk_indices[l]:
            activation_values[l][h] = []

    print(f"\n{'='*60}")
    print("Starting second-pass distribution building (no_grad)...")
    print(f"{'='*60}")

    with torch.no_grad():
        pbar = tqdm(train_loader, desc="Distribution")
        for images, labels, indices in pbar:
            images = images.to(device)
            _ = model(images, collect_for_dream=True)

            for l in range(num_layers):
                attn_out = model._attn_outputs[l]
                residual = torch.norm(attn_out, p=2, dim=2).mean(dim=1)
                residual_values[l].extend(residual.cpu().numpy().tolist())

            for l in range(num_layers):
                head_out = model._dream_head_outputs[l]
                for h in topk_indices[l]:
                    h_out = head_out[:, h, :, :]
                    activation = torch.norm(h_out, p=2, dim=2).mean(dim=1)
                    activation_values[l][h].extend(activation.cpu().numpy().tolist())

    distributions = {
        'K': K,
        'num_layers': num_layers,
        'residual': {},
        'activation': {}
    }

    print(f"\n{'='*60}")
    print("Computing Median + MAD for Residuals (layer-wise):")
    print(f"{'='*60}")
    for l in range(num_layers):
        values = np.array(residual_values[l])
        median, mad = compute_median_mad(values)
        distributions['residual'][l] = {
            'median': median,
            'mad': mad,
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'num_samples': len(values)
        }
        print(f"Layer {l+1:2d}: median={median:.6f}, MAD={mad:.6f}, "
              f"mean={np.mean(values):.6f}, N={len(values)}")

    print(f"\n{'='*60}")
    print("Computing Median + MAD for Activations (Top-K heads):")
    print(f"{'='*60}")
    for l in range(num_layers):
        distributions['activation'][l] = {}
        for h in topk_indices[l]:
            values = np.array(activation_values[l][h])
            median, mad = compute_median_mad(values)
            distributions['activation'][l][h] = {
                'median': median,
                'mad': mad,
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'num_samples': len(values)
            }
            print(f"Layer {l+1:2d} Head {h:2d}: median={median:.6f}, MAD={mad:.6f}, "
                  f"mean={np.mean(values):.6f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(distributions, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Saved distributions to: {save_path}")
    print(f"{'='*60}")

    return distributions


def main():
    parser = argparse.ArgumentParser(description='DReaM Distribution - Second-pass median+MAD baseline construction')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_name = cfg.dataset_name
    num_classes = cfg.num_classes
    K = cfg.get('dream.K')
    batch_size = cfg.get('training_free.batch_size_distribution')
    num_workers = cfg.get('training_free.num_workers')
    device = cfg.device
    weights_path = cfg.weights_path

    topk_path = cfg.paths['topk']
    save_path = cfg.paths['distribution']

    print(f"\n{'='*60}")
    print(f"Configuration: {dataset_name}")
    print(f"  num_classes: {num_classes}")
    print(f"  K: {K}")
    print(f"  batch_size: {batch_size}")
    print(f"  device: {device}")
    print(f"  topk input: {topk_path}")
    print(f"  output: {save_path}")
    print(f"{'='*60}")

    dist = build_distributions(
        K=K,
        dataset_name=dataset_name,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        weights_path=weights_path,
        topk_path=topk_path,
        save_path=save_path
    )

    print(f"\n{'='*60}")
    print("Second-pass distribution building complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()