import os
import sys
import argparse
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vit import ViTWrapper
from src.data.dataset_registry import get_loaders, get_num_samples
from src.attacks.pgd import PGDAttack, FGSMAttack
from src.utils.config import load_config
import pickle


def generate_adversarial_dataset(dataset_name, num_classes, split, attack_name,
                                  batch_size, num_workers, device, weights_path):
    save_dir = os.path.join(PROJECT_ROOT, 'data', 'adversarial',
                            f'{attack_name}_{dataset_name}_{split}')
    os.makedirs(save_dir, exist_ok=True)

    total_samples = get_num_samples(dataset_name, split=split)

    # 关键修改：直接按 split 请求 loader，不再把 val 硬编码成 train
    loader = get_loaders(dataset_name, split=split,
                         batch_size=batch_size, num_workers=num_workers)

    print(f"Generating adversarial samples for {dataset_name.upper()} "
          f"{split} set ({total_samples} samples)...")

    print("Loading ViT-B/16...")
    if weights_path and os.path.exists(weights_path):
        model = ViTWrapper(num_classes=num_classes, weights_path=weights_path).to(device)
        print(f"Loaded weights from: {weights_path}")
    else:
        model = ViTWrapper(num_classes=num_classes, weights='imagenet').to(device)
    model.eval()

    if attack_name == 'fgsm':
        attack = FGSMAttack(model, epsilon=8/255)
    elif attack_name == 'pgd20':
        attack = PGDAttack(model, epsilon=8/255, alpha=2/255, steps=20)
    elif attack_name == 'pgd100':
        attack = PGDAttack(model, epsilon=8/255, alpha=2/255, steps=100)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    all_images = torch.zeros(total_samples, 3, 224, 224, dtype=torch.uint8)
    all_labels = torch.zeros(total_samples, dtype=torch.long)
    all_indices = torch.zeros(total_samples, dtype=torch.long)

    idx = 0
    pbar = tqdm(loader, desc=f"{attack_name}_{dataset_name}_{split}")
    for images, labels, indices in pbar:
        images = images.to(device)
        labels_dev = labels.to(device)

        adv_images = attack.attack(images, labels_dev)

        adv_images = adv_images.cpu()
        adv_images = adv_images * std + mean
        adv_images = (adv_images * 255.0).clamp(0, 255).to(torch.uint8)

        B = adv_images.size(0)
        all_images[idx:idx+B] = adv_images
        all_labels[idx:idx+B] = labels
        all_indices[idx:idx+B] = indices

        idx += B

        del images, labels_dev, adv_images
        torch.cuda.empty_cache()

    save_path = os.path.join(save_dir, 'adv_data.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'images': all_images,
            'labels': all_labels,
            'indices': all_indices
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved to: {save_path}")
    print(f"Total samples: {idx}")
    print(f"File size: {os.path.getsize(save_path)/1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description='Generate adversarial samples for DReaM evaluation'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML')
    parser.add_argument('--attack', type=str, default=None,
                        help='Attack name (fgsm/pgd20/pgd100)')
    parser.add_argument('--split', type=str, default=None,
                        help='Split (train/val/test)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_name = cfg.dataset_name
    num_classes = cfg.num_classes
    batch_size = cfg.get('training_free.batch_size_generate')
    num_workers = cfg.get('training_free.num_workers')
    device = cfg.device
    weights_path = cfg.weights_path

    attacks = [args.attack] if args.attack else cfg.get('attack.attacks_to_generate')
    splits = [args.split] if args.split else cfg.get('attack.splits_to_generate')

    for split in splits:
        for attack_name in attacks:
            print(f"\n{'='*60}")
            generate_adversarial_dataset(
                dataset_name=dataset_name,
                num_classes=num_classes,
                split=split,
                attack_name=attack_name,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                weights_path=weights_path
            )

    print(f"\n{'='*60}")
    print("All adversarial samples generated!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()