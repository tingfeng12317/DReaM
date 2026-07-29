import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

# ============================================
# CIFAR-10 (torchvision 正常)
# ============================================
import torchvision


class CIFAR10Dataset(Dataset):
    def __init__(self, train=True, download=True):
        self.train = train
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.dataset = torchvision.datasets.CIFAR10(
            root=os.path.join(DATA_ROOT, 'cifar10'),
            train=train,
            download=download,
            transform=None
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label, idx


def get_cifar10_loaders(batch_size=64, num_workers=4, pin_memory=True):
    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    return train_loader, test_loader


# ============================================
# CIFAR-100 (自定义 pickle 加载，绕过 torchvision 下载校验)
# ============================================
class CIFAR100Dataset(Dataset):
    def __init__(self, split='train'):
        assert split in ('train', 'test'), f"split must be 'train' or 'test', got {split}"

        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        data_path = os.path.join(DATA_ROOT, 'cifar100', 'cifar-100-batches-py', split)

        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')

        # data: (N, 3072) uint8 -> (N, 3, 32, 32) -> (N, 32, 32, 3) for PIL
        self.images = data_dict[b'data'].reshape(-1, 3, 32, 32).astype(np.uint8)
        self.images = np.transpose(self.images, (0, 2, 3, 1))  # (N, 32, 32, 3)
        self.labels = data_dict[b'fine_labels']

        print(f"[CIFAR-100] Loaded {split}: {len(self.images)} samples, shape {self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (32, 32, 3), uint8
        img = Image.fromarray(img)  # PIL Image
        img = self.transform(img)  # Tensor (3, 224, 224), normalized
        return img, self.labels[idx], idx


def get_cifar100_loaders(batch_size=64, num_workers=0, pin_memory=True):
    train_dataset = CIFAR100Dataset(split='train')
    test_dataset = CIFAR100Dataset(split='test')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    return train_loader, test_loader


# ============================================
# Adversarial Dataset (不变)
# ============================================
class AdversarialDataset(Dataset):
    def __init__(self, dataset_name='cifar10', attack_name='pgd20', split='val'):
        self.dataset_name = dataset_name
        self.attack_name = attack_name
        self.split = split

        new_path_dir = os.path.join(DATA_ROOT, 'adversarial', f'{attack_name}_{dataset_name}_{split}')
        new_path = os.path.join(new_path_dir, 'adv_data.pkl')

        legacy_path_dir = os.path.join(DATA_ROOT, 'adversarial', f'{attack_name}_{split}')
        legacy_path = os.path.join(legacy_path_dir, 'adv_data.pkl')

        if os.path.exists(new_path):
            self.save_path = new_path
            self.using_legacy = False
        elif dataset_name == 'cifar10' and os.path.exists(legacy_path):
            self.save_path = legacy_path
            self.using_legacy = True
            print(f"[AdversarialDataset] Legacy path detected: {legacy_path}")
        else:
            raise FileNotFoundError(
                f"对抗样本未找到。\n"
                f"  已查找: {new_path}\n"
                f"  {'已查找: ' + legacy_path if dataset_name == 'cifar10' else ''}\n"
                f"  请先运行: python src/attacks/generate_adv.py --config configs/{dataset_name}.yaml"
            )

        with open(self.save_path, 'rb') as f:
            data = pickle.load(f)

        self.images = data['images']
        self.labels = data['labels']
        self.indices = data['indices']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return img, self.labels[idx], self.indices[idx]


def get_adversarial_loader(dataset_name='cifar10', attack_name='pgd20', split='val',
                           batch_size=64, num_workers=0, pin_memory=True):
    dataset = AdversarialDataset(
        dataset_name=dataset_name,
        attack_name=attack_name,
        split=split
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )


if __name__ == '__main__':
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_ROOT: {DATA_ROOT}")

    print("\n=== Test CIFAR-100 loader ===")
    train_loader, test_loader = get_cifar100_loaders(batch_size=4, num_workers=0)
    batch = next(iter(train_loader))
    print(f"images={batch[0].shape}, labels={batch[1].shape}, indices={batch[2].shape}")
    print(f"image range: [{batch[0].min():.3f}, {batch[0].max():.3f}]")
    print(f"labels: {batch[1].tolist()}")