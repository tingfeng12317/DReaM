import os
import sys
import importlib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


REGISTRY = {
    'cifar10': {
        'torchvision_name': 'CIFAR10',
        'num_train': 50000,
        'num_test': 10000,
        'has_val_split': False,
        'legacy_adv_path': True,
    },
    'cifar100': {
        'torchvision_name': 'CIFAR100',
        'num_train': 50000,
        'num_test': 10000,
        'has_val_split': False,
        'legacy_adv_path': False,
    },
    'isic2018': {
        'custom_loader': 'src.data.isic_loader.get_isic2018_loader',
        'num_train': 10015,
        'num_val': 253,
        'num_test': 1512,
        'has_val_split': True,
        'legacy_adv_path': False,
    },
}


def get_spec(dataset_name):
    if dataset_name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Registered: {list(REGISTRY.keys())}"
        )
    return REGISTRY[dataset_name]


def get_num_samples(dataset_name, split='train'):
    """获取数据集样本数，支持独立 val 集"""
    spec = get_spec(dataset_name)
    if split == 'train':
        return spec.get('num_train', 0)
    elif split == 'val':
        return spec.get('num_val', spec.get('num_train', 0))
    elif split == 'test':
        return spec.get('num_test', 0)
    else:
        raise ValueError(f"Unknown split: {split}")


def _check_data_exists(data_root, tv_name):
    if tv_name == 'CIFAR10':
        marker = os.path.join(data_root, 'cifar-10-batches-py', 'data_batch_1')
    elif tv_name == 'CIFAR100':
        marker = os.path.join(data_root, 'cifar-100-batches-py', 'train')
    else:
        return False
    return os.path.exists(marker)


def get_loaders(dataset_name, split='train', batch_size=64, num_workers=0, pin_memory=True):
    spec = get_spec(dataset_name)

    # 非标准数据集（ISIC2018 等）
    if 'custom_loader' in spec:
        module_path, func_name = spec['custom_loader'].rsplit('.', 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name)(
            split=split, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory
        )

    # CIFAR-100: 自定义 pickle 加载
    if dataset_name == 'cifar100':
        from src.data.cifar_loader import get_cifar100_loaders
        train_loader, test_loader = get_cifar100_loaders(
            batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
        )
        if split == 'train':
            return train_loader
        elif split == 'val':
            return train_loader  # CIFAR-100 无独立 val，回退到 train
        elif split == 'test':
            return test_loader
        else:
            raise ValueError(f"Unknown split: {split}")

    # 标准 torchvision 数据集（CIFAR-10 等）
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    tv_name = spec['torchvision_name']
    tv_cls = getattr(datasets, tv_name)

    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_root = os.path.join(PROJECT_ROOT, 'data', dataset_name)
    os.makedirs(data_root, exist_ok=True)

    should_download = not _check_data_exists(data_root, tv_name)
    if not should_download:
        print(f"[{dataset_name}] Local data detected, skip download.")

    if split == 'train':
        dataset = tv_cls(root=data_root, train=True, download=should_download, transform=None)
        shuffle = True
    elif split == 'val':
        dataset = tv_cls(root=data_root, train=True, download=should_download, transform=None)
        shuffle = False
    elif split == 'test':
        dataset = tv_cls(root=data_root, train=False, download=should_download, transform=None)
        shuffle = False
    else:
        raise ValueError(f"Unknown split: {split}")

    class _Wrapper:
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            img, label = self.ds[idx]
            img = transform(img)
            return img, label, idx

    wrapped = _Wrapper(dataset)

    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )


def get_adversarial_loader(dataset_name, attack_name, split,
                           batch_size=64, num_workers=0, pin_memory=True):
    from torch.utils.data import DataLoader
    import pickle
    import torch

    new_path_dir = os.path.join(
        PROJECT_ROOT, 'data', 'adversarial',
        f'{attack_name}_{dataset_name}_{split}'
    )
    new_path = os.path.join(new_path_dir, 'adv_data.pkl')

    legacy_path_dir = os.path.join(
        PROJECT_ROOT, 'data', 'adversarial',
        f'{attack_name}_{split}'
    )
    legacy_path = os.path.join(legacy_path_dir, 'adv_data.pkl')

    if os.path.exists(new_path):
        load_path = new_path
    elif dataset_name == 'cifar10' and os.path.exists(legacy_path):
        load_path = legacy_path
        print(f"[dataset_registry] Legacy path detected: {load_path}")
    else:
        raise FileNotFoundError(
            f"对抗样本未找到。\n"
            f"  已查找: {new_path}\n"
            f"  {'已查找: ' + legacy_path if dataset_name == 'cifar10' else ''}\n"
            f"  请先运行: python src/attacks/generate_adv.py --config configs/{dataset_name}.yaml"
        )

    with open(load_path, 'rb') as f:
        data = pickle.load(f)

    class _AdvWrapper:
        def __init__(self, images, labels, indices):
            self.images = images
            self.labels = labels
            self.indices = indices
        def __len__(self):
            return len(self.images)
        def __getitem__(self, idx):
            img = self.images[idx].float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
            return img, self.labels[idx], self.indices[idx]

    wrapped = _AdvWrapper(data['images'], data['labels'], data['indices'])

    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )