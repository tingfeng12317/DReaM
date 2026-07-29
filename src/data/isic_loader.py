import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class ISIC2018Dataset(Dataset):
    def __init__(self, split='train', data_root=None):
        if data_root is None:
            data_root = os.path.join(PROJECT_ROOT, 'data', 'isic2018')
        self.split = split
        self.img_dir = os.path.join(data_root, split, 'images')

        labels_csv = os.path.join(data_root, split, 'labels.csv')
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(
                f"{labels_csv} not found. "
                f"Run: python scripts/prepare_isic2018.py"
            )

        df = pd.read_csv(labels_csv)
        self.image_names = df['image'].astype(str).values
        self.labels = df['label'].values

        # 关键修改：强制 resize 为 224x224 正方形，匹配 ViT 预训练权重
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            candidate = os.path.join(self.img_dir, img_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            candidate = os.path.join(self.img_dir, img_name)
            if os.path.exists(candidate):
                img_path = candidate
            else:
                raise FileNotFoundError(
                    f"Image '{img_name}' not found in {self.img_dir}"
                )

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = int(self.labels[idx])
        return img, label, idx


def get_isic2018_loaders(batch_size=64, num_workers=0, pin_memory=True):
    """返回 (train_loader, val_loader, test_loader)"""
    train_ds = ISIC2018Dataset('train')
    val_ds = ISIC2018Dataset('val')
    test_ds = ISIC2018Dataset('test')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    return train_loader, val_loader, test_loader


def get_isic2018_loader(split='train', batch_size=64, num_workers=0, pin_memory=True):
    """Registry 使用的单 split 入口"""
    train_loader, val_loader, test_loader = get_isic2018_loaders(
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
    )
    if split == 'train':
        return train_loader
    elif split == 'val':
        return val_loader
    elif split == 'test':
        return test_loader
    else:
        raise ValueError(f"Unknown split: {split}")


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_isic2018_loaders(batch_size=8)
    for name, loader in [('train', train_loader),
                          ('val', val_loader),
                          ('test', test_loader)]:
        images, labels, indices = next(iter(loader))
        print(f"{name}: images={images.shape}, labels={labels.shape}, "
              f"value_range=[{images.min():.3f}, {images.max():.3f}]")