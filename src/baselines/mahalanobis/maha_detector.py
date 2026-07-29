import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from scipy import linalg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vit import ViTWrapper
from src.data.cifar_loader import get_cifar10_loaders, get_adversarial_loader


class ViTFeatureExtractor(torch.nn.Module):
    def __init__(self, vit_wrapper):
        super().__init__()
        self.model = vit_wrapper.model
        self.num_layers = len(self.model.encoder.layers)
        self.features = []

    def hook_fn(self, layer_idx):
        def hook(module, input, output):
            self.features[layer_idx] = output[:, 0, :].detach()
        return hook

    def forward(self, x):
        self.features = [None] * self.num_layers
        hooks = []
        for i, layer in enumerate(self.model.encoder.layers):
            h = layer.register_forward_hook(self.hook_fn(i))
            hooks.append(h)
        with torch.no_grad():
            _ = self.model(x)
        for h in hooks:
            h.remove()
        return torch.stack(self.features, dim=0)


class MahalanobisDetector:
    def __init__(self, model, device='cuda'):
        self.device = device
        self.extractor = ViTFeatureExtractor(model).to(device)
        self.num_layers = self.extractor.num_layers
        self.num_classes = 10
        self.class_means = {}
        self.precision_matrices = {}

    def fit(self, dataloader):
        print("Extracting features from training set...")
        all_features = [[[] for _ in range(self.num_classes)] for _ in range(self.num_layers)]
        for images, labels, _ in tqdm(dataloader, desc="Fitting"):
            images = images.to(self.device)
            features = self.extractor(images)
            for l in range(self.num_layers):
                feat_l = features[l].cpu().numpy()
                for i in range(feat_l.shape[0]):
                    c = labels[i].item()
                    all_features[l][c].append(feat_l[i])

        print("Computing class-conditional statistics...")
        for l in range(self.num_layers):
            self.class_means[l] = []
            self.precision_matrices[l] = []
            for c in range(self.num_classes):
                feats = np.stack(all_features[l][c], axis=0)
                mean = np.mean(feats, axis=0)
                cov = np.cov(feats, rowvar=False)
                cov += 0.01 * np.eye(cov.shape[0])
                precision = linalg.inv(cov)
                self.class_means[l].append(mean)
                self.precision_matrices[l].append(precision)
            self.class_means[l] = np.stack(self.class_means[l], axis=0)
            self.precision_matrices[l] = np.stack(self.precision_matrices[l], axis=0)
        print("Fit complete.")

    def compute_scores(self, dataloader):
        scores = []
        for images, labels, _ in tqdm(dataloader, desc="Scoring"):
            images = images.to(self.device)
            features = self.extractor(images)
            B = features.shape[1]
            min_distances = np.full(B, np.inf)
            for l in range(self.num_layers):
                feat_l = features[l].cpu().numpy()
                means_l = self.class_means[l]
                prec_l = self.precision_matrices[l]
                layer_min = np.full(B, np.inf)
                for c in range(self.num_classes):
                    diff = feat_l - means_l[c]
                    dist_sq = np.einsum('bd,de,be->b', diff, prec_l[c], diff)
                    dist = np.sqrt(dist_sq)
                    layer_min = np.minimum(layer_min, dist)
                min_distances = np.minimum(min_distances, layer_min)
            scores.extend(min_distances.tolist())
        return np.array(scores)


def evaluate_attack(detector, attack_name, test_loader, metrics_dir):
    print(f"\n{'='*60}")
    print(f"Evaluating on {attack_name.upper()} attack...")
    print(f"{'='*60}")

    clean_scores = detector.compute_scores(test_loader)
    print(f"Clean scores: mean={clean_scores.mean():.2f}, std={clean_scores.std():.2f}")

    adv_loader = get_adversarial_loader(
        attack_name=attack_name, split='test', batch_size=64, num_workers=0
    )
    adv_scores = detector.compute_scores(adv_loader)
    print(f"Adv scores:   mean={adv_scores.mean():.2f}, std={adv_scores.std():.2f}")

    from src.evaluation.metrics import summarize_metrics
    summary = summarize_metrics(
        clean_scores=clean_scores,
        adv_scores=adv_scores,
        save_path=os.path.join(metrics_dir, f'mahalanobis_{attack_name}_test.json'),
    )
    return summary


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    weights_path = os.path.join(PROJECT_ROOT, 'weights', 'vit_b_16_cifar10.pth')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"本地权重未找到: {weights_path}")
    print(f"Loading local weights from: {weights_path}")
    model = ViTWrapper(num_classes=10, weights_path=weights_path).to(device)
    model.eval()

    detector = MahalanobisDetector(model, device=device)

    train_loader, _ = get_cifar10_loaders(batch_size=64, num_workers=0)
    detector.fit(train_loader)

    _, test_loader = get_cifar10_loaders(batch_size=64, num_workers=0)
    metrics_dir = os.path.join(PROJECT_ROOT, 'outputs', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    for attack in ['fgsm', 'pgd20']:
        evaluate_attack(detector, attack, test_loader, metrics_dir)

    print(f"\n{'='*60}")
    print("All evaluations complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()