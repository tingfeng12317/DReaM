import os
import sys
import torch
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vit import ViTWrapper
from src.data.cifar_loader import get_cifar10_loaders, get_adversarial_loader


class ReActDetector:
    def __init__(self, model, device='cuda', percentile=90):
        self.device = device
        self.model = model
        self.model.eval()
        self.percentile = percentile
        self.tau = None
        self.id_energy_mean = None
        self.id_energy_std = None

    def _extract_energy(self, x):
        with torch.no_grad():
            _ = self.model(x)
            layer_out = self.model._layer_outputs[10]
            if self.tau is not None:
                layer_out = torch.where(layer_out > self.tau, layer_out, torch.zeros_like(layer_out))
            layer11 = self.model.model.encoder.layers[11]
            x_norm = layer11.ln_1(layer_out)
            attn_out, _ = self.model._extract_attention_heads(layer11, x_norm)
            x = layer_out + layer11.dropout(attn_out)
            x = x + layer11.mlp(layer11.ln_2(x))
            x = self.model.model.encoder.ln(x)
            x = x[:, 0]
            logits = self.model.model.heads(x)
            T = 1.0
            energy = -T * torch.logsumexp(logits / T, dim=1)
        return energy.cpu().numpy()

    def calibrate(self, dataloader):
        all_activations = []
        all_energies = []
        for images, labels, _ in tqdm(dataloader, desc="ReAct Calibrating"):
            images = images.to(self.device)
            with torch.no_grad():
                _ = self.model(images)
                all_activations.append(self.model._layer_outputs[10].cpu().numpy())
            energies = self._extract_energy(images)
            all_energies.extend(energies.tolist())

        all_activations = np.concatenate(all_activations, axis=0).reshape(-1)
        self.tau = np.percentile(all_activations, self.percentile)
        print(f"\nReAct threshold τ (p={self.percentile}): {self.tau:.4f}")

        all_energies = np.array(all_energies)
        self.id_energy_mean = np.mean(all_energies)
        self.id_energy_std = np.std(all_energies) + 1e-8
        print(f"ID Energy: mean={self.id_energy_mean:.2f}, std={self.id_energy_std:.2f}")

    def compute_scores(self, dataloader):
        scores = []
        for images, labels, _ in tqdm(dataloader, desc="ReAct Scoring"):
            images = images.to(self.device)
            energies = self._extract_energy(images)
            z_scores = (energies - self.id_energy_mean) / self.id_energy_std
            anomaly_scores = -z_scores
            scores.extend(anomaly_scores.tolist())
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
        save_path=os.path.join(metrics_dir, f'react_{attack_name}_test.json'),
    )
    return summary


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    weights_path = os.path.join(PROJECT_ROOT, 'weights', 'vit_b_16_cifar10.pth')
    model = ViTWrapper(num_classes=10, weights_path=weights_path).to(device)
    model.eval()

    detector = ReActDetector(model, device=device, percentile=90)

    _, val_loader = get_cifar10_loaders(batch_size=64, num_workers=0)
    detector.calibrate(val_loader)

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