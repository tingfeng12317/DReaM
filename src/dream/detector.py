import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vit import ViTWrapper


class DReaMDetector:
    def __init__(self, model, distributions_path, topk_path,
                 a=1.5, lambda_weight=0.5, lambda_b=3.0, K=3, device='cuda'):
        self.model = model
        self.model.eval()
        self.device = device
        self.a = a
        self.lambda_weight = lambda_weight
        self.lambda_b = lambda_b
        self.K = K
        self.B_max = None

        with open(distributions_path, 'r') as f:
            self.distributions = json.load(f)
        with open(topk_path, 'r') as f:
            self.topk_indices = {int(k): v for k, v in json.load(f).items()}

        self.num_layers = self.distributions['num_layers']

    def _compute_layer_score(self, l, attn_out, head_out):
        residual = torch.norm(attn_out, p=2, dim=2).mean(dim=1)
        median_res = self.distributions['residual'][str(l)]['median']
        mad_res = self.distributions['residual'][str(l)]['mad']
        S_res = torch.abs(residual - median_res) / (mad_res + 1e-8)

        S_act_list = []
        for h in self.topk_indices[l]:
            h_out = head_out[:, h, :, :]
            activation = torch.norm(h_out, p=2, dim=2).mean(dim=1)
            median_act = self.distributions['activation'][str(l)][str(h)]['median']
            mad_act = self.distributions['activation'][str(l)][str(h)]['mad']
            S_act_h = torch.abs(activation - median_act) / (mad_act + 1e-8)
            S_act_list.append(S_act_h)

        S_act = torch.stack(S_act_list, dim=0).mean(dim=0)
        layer_score = self.lambda_weight * S_res + (1 - self.lambda_weight) * S_act
        return layer_score

    def detect(self, x, early_stop=True):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            _ = self.model(x, collect_for_dream=True)

        B = x.size(0)
        W = torch.zeros(B, device=self.device)
        stop_layer = torch.full((B,), -1, dtype=torch.long, device=self.device)

        for l in range(self.num_layers):
            attn_out = self.model._attn_outputs[l]
            head_out = self.model._dream_head_outputs[l]

            layer_score = self._compute_layer_score(l, attn_out, head_out)
            W += layer_score * (self.a ** l)

            if early_stop and self.B_max is not None:
                mask = (W >= self.B_max) & (stop_layer == -1)
                stop_layer[mask] = l
                if (stop_layer != -1).all():
                    break

        is_adversarial = (self.B_max is not None) and (W >= self.B_max)
        return is_adversarial, W, stop_layer

    def calibrate_B_max(self, dataloader_clean):
        all_W = []
        for images, labels, _ in tqdm(dataloader_clean, desc="Calibrating B_max"):
            images = images.to(self.device)
            _, W, _ = self.detect(images, early_stop=False)
            all_W.extend(W.cpu().numpy().tolist())

        all_W = np.array(all_W)
        median_W = np.median(all_W)
        mad_W = np.median(np.abs(all_W - median_W))
        if mad_W < 1e-8:
            mad_W = 1e-8

        self.B_max = median_W + self.lambda_b * mad_W
        print(f"\nB_max calibrated: median_W={median_W:.4f}, MAD_W={mad_W:.4f}, B_max={self.B_max:.4f}")
        return self.B_max

    def evaluate(self, dataloader, is_adversarial=False):
        if self.B_max is None:
            raise RuntimeError("请先调用 calibrate_B_max() 校准 B_max")

        self.model.eval()
        all_preds = []
        all_scores = []

        pbar = tqdm(dataloader, desc="Detecting")
        for images, labels, _ in pbar:
            images = images.to(self.device)
            is_adv, W, _ = self.detect(images, early_stop=True)
            all_preds.extend(is_adv.cpu().numpy().tolist())
            all_scores.extend(W.cpu().numpy().tolist())

        all_preds = np.array(all_preds)
        all_scores = np.array(all_scores)

        if is_adversarial:
            tpr = np.mean(all_preds == 1)
            print(f"TPR (True Positive Rate): {tpr:.4f}")
            return {'tpr': tpr, 'scores': all_scores}
        else:
            fpr = np.mean(all_preds == 1)
            print(f"FPR (False Positive Rate): {fpr:.4f}")
            return {'fpr': fpr, 'scores': all_scores}


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    local_weights = os.path.join(PROJECT_ROOT, 'weights', 'vit_b_16-c867db91.pth')
    if os.path.exists(local_weights):
        model = ViTWrapper(num_classes=10, weights_path=local_weights).to(device)
        print(f"Loaded weights from: {local_weights}")
    else:
        model = ViTWrapper(num_classes=10, weights='imagenet').to(device)

    dist_path = os.path.join(PROJECT_ROOT, 'outputs', 'distributions', 'k3_distributions.json')
    topk_path = os.path.join(PROJECT_ROOT, 'outputs', 'topk_indices', 'k3.json')

    detector = DReaMDetector(
        model=model,
        distributions_path=dist_path,
        topk_path=topk_path,
        a=1.5,
        lambda_weight=0.5,
        lambda_b=3.0,
        K=3,
        device=device
    )

    from src.data.cifar_loader import get_cifar10_loaders, get_adversarial_loader

    val_loader, _ = get_cifar10_loaders(batch_size=256, num_workers=0)

    print(f"\n{'='*60}")
    print("Step 1: Calibrating B_max on clean validation set...")
    print(f"{'='*60}")
    detector.calibrate_B_max(val_loader)

    print(f"\n{'='*60}")
    print("Step 2: Evaluating on CLEAN validation set...")
    print(f"{'='*60}")
    clean_result = detector.evaluate(val_loader, is_adversarial=False)

    print(f"\n{'='*60}")
    print("Step 3: Evaluating on FGSM adversarial validation set...")
    print(f"{'='*60}")
    adv_loader = get_adversarial_loader(attack_name='fgsm', split='val', batch_size=256, num_workers=0)
    adv_result = detector.evaluate(adv_loader, is_adversarial=True)

    print(f"\n{'='*60}")
    print("Step 4: Evaluating on PGD-20 adversarial validation set...")
    print(f"{'='*60}")
    adv_loader_pgd = get_adversarial_loader(attack_name='pgd20', split='val', batch_size=256, num_workers=0)
    adv_result_pgd = detector.evaluate(adv_loader_pgd, is_adversarial=True)

    from src.evaluation.metrics import summarize_metrics

    print(f"\n{'='*60}")
    print("Computing final metrics (FGSM)...")
    print(f"{'='*60}")
    summary_fgsm = summarize_metrics(
        clean_scores=clean_result['scores'],
        adv_scores=adv_result['scores'],
        save_path=os.path.join(PROJECT_ROOT, 'outputs', 'metrics', 'k3_fgsm_val.json')
    )

    print(f"\n{'='*60}")
    print("Computing final metrics (PGD-20)...")
    print(f"{'='*60}")
    summary_pgd = summarize_metrics(
        clean_scores=clean_result['scores'],
        adv_scores=adv_result_pgd['scores'],
        save_path=os.path.join(PROJECT_ROOT, 'outputs', 'metrics', 'k3_pgd20_val.json')
    )

    print(f"\n{'='*60}")
    print("Detection test complete!")
    print(f"{'='*60}")