import torch
import torch.nn.functional as F


class PGDAttack:
    """
    PGD (Projected Gradient Descent) 攻击。
    在 [0,1] 像素空间执行，攻击完成后重新标准化。
    """
    def __init__(self, model, epsilon=8/255, alpha=2/255, steps=20,
                 random_start=True, norm='linf'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.norm = norm

        # ImageNet 标准化参数
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _denormalize(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def _normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def attack(self, images, labels):
        """
        Args:
            images: [B, 3, 224, 224], 已 ImageNet 标准化
            labels: [B]
        Returns:
            adv_images: [B, 3, 224, 224], 已 ImageNet 标准化
        """
        images = images.clone().detach()
        labels = labels.clone().detach()

        # 转换到 [0,1] 空间
        x = self._denormalize(images)

        if self.random_start:
            delta = torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + delta, 0, 1).detach()
        else:
            x_adv = x.clone().detach()

        for _ in range(self.steps):
            x_adv.requires_grad = True
            x_adv_norm = self._normalize(x_adv)
            outputs = self.model(x_adv_norm)
            loss = F.cross_entropy(outputs, labels)

            grad = torch.autograd.grad(loss, x_adv)[0]

            if self.norm == 'linf':
                x_adv = x_adv.detach() + self.alpha * grad.sign()
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1).detach()
            else:
                raise NotImplementedError("Only L-inf norm is supported currently.")

        return self._normalize(x_adv)


class FGSMAttack:
    """
    FGSM (Fast Gradient Sign Method) 单步攻击。
    """
    def __init__(self, model, epsilon=8/255):
        self.model = model
        self.epsilon = epsilon

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _denormalize(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def _normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def attack(self, images, labels):
        images = images.clone().detach()
        labels = labels.clone().detach()

        x = self._denormalize(images)
        x.requires_grad = True
        x_norm = self._normalize(x)
        outputs = self.model(x_norm)
        loss = F.cross_entropy(outputs, labels)

        grad = torch.autograd.grad(loss, x)[0]
        x_adv = x + self.epsilon * grad.sign()
        x_adv = torch.clamp(x_adv, 0, 1).detach()

        return self._normalize(x_adv)


if __name__ == '__main__':
    import sys
    import os
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, PROJECT_ROOT)

    from src.models.vit import ViTWrapper
    from src.data.cifar_loader import get_cifar10_loaders

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = ViTWrapper(num_classes=10, weights='imagenet').to(device)
    model.eval()

    _, test_loader = get_cifar10_loaders(batch_size=8, num_workers=0)
    images, labels, _ = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    print(f"Original labels: {labels.tolist()}")
    with torch.no_grad():
        orig_pred = model(images).argmax(dim=1)
    print(f"Original preds:  {orig_pred.tolist()}")

    # FGSM
    fgsm = FGSMAttack(model, epsilon=8/255)
    adv_fgsm = fgsm.attack(images, labels)
    with torch.no_grad():
        pred_fgsm = model(adv_fgsm).argmax(dim=1)
    print(f"FGSM preds:      {pred_fgsm.tolist()}")

    # PGD-20
    pgd = PGDAttack(model, epsilon=8/255, alpha=2/255, steps=20)
    adv_pgd = pgd.attack(images, labels)
    with torch.no_grad():
        pred_pgd = model(adv_pgd).argmax(dim=1)
    print(f"PGD-20 preds:    {pred_pgd.tolist()}")

    # 验证对抗扰动大小
    diff = (adv_pgd - images).abs().max().item()
    print(f"Max perturbation (normalized space): {diff:.4f}")