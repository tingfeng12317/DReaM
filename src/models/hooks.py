import torch
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class GradientCollector:
    def __init__(self, model):
        self.model = model

    def collect(self):
        if not hasattr(self.model, '_dream_head_outputs') or len(self.model._dream_head_outputs) == 0:
            raise RuntimeError(
                "ViTWrapper._dream_head_outputs 为空。请确认：\n"
                "1. 已按说明修改 vit.py\n"
                "2. 已执行 forward(x)"
            )

        grads = []
        for head_output in self.model._dream_head_outputs:
            if head_output.grad is not None:
                grads.append(head_output.grad.clone())
            else:
                grads.append(torch.zeros_like(head_output))
        return grads

    def compute_fisher(self, grads=None):
        if grads is None:
            grads = self.collect()

        fisher_info = []
        for grad in grads:
            l2_per_head = torch.norm(grad, p=2, dim=(2, 3))
            fisher = l2_per_head.mean(dim=0)
            fisher_info.append(fisher)

        return fisher_info


if __name__ == '__main__':
    from vit import ViTWrapper
    import torch.nn as nn

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 优先从本地 weights 目录加载，避免网络下载
    local_weights = os.path.join(PROJECT_ROOT, 'weights', 'vit_b_16-c867db91.pth')
    if os.path.exists(local_weights):
        model = ViTWrapper(num_classes=10, weights_path=local_weights).to(device)
        print(f"Loaded weights from: {local_weights}")
    else:
        model = ViTWrapper(num_classes=10, weights='imagenet').to(device)
        print("Loaded ImageNet-1K pretrained weights (downloaded).")

    collector = GradientCollector(model)

    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    dummy_labels = torch.randint(0, 10, (2,)).to(device)

    model.eval()
    model.zero_grad()
    dummy_input.requires_grad = True

    output = model(dummy_input)
    loss = nn.CrossEntropyLoss()(output, dummy_labels)
    loss.backward()

    grads = collector.collect()
    fisher = collector.compute_fisher(grads)

    print(f"\nGradient shapes:")
    for i, g in enumerate(grads):
        print(f"  Layer {i+1:2d}: grad={g.shape}")

    print(f"\nFisher information shapes:")
    for i, f in enumerate(fisher):
        print(f"  Layer {i+1:2d}: fisher={f.shape}, values={f[:3].tolist()}")

    print("\nAll checks passed!")