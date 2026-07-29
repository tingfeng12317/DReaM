import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTWrapper(nn.Module):
    def __init__(self, num_classes=10, weights='imagenet', weights_path=None):
        super().__init__()
        self.model = vit_b_16(weights=None)

        # 先替换分类头（num_classes 类），再加载权重
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)

        if weights_path and os.path.exists(weights_path):
            raw = torch.load(weights_path, map_location='cpu')
            source = os.path.basename(weights_path)
        elif weights == 'imagenet':
            raw = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).state_dict()
            source = 'IMAGENET1K_V1 (torchvision 官方缓存)'
        else:
            raise ValueError("[ViTWrapper] 必须提供 weights_path 或 weights='imagenet'")

        # 兼容带 'model.' 前缀的保存格式
        raw = {(k[6:] if k.startswith('model.') else k): v for k, v in raw.items()}

        # 只弹出类别数不匹配的分类头；其余所有键必须形状完全匹配
        head_w = raw.get('heads.head.weight')
        if head_w is not None and head_w.shape[0] != num_classes:
            raw.pop('heads.head.weight')
            raw.pop('heads.head.bias', None)
            print(f"[ViTWrapper] 源文件是 {head_w.shape[0]} 类头，已跳过；"
                  f"{num_classes} 类新头随机初始化待训练")

        missing, unexpected = self.model.load_state_dict(raw, strict=False)

        # 严格校验：除分类头外不允许任何缺失/多余/形状不符
        extra_missing = [k for k in missing if not k.startswith('heads.head.')]
        if extra_missing or unexpected:
            raise RuntimeError(
                f"\n[ViTWrapper] 权重文件与 vit_b_16 架构不匹配！\n"
                f"  missing (除头以外): {extra_missing}\n"
                f"  unexpected: {unexpected}\n"
                f"  → 这个 .pth 文件已损坏或非官方版本，请删除后改用 "
                f"weights='imagenet' 让 torchvision 重新下载官方权重。"
            )

        # 打印一层 MLP 的统计量，肉眼验证加载的是真实预训练权重
        w = self.model.encoder.layers[0].mlp[0].weight
        print(f"[ViTWrapper] 已加载 {source}，主干权重完整 "
              f"(layer0 mlp.0.weight: mean={w.mean():.6f}, std={w.std():.6f})")

        self.num_layers = len(self.model.encoder.layers)
        self.num_heads = 12
        self.head_dim = 64
        self.embed_dim = 768
        self.head_outputs = []
        self._dream_head_outputs = []
        self._attn_outputs = []
        self._layer_outputs = []  # 供 ReAct 等基线读取每层 block 输出

    def _extract_attention_heads(self, layer, x):
        mha = layer.self_attention
        B, N, C = x.shape
        if mha._qkv_same_embed_dim:
            qkv = nn.functional.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        else:
            q = nn.functional.linear(x, mha.q_proj_weight, mha.in_proj_bias[:C])
            k = nn.functional.linear(x, mha.k_proj_weight, mha.in_proj_bias[C:2*C])
            v = nn.functional.linear(x, mha.v_proj_weight, mha.in_proj_bias[2*C:])
            qkv = torch.cat([q, k, v], dim=-1)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        if mha.dropout > 0:
            attn = nn.functional.dropout(attn, p=mha.dropout, training=mha.training)

        head_output = attn @ v

        x = head_output.transpose(1, 2).reshape(B, N, C)
        x = nn.functional.linear(x, mha.out_proj.weight, mha.out_proj.bias)
        return x, head_output

    def forward(self, x, collect_for_dream=False):
        self.head_outputs = []
        self._dream_head_outputs = []
        self._attn_outputs = []
        self._layer_outputs = []

        x = self.model.conv_proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.model.encoder.pos_embedding
        x = self.model.encoder.dropout(x)

        for i, layer in enumerate(self.model.encoder.layers):
            x_norm = layer.ln_1(x)
            attn_out, head_output = self._extract_attention_heads(layer, x_norm)

            if collect_for_dream:
                self._attn_outputs.append(attn_out.detach())
                self._dream_head_outputs.append(head_output.detach())
            else:
                if head_output.requires_grad:
                    head_output.retain_grad()
                self._dream_head_outputs.append(head_output)

            x = x + layer.dropout(attn_out)
            x = x + layer.mlp(layer.ln_2(x))
            self._layer_outputs.append(x.detach().clone())

        x = self.model.encoder.ln(x)
        x = x[:, 0]
        x = self.model.heads(x)
        return x

    @torch.no_grad()
    def forward_features(self, x):
        """提取分类头之前的 768 维特征（供 linear probe 训练分类头用）。"""
        x = self.model.conv_proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.model.encoder.pos_embedding
        x = self.model.encoder.dropout(x)
        x = self.model.encoder.layers(x)
        x = self.model.encoder.ln(x)
        return x[:, 0]

    def get_attention_heads(self):
        return self.head_outputs


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViTWrapper(num_classes=10, weights='imagenet').to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    out = model(x)
    print(f"Output: {out.shape}, features: {model.forward_features(x).shape}")
    print("All checks passed!")