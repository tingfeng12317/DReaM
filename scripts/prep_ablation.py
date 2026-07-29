"""消融实验准备脚本：校验 top-3 前缀 + 切片 k1/k6 + 生成随机头 JSON
位置：项目根/scripts/prep_ablation.py
运行：python scripts/prep_ablation.py
"""
import os
import json
import numpy as np

# 从脚本自身位置定位项目根（script/ 的上一级），不写死绝对路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
base = os.path.join(PROJECT_ROOT, 'outputs', 'cifar10', 'topk_indices')

with open(os.path.join(base, 'k12.json')) as f:
    k12 = json.load(f)
with open(os.path.join(base, 'k3.json')) as f:
    k3 = json.load(f)

# 1. 校验：k12 每层前3个头 == 当前 k3.json
mismatch = [l for l in range(12) if k12[str(l)][:3] != k3[str(l)]]
print('top-3前缀校验:', '全部一致 ✓' if not mismatch else f'不一致的层: {mismatch} ✗')

# 2. 切片生成 k1/k6
for K in [1, 6]:
    sub = {l: h[:K] for l, h in k12.items()}
    out = os.path.join(base, f'k{K}.json')
    with open(out, 'w') as f:
        json.dump(sub, f, indent=2)
    print(f'k{K}.json 已生成 -> {out}')

# 3. 生成3组随机头
for seed in [0, 1, 2]:
    rng = np.random.default_rng(seed)
    heads = {str(l): sorted(rng.choice(12, size=3, replace=False).tolist())
             for l in range(12)}
    out = os.path.join(base, f'k3_random_s{seed}.json')
    with open(out, 'w') as f:
        json.dump(heads, f, indent=2)
    print(f'k3_random_s{seed}.json 已生成 -> {out}')
