# DReaM: Dual-Distribution Residual-stream Monitoring for Training-Free Adversarial Detection in Vision Transformers

Official implementation of the paper:

> **DReaM: Dual-Distribution Residual-stream Monitoring for Training-Free Adversarial Detection in Vision Transformers**
> Haijing Sun, Shuyu Jin, Yichuan Shao, Zhitao Zhang
> *Submitted to Pattern Recognition*

## Overview

DReaM is a **training-free** adversarial detection framework for Vision Transformers (ViTs). It requires only a single offline analysis after model training:

1. **Pass 1 — Fisher head screening**: selects the most loss-sensitive attention heads per layer using a Fisher-information-inspired importance score (one backward pass over the calibration set);
2. **Pass 2 — Dual-distribution baselines**: builds robust median + MAD baselines for both head-output activation magnitudes and attention residual increments (one gradient-free forward pass);
3. **Online detection**: accumulates depth-weighted layer-wise anomaly scores into a global score and declares a sample adversarial once the score crosses a calibrated threshold (monotonic early stopping).

No adversarial samples, no retraining, and no architectural modification are needed at any stage.

![Offline analysis pipeline](assets/fig_framework_offline_en.png)

![Online detection pipeline](assets/fig_framework_online_en.png)

## Key Results (CIFAR-10 test set, ViT-B/16)

| Attack | FPR | TPR | AUROC | FPR@95TPR |
|---|---|---|---|---|
| FGSM | 0.0407 | **1.0000** | 0.9933 | 0.0169 |
| PGD-20 | 0.0407 | **0.9969** | 0.9958 | 0.0159 |

Under PGD-20, logit-space baselines (MSP, Energy) collapse below random guessing (AUROC < 0.1), while DReaM remains stable. DReaM also generalizes to CIFAR-100 and ISIC 2018 without any architectural change.

## Repository Structure

```
├── configs/                  # YAML configs for all experiments
│   ├── cifar10.yaml          #   main experiment (K=3, tuned hyperparameters)
│   ├── cifar100.yaml         #   cross-dataset generalization
│   ├── isic2018.yaml         #   cross-domain generalization
│   ├── cifar10_k1.yaml       #   ablation: K=1 head per layer
│   ├── cifar10_k6.yaml       #   ablation: K=6 heads per layer
│   ├── cifar10_k12.yaml      #   ablation: full-network monitoring (K=12)
│   └── cifar10_random_s{0,1,2}.yaml  # ablation: random head selection, 3 seeds
├── scripts/
│   ├── main.py               # main pipeline entry (offline + online)
│   ├── eval_clean_acc.py     # clean accuracy of the linear-probe model
│   ├── prep_ablation.py      # prepare ablation runs
│   ├── extract_tuning.py     # extract hyperparameter tuning results
│   ├── plot_results.py       # reproduce paper figures
│   └── prepare_isic2018.py   # ISIC 2018 download/split helper
├── src/
│   ├── dream/                # core DReaM implementation
│   │   ├── profiler.py       #   Pass 1: Fisher head screening
│   │   ├── distribution.py   #   Pass 2: median+MAD baseline construction
│   │   └── detector.py       #   online layer-wise scoring and decision
│   ├── models/               # frozen ViT-B/16, linear head, forward hooks
│   ├── attacks/              # FGSM / PGD-20 adversarial sample generation
│   ├── baselines/            # Mahalanobis, ReAct, MSP, Energy detectors
│   ├── data/                 # dataset loaders (CIFAR-10/100, ISIC 2018)
│   ├── evaluation/           # metrics, test evaluation, tuning, ablation
│   └── utils/                # config handling
├── assets/                   # framework figures and baseline JSONs
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/tingfeng12317/DReaM.git
cd DReaM
pip install -r requirements.txt
```

Tested with Python 3.12 + PyTorch 2.12.0 on a single NVIDIA GPU (CUDA required; ~8 GB VRAM is sufficient).

## Data Preparation

**CIFAR-10 / CIFAR-100**: downloaded automatically on first run. The clean calibration set is the standard training split (50,000 images), following the standard training-free detection protocol.

**ISIC 2018** (Task 3): requires manual registration and download from the [official challenge site](https://challenge.isic-archive.com/data/#2018). Then run:

```bash
python scripts/prepare_isic2018.py --data_dir /path/to/isic2018
```

This reproduces the official split used in the paper: 10,015 training / 193 validation / 1,512 test images.

## Model Weights

The backbone is the torchvision ViT-B/16 with ImageNet-1K pretrained weights (downloaded automatically); only the linear classification head is trained (linear probe). Two options:

1. **Download our trained heads** from [Releases](https://github.com/tingfeng12317/DReaM/releases) and place them under `checkpoints/`;
2. **Train from scratch** (a few minutes on one GPU):

```bash
python scripts/eval_clean_acc.py --config configs/cifar10.yaml   # trains/evaluates the linear head
```

Expected clean test accuracies: CIFAR-10 94.25%, CIFAR-100 77.59%, ISIC 2018 62.50%.

## Quick Start

Run the full DReaM pipeline (Pass 1 → Pass 2 → threshold calibration → online detection on the test set):

```bash
python scripts/main.py --config configs/cifar10.yaml
```

- Pass 1 (Fisher screening) takes ~4 minutes for the 50,000-sample calibration set on a single RTX 5090 D;
- Pass 2 is a single gradient-free forward pass;
- Pre-computed screening indices (`topk.json`) and median+MAD baselines are provided in `assets/`, so you can skip the offline passes and run online detection directly.

Cross-dataset experiments:

```bash
python scripts/main.py --config configs/cifar100.yaml
python scripts/main.py --config configs/isic2018.yaml
```

## Reproducing the Paper

| Paper item | Command |
|---|---|
| Table 1 (main results) | `python scripts/main.py --config configs/cifar10.yaml` |
| Table 6 (head screening, K) | `python scripts/main.py --config configs/cifar10_k{1,6,12}.yaml` |
| Table 6 (random heads) | `python scripts/main.py --config configs/cifar10_random_s{0,1,2}.yaml` |
| Figures 3–7 | `python scripts/plot_results.py` |
| Hyperparameter tuning (Tables 3–5) | `python scripts/extract_tuning.py` |

All experiments use fixed random seeds for reproducibility.

## Acknowledgements

- The Mahalanobis baseline in `src/baselines/mahalanobis/` is adapted from the [official implementation](https://github.com/pokaxpoka/deep_Mahalanobis_detector) of Lee et al., *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks* (NeurIPS 2018).
- The ViT-B/16 backbone uses ImageNet-1K pretrained weights provided by [torchvision](https://pytorch.org/vision/stable/models.html).

## Citation

If you find this work useful, please cite:

```bibtex
@article{dream2026,
  title   = {DReaM: Dual-Distribution Residual-stream Monitoring for Training-Free Adversarial Detection in Vision Transformers},
  author  = {Sun, Haijing and Jin, Shuyu and Shao, Yichuan and Zhang, Zhitao},
  journal = {Pattern Recognition (submitted)},
  year    = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
