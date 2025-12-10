# SHViT Analysis: Comprehensive Model Evaluation and Comparison

## 🤗 Pre-trained Models on Hugging Face

Fine-tuned **SHViT** models are available on Hugging Face Hub for easy use and reproducibility:

| Model | Dataset | Accuracy | HuggingFace |
|-------|---------|----------|-------------|
| SHViT-S2 | CIFAR-100 | 72.64% | [🤗 christo357/shvit-s2-cifar](https://huggingface.co/christo357/shvit-s2-cifar) |
| SHViT-S2 | EuroSAT | 93.83% | [🤗 christo357/shvit-s2-eurosat](https://huggingface.co/christo357/shvit-s2-eurosat) |
| SHViT-S2 | MedMNIST (PathMNIST) | 98.05% | [🤗 christo357/shvit-s2-medmnist](https://huggingface.co/christo357/shvit-s2-medmnist) |
| SHViT-S3 | CIFAR-100 | 68.59% | [🤗 christo357/shvit-s3-cifar](https://huggingface.co/christo357/shvit-s3-cifar) |
| SHViT-S3 | EuroSAT | 94.52% | [🤗 christo357/shvit-s3-eurosat](https://huggingface.co/christo357/shvit-s3-eurosat) |
| SHViT-S3 | MedMNIST (PathMNIST) | 98.04% | [🤗 christo357/shvit-s3-medmnist](https://huggingface.co/christo357/shvit-s3-medmnist) |

### Quick Start

```python
from huggingface_hub import hf_hub_download
import torch
from timm import create_model

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="christo357/shvit-s2-cifar",
    filename="checkpoint_99.pth"
)

# Load model
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model = create_model('shvit_s2', num_classes=100, pretrained=False)
model.load_state_dict(checkpoint['model'])
model.eval()
```

---

## Table of Contents
- [Pre-trained Models](#-pre-trained-models-on-hugging-face)
- [Overview](#overview)
- [Analysis Scripts](#analysis-scripts)
- [Original SHViT](#original-shvit)
- [Setup](#setup)
- [Citation](#citation)

---

## Overview

This repository extends the official SHViT implementation with extensive analysis tools to evaluate and compare vision transformer architectures. All analysis scripts (prefixed with `analyze_`) provide comprehensive model comparisons across multiple datasets (CIFAR-100, EuroSAT, MedMNIST).

**Key Features:**
- Multi-model comparison framework (SHViT variants, DeiT-Tiny, MobileNetV2)
- Robustness evaluation under various corruptions
- Data efficiency and learning curve analysis
- Representation similarity metrics (CKA, CCA)
- Geometric invariance testing
- Domain adaptation and transfer learning analysis

---

## Analysis Scripts

### `analyze_learning_curve.py`
Analyzes model performance across different training data fractions to evaluate data efficiency.

**Example Usage:**
```bash
python extensions/analysis/analyze_learning_curve.py \
  --results-dir results/ \
  --dataset CIFAR
```


### `analyze_robustness_vs_data.py`
Evaluates how model robustness to corruptions scales with training data size.

**Example Usage:**
```bash
python extensions/analysis/analyze_robustness_vs_data.py \
  --dataset CIFAR \
  --data-path dataset \
  --checkpoint-dir results \
  --models shvit_s2 deit_tiny_patch16_224 mobilenetv2_100 \
  --device cuda
```


### `analyze_geometric_invariance.py`
Tests model robustness to geometric transformations and color variations.

**Example Usage:**
```bash
python extensions/analysis/analyze_geometric_invariance.py \
  --dataset CIFAR \
  --data-path dataset/ \
  --checkpoint-dir results \
  --models shvit_s2 deit_tiny_patch16_224 mobilenetv2_100 \
  --device cuda
```
### `analyze_domain_shift.py`
Evaluates model transferability across different domains through fine-tuning experiments.

**Example Usage:**
```bash
python extensions/analysis/analyze_domain_shift.py \
  --source-dataset CIFAR \
  --target-dataset EUROSAT \
  --models shvit_s2 deit_tiny_patch16_224 mobilenetv2_100 \
  --data-path dataset \
  --checkpoint-dir results \
  --ft-epochs 10 \
  --device cuda
```

### `analyze_rep_similarity.py`
Measures representation similarity between SHViT and baseline models using multiple metrics.


**Example Usage:**
```bash
python extensions/analysis/analyze_rep_similarity.py \
  --model-a shvit_s2 \
  --ckpt-a results/shvit_s2_CIFAR_frac1.0/checkpoint_99.pth \
  --model-b deit_tiny_patch16_224 \
  --ckpt-b results/deit_tiny_patch16_224_CIFAR_frac1.0/checkpoint_99.pth \
  --dataset CIFAR \
  --data-path dataset/ \
  --output-dir outputs/rep_similarity
```

### `analyze_representations.py`
Comprehensive layer-wise representation analysis comparing SHViT and DeiT architectures.

**Example Usage:**
```bash
python extensions/analysis/analyze_representations.py \
  --shvit-checkpoint results/shvit_s2_CIFAR_frac1.0/checkpoint_99.pth \
  --deit-checkpoint results/deit_tiny_patch16_224_CIFAR_frac1.0/checkpoint_99.pth \
  --shvit-model shvit_s2 \
  --deit-model deit_tiny_patch16_224 \
  --data-path dataset/ \
  --data-set CIFAR \
  --nb-classes 100 \
  --output-dir representation_analysis/
```

### `analyze_gradcam_compare.py`
Qualitative saliency map comparison to understand model attention patterns.


**Example Usage:**
```bash
python extensions/analysis/analyze_gradcam_compare.py \
  --model-a shvit_s2 \
  --ckpt-a results/shvit_s2_CIFAR_frac1.0/checkpoint_99.pth \
  --model-b deit_tiny_patch16_224 \
  --ckpt-b results/deit_tiny_patch16_224_CIFAR_frac1.0/checkpoint_99.pth \
  --dataset CIFAR \
  --data-path dataset/ \
  --severity 3 \
  --max-search 300 \
  --output-dir outputs/saliency
```

### `analyze_patchify_stride.py`
Investigates how patchify stride affects domain generalization and performance.

**Example Usage:**
```bash
python extensions/analysis/analyze_patchify_stride.py \
  --model shvit_s2 \
  --datasets CIFAR EUROSAT MEDMNIST \
  --checkpoint-dir stride_experiments \
  --strides 8 16 32
```