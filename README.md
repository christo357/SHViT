# SHViT Analysis: Comprehensive Model Evaluation and Comparison

This repository is built upon the official implementation of [**SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design**](https://arxiv.org/abs/2401.16456) (Seokju Yun, Youngmin Ro. CVPR 2024). 

This work presents a comprehensive analysis of the SHViT model, comparing it with baseline models (MobileNetV2 and DeiT-Tiny) across various dimensions including robustness, data efficiency, representation similarity, and geometric invariance.

**Contributors:** [Vishal V](https://github.com/VizalV) • [Priyal Garg](https://github.com/gargpriyal)

> *Note: If the GitHub usernames above are incorrect, please provide the correct ones to update the links.*

---

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
python analyze_learning_curve.py \
  --results-dir results/ \
  --dataset CIFAR
```


### `analyze_robustness_vs_data.py`
Evaluates how model robustness to corruptions scales with training data size.

**Example Usage:**
```bash
python analyze_robustness_vs_data.py \
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
python analyze_geometric_invariance.py \
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
python analyze_domain_shift.py \
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
python analyze_rep_similarity.py \
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
python analyze_representations.py \
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
python analyze_gradcam_compare.py \
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
python analyze_patchify_stride.py \
  --model shvit_s2 \
  --datasets CIFAR EUROSAT MEDMNIST \
  --checkpoint-dir stride_experiments \
  --strides 8 16 32
```

## Original SHViT

<details>
  <summary>
  <font size="+1">About SHViT</font>
  </summary>

[**SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design**](https://arxiv.org/abs/2401.16456)  
*Seokju Yun, Youngmin Ro.* CVPR 2024

Recently, efficient Vision Transformers have shown great performance with low latency on resource-constrained devices. Conventionally, they use 4x4 patch embeddings and a 4-stage structure at the macro level, while utilizing sophisticated attention with multi-head configuration at the micro level. This paper aims to address computational redundancy at all design levels in a memory-efficient manner. We discover that using larger-stride patchify stem not only reduces memory access costs but also achieves competitive performance by leveraging token representations with reduced spatial redundancy from the early stages. Furthermore, our preliminary analyses suggest that attention layers in the early stages can be substituted with convolutions, and several attention heads in the latter stages are computationally redundant. To handle this, we introduce a single-head attention module that inherently prevents head redundancy and simultaneously boosts accuracy by parallelly combining global and local information. Building upon our solutions, we introduce SHViT, a Single-Head Vision Transformer that obtains the state-of-the-art speed-accuracy tradeoff. For example, on ImageNet-1k, our SHViT-S4 is 3.3x, 8.1x, and 2.4x faster than MobileViTv2 x1.0 on GPU, CPU, and iPhone12 mobile device, respectively, while being 1.3% more accurate. For object detection and instance segmentation on MS COCO using Mask-RCNN head, our model achieves performance comparable to FastViT-SA12 while exhibiting 3.8x and 2.0x lower backbone latency on GPU and mobile device, respectively.

</details>

### Pre-trained Models
| name | resolution | acc | #params | FLOPs | Throughput | model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| SHViT-S1 | 224x224 | 72.8 | 6.3M | 241M | 33489 |[model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s1.pth) |
| SHViT-S2 | 224x224 | 75.2 | 11.4M | 366M | 26878 | [model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s2.pth) |
| SHViT-S3 | 224x224 | 77.4 | 14.2M | 601M | 20522 | [model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s3.pth) |
| SHViT-S4 | 256x256 | 79.4 | 16.5M | 986M | 14283 | [model](https://github.com/ysj9909/SHViT/releases/download/v1.0/shvit_s4.pth) |

---

## Setup

### Environment Setup
```bash
conda create -n shvit python=3.9
conda activate shvit
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  validation/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

### Training

To train SHViT models, follow the respective command below:

<details>
<summary>
SHViT-S1
</summary>

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py \
  --model shvit_s1 \
  --data-path $PATH_TO_IMAGENET \
  --dist-eval \
  --weight-decay 0.025
```
</details>

<details>
<summary>
SHViT-S2
</summary>

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py \
  --model shvit_s2 \
  --data-path $PATH_TO_IMAGENET \
  --dist-eval \
  --weight-decay 0.032
```
</details>

<details>
<summary>
SHViT-S3
</summary>

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py \
  --model shvit_s3 \
  --data-path $PATH_TO_IMAGENET \
  --dist-eval \
  --weight-decay 0.035
```
</details>

<details>
<summary>
SHViT-S4
</summary>

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 --use_env main.py \
  --model shvit_s4 \
  --data-path $PATH_TO_IMAGENET \
  --dist-eval \
  --weight-decay 0.03 \
  --input-size 256
```
</details>

### Evaluation
Run the following command to evaluate a pre-trained SHViT-S4 on ImageNet-1K validation set:
```bash
python main.py \
  --eval \
  --model shvit_s4 \
  --resume ./shvit_s4.pth \
  --data-path $PATH_TO_IMAGENET \
  --input-size 256
```

### Latency Measurement
Compare throughputs on GPU/CPU:
```bash
python speed_test.py
```

For mobile latency (iPhone 12), use [XCode 14](https://developer.apple.com/videos/play/wwdc2022/10027/) deployment tools. Export the model to Core ML format:
```bash
python export_model.py \
  --variant shvit_s4 \
  --output-dir /path/to/save/exported_model \
  --checkpoint /path/to/pretrained_checkpoints/shvit_s4.pth
```

---

## Citation

If you use this work or code in your research, please cite:

```bibtex
@inproceedings{yun2024shvit,
  author={Yun, Seokju and Ro, Youngmin},
  title={SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={5756--5767},
  year={2024}
}
```

---

## Acknowledgements

We sincerely appreciate:
- [SHViT Official Repository](https://github.com/ysj9909/SHViT) - The foundation of this work
- [Swin Transformer](https://github.com/microsoft/swin-transformer)
- [LeViT](https://github.com/facebookresearch/LeViT)
- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT)
- [PyTorch](https://github.com/pytorch/pytorch)

---

## License

This project follows the same license as the original SHViT repository. See [LICENSE](LICENSE) for details.
