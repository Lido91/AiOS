# AiOS Training Methodology

## Overview

AiOS is a whole-body human mesh recovery model that uses a transformer-based architecture (Deformable DETR) for end-to-end SMPL-X estimation. This document provides a comprehensive overview of how the model has been trained.

---

## Table of Contents

- [Training Datasets](#training-datasets)
- [Model Architecture](#model-architecture)
- [Training Hyperparameters](#training-hyperparameters)
- [Loss Functions](#loss-functions)
- [Data Augmentation](#data-augmentation)
- [Training Strategy](#training-strategy)
- [Initialization & Checkpointing](#initialization--checkpointing)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Training Files](#key-training-files)

---

## Training Datasets

The model is trained on **6 major datasets** with a balanced sampling strategy:

| Dataset | Partition | Description |
|---------|-----------|-------------|
| **AGORA_MM** | 20% | Multi-person 3D human pose dataset |
| **BEDLAM** | 40% | Synthetic dataset with SMPL-X annotations |
| **COCO_NA** | 60% | COCO with natural annotations |
| **UBody_MM** | 80% | Upper body focus dataset |
| **EgoBody_Egocentric** | 90% | Egocentric view dataset |
| **ARCTIC** | 100% | Hand-object interaction dataset |

### Data Format

All datasets are processed into **HumanData format** (NPZ files) with SMPL-X annotations including:
- 2D keypoints (137 keypoints in SMPL-X format)
- 3D keypoints
- SMPL-X parameters (pose, shape, expression)
- Bounding boxes for body, hands, and face

### Dataset-Specific Settings

**Person Sampling Probabilities:**
- **AGORA_MM:** 70% probability, 50% sample ratio
- **BEDLAM:** 70% probability, 60% sample ratio
- **COCO_NA:** 70% probability, 60% sample ratio

**Sample Intervals:**
- **BEDLAM:** Interval of 5 (processes every 5th frame)
- Other datasets use standard sampling

---

## Model Architecture

### Backbone

- **Architecture:** ResNet-50 (other backbones supported)
- **Multi-scale features:** 3 intermediate layers (indices [1, 2, 3])
- **Normalization:** Frozen BatchNorm2d for training stability

### Transformer Architecture (Deformable DETR-based)

| Component | Value |
|-----------|-------|
| Encoder layers | 6 |
| Decoder layers | 6 (2 box + 4 hand/face) |
| Hidden dimension | 256 |
| Number of heads | 8 |
| Feature levels | 4 (multi-scale) |
| Feed-forward dimension | 2048 |
| Dropout | 0.0 |

### Query-based Detection

- **Number of queries:** 900
- **Query dimension:** 4
- **Number of groups:** 100 (for grouping queries)

### Progressive Refinement Strategy

1. **Body localization stage** (first 2 decoder layers)
2. **Body refinement + hand/face localization** (layers 3-6)
3. **Whole-body refinement** with SMPL-X regression

### SMPL-X Body Model

- **Shape coefficients (beta):** 10
- **Expression coefficients:** 10
- **Pose parameters:**
  - Root: 3 (global orientation)
  - Body: 21 × 3 = 63 (joint rotations)
  - Hands: 2 × 15 × 3 = 90 (left + right)
  - Jaw: 3
- **Model path:** `data/body_models/smplx`

---

## Training Hyperparameters

### Optimizer Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Base learning rate | 1.414e-5 (0.0001 × 1.414/10) |
| Backbone learning rate | 1.414e-6 (10x smaller) |
| Linear projection LR multiplier | 0.1 |
| Weight decay | 0.0001 |

### Training Schedule

| Parameter | Value |
|-----------|-------|
| Total epochs | 200 |
| Batch size per GPU | 2 |
| Number of GPUs | 8 |
| Effective batch size | 16 |
| LR scheduler | Multi-step LR |
| LR drop epochs | [30, 60] |
| Gradient clipping | max_norm = 0.1 |
| Mixed precision (AMP) | Supported |

### Exponential Moving Average (EMA)

- **Enabled:** Yes
- **Decay:** 0.9997
- **Start epoch:** 0

### Denoising Training (DN-DETR)

| Parameter | Value |
|-----------|-------|
| DN number | 100 |
| Box noise scale | 0.4 |
| Label noise ratio | 0.5 |
| DN labelbook size | 100 |

---

## Loss Functions

The training uses **multiple loss components** with carefully tuned weights (all scaled by 0.1):

### Classification Loss

- **Type:** Focal loss
- **Alpha:** 0.25
- **Gamma:** 2.0
- **Weight:** 2.0

### Bounding Box Losses

Applied separately for body, left hand, right hand, and face:

| Loss Type | Weight |
|-----------|--------|
| L1 loss (bbox coordinates) | 5.0 |
| GIoU loss (bbox overlap) | 2.0 |

### 2D Keypoint Losses

| Component | Body | Hands | Face |
|-----------|------|-------|------|
| L1 loss | 10.0 | 5.0 | 1.0 |
| OKS (Object Keypoint Similarity) | 4.0 | 0.5 | 4.0 |

### SMPL-X Parameter Losses

**Pose Loss (L1 on rotation matrices):**

| Component | Weight |
|-----------|--------|
| Root pose | 1.0 |
| Body pose | 0.1 |
| Left hand pose | 0.1 |
| Right hand pose | 0.1 |
| Jaw pose | 0.1 |

**Shape and Expression:**

| Component | Weight |
|-----------|--------|
| Beta (shape) | 0.01 |
| Expression | 0.01 |

### 3D Keypoint Losses

**L1 Loss on 3D keypoints:**

| Component | Weight |
|-----------|--------|
| Body 3D | 1.0 |
| Face 3D | 0.1 |
| Left hand 3D | 0.1 |
| Right hand 3D | 0.1 |

**Root-aligned 3D keypoint loss:**

| Component | Weight |
|-----------|--------|
| Body 3D (root-aligned) | 1.0 |
| Face 3D (root-aligned) | 0.1 |
| Left hand 3D (root-aligned) | 0.1 |
| Right hand 3D (root-aligned) | 0.1 |

### Auxiliary Losses

- **Auxiliary loss enabled:** Yes (all decoder layers contribute)
- **Intermediate loss coefficient:** 1.0
- **Encoder loss coefficient:** 1.0

### Special Loss Weighting

- Hand and face losses at decoder layer 4 are reduced by 10x during training
- This helps progressive refinement strategy

---

## Data Augmentation

### Geometric Augmentations

| Augmentation | Value |
|--------------|-------|
| Scale jittering | ±25% |
| Rotation | 0° (disabled in training) |
| Horizontal flip | Random |
| Crop factor | 0.1 |

### Color Augmentation

| Augmentation | Value |
|--------------|-------|
| Color factor | 0.2 (random color scale) |

### Multi-scale Training

- **Training sizes:** [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
- **Max size:** 1333
- **Strategy:** Random size selection per batch

### Normalization

- **Mean:** [123.675, 116.28, 103.53]
- **Std:** [58.395, 57.12, 57.375]

### Person Sampling

Creates crops with variable numbers of people (1 to N) for multi-person scenarios:
- Helps the model learn to handle crowded scenes
- Dataset-specific sampling probabilities and ratios
- Balanced representation of single-person and multi-person cases

---

## Training Strategy

### Multi-task Learning

The model simultaneously learns:
1. **Human detection** - Locating people in the image
2. **2D pose estimation** - Predicting 2D keypoint locations
3. **3D pose estimation** - Estimating 3D joint positions
4. **SMPL-X parameter regression** - Generating body mesh parameters
5. **Hand and face localization** - Detailed extremity detection

### Progressive Training

- **Body-first approach:** Early decoder layers focus on body detection and coarse pose
- **Hand/face refinement:** Later layers add detailed hand and face estimation
- **All-in-one-stage inference:** No separate detection step needed at inference time

### Matcher (Hungarian Algorithm)

Query-to-target assignment costs:

| Cost Component | Weight |
|----------------|--------|
| Classification | 2.0 |
| BBox L1 | 5.0 |
| GIoU | 2.0 |
| Keypoints | 10.0 |
| OKS | 4.0 |

### Data Loading Strategy

- **Pipeline:** MMHuman3D-style data pipeline
- **Caching:** Preprocessed data cached as NPZ files
- **Balanced sampling:** Maintains balance across different datasets
- **Workers per GPU:** 0 for training (uses main thread)

---

## Initialization & Checkpointing

### Pretrained Initialization

- **Starting checkpoint:** ED-Pose ResNet-50 trained on COCO
- **Path:** `./data/checkpoint/edpose_r50_coco.pth`
- **Strategy:** Allows ignoring specific layers via `finetune_ignore` parameter

### Checkpointing Strategy

**Regular checkpoints:**
- Saves `checkpoint.pth` every epoch
- Contains: model weights, optimizer state, LR scheduler state, epoch number

**Additional checkpoints saved at:**
- LR drop epochs (30, 60)
- Every `save_checkpoint_interval` epochs

**Checkpoint contents:**
- Model state dict
- Optimizer state dict
- LR scheduler state dict
- EMA model state dict (if EMA enabled)
- Current epoch number
- Training arguments

### Resume Training

- Automatically resumes from `checkpoint.pth` if exists in output directory
- Can manually specify checkpoint via `--resume` argument
- Supports resuming from URL checkpoints

---

## Evaluation Metrics

The model is evaluated on standard 3D human pose and shape metrics:

### Primary Metrics

| Metric | Description |
|--------|-------------|
| **PA-MPVPE** | Procrustes-Aligned Mean Per-Vertex Position Error |
| **MPVPE** | Mean Per-Vertex Position Error |
| **PA-MPJPE** | Procrustes-Aligned Mean Per-Joint Position Error |

### Component-wise Evaluation

Metrics are computed separately for:
- Full body
- Left hand
- Right hand
- Face

### Benchmark Datasets

- **AGORA:** Challenging multi-person benchmark
- **BEDLAM:** Synthetic data with ground truth SMPL-X
- Other standard 3D human pose benchmarks

---

## Key Training Files

### Main Scripts

| File | Purpose |
|------|---------|
| [`main.py`](main.py) | Main training script entry point |
| [`scripts/train.sh`](scripts/train.sh) | Training launch script |
| [`engine.py`](engine.py) | Training loop implementation |

### Configuration

| File | Purpose |
|------|---------|
| [`config/aios_smplx_train.py`](config/aios_smplx_train.py) | Main training configuration |
| [`config/config.py`](config/config.py) | Base config utilities |

### Model Definition

| File | Purpose |
|------|---------|
| [`models/aios/aios_smplx.py`](models/aios/aios_smplx.py) | Main model architecture |
| [`models/aios/criterion_smplx.py`](models/aios/criterion_smplx.py) | Loss function definitions |
| [`models/aios/deformable_detr_smplx.py`](models/aios/deformable_detr_smplx.py) | Deformable DETR implementation |

### Data Loading

| File | Purpose |
|------|---------|
| [`datasets/humandata.py`](datasets/humandata.py) | HumanData dataset loader |
| [`datasets/dataset.py`](datasets/dataset.py) | Multi-dataset wrapper |
| [`util/preprocessing.py`](util/preprocessing.py) | Data augmentation pipeline |

### Utilities

| File | Purpose |
|------|---------|
| [`util/get_param_dicts.py`](util/get_param_dicts.py) | Parameter grouping for optimizer |
| [`util/utils.py`](util/utils.py) | EMA and other utilities |
| [`util/logger.py`](util/logger.py) | Logging setup |

---

## Training Command Example

```bash
# 8 GPU distributed training
bash scripts/train.sh

# Typical command structure:
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --use_env main.py \
    --config_file config/aios_smplx_train.py \
    --output_dir output/aios_smplx_train \
    --batch_size 2 \
    --num_workers 0
```

---

## Performance

With this training methodology, AiOS achieves state-of-the-art results on:
- **AGORA benchmark:** Whole-body mesh recovery in challenging multi-person scenes
- **BEDLAM benchmark:** Accurate SMPL-X parameter estimation
- **Other benchmarks:** Competitive performance on standard 3D human pose datasets

The progressive refinement strategy and multi-task learning approach enable the model to produce accurate whole-body meshes including detailed hand and face estimation.

---

## References

- **SMPL-X:** Expressive Body Capture: 3D Hands, Face, and Body from a Single Image
- **Deformable DETR:** Deformable DETR: Deformable Transformers for End-to-End Object Detection
- **ED-Pose:** Explicit Box Detection Unifies End-to-End Multi-Person Pose Estimation
- **DN-DETR:** DN-DETR: Accelerate DETR Training by Introducing Query DeNoising

---

*This documentation was generated based on the AiOS codebase analysis. For the latest updates, please refer to the official repository.*
