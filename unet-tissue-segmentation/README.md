# Deep Learning Modeling and Hyperparameter Optimization for Large-Scale Tissue Image Segmentation

Implemented a U-Net convolutional neural network for pixel-level segmentation of cellular structures in tissue microscopy images using the large-scale TissueNet dataset. Individual course project for **AI in Health Sciences (EE4069)**, Spring 2024.

## Overview

The project covers the full deep learning pipeline: data preprocessing, model training, hyperparameter optimization, and quantitative evaluation. The U-Net encoder-decoder architecture with skip connections is trained on 2,500 images from TissueNet v1.1 to segment nuclear structures at pixel resolution.

### Pipeline

```
TissueNet v1.1 (.npz)
       │
       ▼
┌──────────────────┐
│ 01_preprocess.py │  Extract cell & nuclear masks → PNG
│                  │  Overlay visualization
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  02_train.py     │  tf.data pipeline (augmentation, batching)
│                  │  U-Net: [32]→[64]→[128]→[64]→[32]
│                  │  Binary cross-entropy + IoU + F1
│                  │  Early stopping + LR scheduling
└────────┬─────────┘
         │
         ▼
   F1 = 0.71 | IoU = 0.55
```

## Results

| Metric | This Project | Mesmer (Greenwald et al.) |
|--------|-------------|--------------------------|
| F1 Score | 0.71 | 0.82 |
| IoU | 0.55 | — |
| Training data | 2,500 images | Full dataset |
| Epochs | 10 | Extensive tuning |

The performance gap is primarily attributed to computational constraints that limited both the training subset size and the scope of hyperparameter search.

## U-Net Architecture

```
Input (128×128×1)
  │
  ├─ Conv2D(32) + BN + Dropout(0.2) ───────────────┐ skip
  │  MaxPool                                       │
  ├─ Conv2D(64) + BN + Dropout(0.2) ────────┐ skip │
  │  MaxPool                                │      │
  ├─ Conv2D(128) + BN + Dropout(0.3)        │      │  Bottleneck
  │  UpSample                               │      │
  ├─ Concat + Conv2D(64) + BN + Dropout ────┘      │
  │  UpSample                                      │
  ├─ Concat + Conv2D(32) + BN + Dropout ───────────┘
  │
  └─ Conv2D(1, sigmoid) → Binary Mask
```

## Requirements

```
tensorflow>=2.10
numpy
matplotlib
Pillow
```

## Repository Structure

```
.
├── README.md
├── 01_preprocess.py     # Data extraction, mask saving, overlay visualization
├── 02_train.py          # U-Net model, training, evaluation, visualization
├── data/                # (not tracked)
│   ├── tissuenet_v1.1_train.npz
│   ├── tissuenet_v1.1_val.npz
│   └── tissuenet_v1.1_test.npz
└── results/             # (generated)
```

## Usage

```bash
# Step 1: Extract masks from NPZ archives
python 01_preprocess.py

# Step 2: Train U-Net and evaluate
python 02_train.py
```
