# 🎤 Speaker Verification Training Pipeline

A comprehensive training framework for speaker verification using Pre-trained Models (PTM) and handcrafted features with flexible argparse-based configuration.

## 📁 File Structure

```
train/
├── config.py           # Default configuration and constants
├── model.py            # Model architectures (PTM, Handcrafted, Fusion, ECAPA-TDNN)
├── dataset.py          # Data loading and processing utilities
├── train.py            # Main training script with argparse
├── main.ipynb          # Interactive Jupyter notebook
├── run_examples.sh     # Example commands for common scenarios
└── README.md           # This file
```

## 🎯 Features

### 1. **Three Training Modes**

| Mode | Description | Best For |
|------|-------------|----------|
| **Mode 1** | PTM embeddings only | Baseline, pre-trained features |
| **Mode 2** | Handcrafted features only | Lightweight, real-time inference |
| **Mode 3** | PTM + Handcrafted fusion | Best performance, multi-modal learning |

### 2. **Fusion Methods (Mode 3)**

| Method | Description |
|--------|-------------|
| **Concatenation** | Simple concat + FC projection (fast) |
| **Cross-Attention** | Multi-head cross-modal attention (8 heads) |
| **Gating** | Dynamic gate to weight PTM vs Handcrafted |

### 3. **Feature Modes (Mode 2, 3)**

| Feature Mode | Features | Dimensions |
|--------------|----------|-----------|
| `mfbe_pitch` | MFBE + F0 | 81-dim |
| `mfcc_pitch` | MFCC + F0 | 41-dim |
| `mfbe_only` | MFBE only | 80-dim |
| `mfcc_only` | MFCC only | 40-dim |
| `pitch_only` | Pitch only | 1-dim |

### 4. **Model Architecture**

- **PTM Encoder**: Multi-layer weighted sum (13 layers, 768-dim)
- **Handcrafted Encoder**: CNN-1D for feature processing (768-dim output)
- **Fusion Module**: Concat / Cross-Attention / Gating
- **ECAPA-TDNN**: 4 bottleneck blocks, statistical pooling, 512-dim embedding
- **Loss**: AAM-Softmax (margin=0.2, scale=30)

### 5. **Training Features**

✅ **Argparse Support** - Full CLI control with defaults  
✅ **Early Stopping** - Configurable patience and delta  
✅ **Learning Rate Scheduling** - Cosine annealing or plateau reduction  
✅ **Optimizers** - Adam or SGD with weight decay  
✅ **Mixed Precision** - AMP support for faster training  
✅ **Gradient Clipping** - Prevent exploding gradients (max_norm=1.0)  
✅ **Experiment Management** - Auto-organize results with timestamps  
✅ **Reproducibility** - Fixed seed support  
✅ **Metrics** - Accuracy, EER (Equal Error Rate), MinDCF  
✅ **Gating Analysis** - Visualize PTM vs Handcrafted preference  

---

## 🚀 Quick Start

### Option 1: Command Line (Recommended)

```bash
cd train

# Simple - Mode 1 (PTM only)
python train.py --mode 1

# Medium - Mode 2 (Handcrafted)
python train.py --mode 2 --feature-mode mfbe_pitch

# Advanced - Mode 3 (Fusion with gating)
python train.py --mode 3 --fusion-method gating --feature-mode mfbe_pitch

# Full control
python train.py \
  --embedding-path ./embedding.pt \
  --feature-path ./feature.pt \
  --mode 3 \
  --fusion-method cross_attention \
  --feature-mode mfbe_pitch \
  --batch-size 32 \
  --learning-rate 0.0005 \
  --epochs 200 \
  --exp-name my_experiment \
  --seed 42
```

### Option 2: Jupyter Notebook

```bash
cd train
jupyter notebook main.ipynb
# Execute cells in order
```

### Option 3: Python API

```python
from train import train
import argparse

args = argparse.Namespace(
    embedding_path="./embedding.pt",
    feature_path="./feature.pt",
    mode=3,
    fusion_method="gating",
    feature_mode="mfbe_pitch",
    batch_size=32,
    learning_rate=0.0005,
    epochs=200,
    # ... more parameters
)

model, history, exp_dir = train(args)
```

---

## 💻 CLI Arguments

### View All Arguments

```bash
python train.py --help
```

### Essential Parameters

```bash
# Data
--embedding-path PATH    # Path to embedding.pt (default: ./embedding.pt)
--feature-path PATH      # Path to feature.pt (default: ./feature.pt)

# Model config
--mode {1,2,3}           # Training mode (default: 1)
--fusion-method {concat,cross_attention,gating}
--feature-mode {mfbe_pitch,mfcc_pitch,mfbe_only,mfcc_only,pitch_only}

# Training
--batch-size SIZE        # (default: 64)
--learning-rate LR       # (default: 0.001)
--epochs NUM             # (default: 100)
--weight-decay WD        # (default: 0.0001)

# Early stopping
--early-stop-patience N  # (default: 10)
--early-stop-delta D     # (default: 0.0001)

# Learning rate scheduler
--lr-scheduler {cosine,plateau}
--cosine-t-max T         # Cosine T_max (default: 50)
--plateau-patience P     # (default: 5)
--plateau-factor F       # (default: 0.5)

# Optimizer
--optimizer {adam,sgd}
--momentum M             # SGD momentum (default: 0.9)
--nesterov               # Enable Nesterov momentum

# Other
--mixed-precision        # Enable AMP training
--device DEVICE          # cuda:0, cpu, etc.
--exp-name NAME          # Experiment name (auto-generated if None)
--seed SEED              # Random seed (default: 42)
--output-dir DIR         # Output directory (default: ./outputs)
```

### Example Commands

```bash
# Mode 1 baseline
python train.py --mode 1 --exp-name baseline_mode1

# Mode 2 with different features
python train.py --mode 2 --feature-mode mfbe_only --exp-name ablation_mfbe_only

# Mode 3 with all fusion methods
python train.py --mode 3 --fusion-method concat --exp-name fusion_concat
python train.py --mode 3 --fusion-method cross_attention --exp-name fusion_crossattn
python train.py --mode 3 --fusion-method gating --exp-name fusion_gating

# Custom hyperparameters
python train.py --mode 3 --batch-size 32 --learning-rate 0.0005 --epochs 200

# Different optimizers & schedulers
python train.py --optimizer sgd --momentum 0.95 --nesterov --lr-scheduler plateau

# Reproducibility
python train.py --seed 42 --exp-name seed_42_run1
```
