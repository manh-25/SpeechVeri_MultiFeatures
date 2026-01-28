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
python train.py --seed 42 --exp-name seed_42_run2  # Same seed = same results
```

---

## 📓 Jupyter Notebook Guide

`main.ipynb` provides an interactive training workflow. This is the **recommended method** for most users.

### Getting Started

```bash
cd train
jupyter notebook main.ipynb
```

### Notebook Structure (13 Cells)

#### **Cell 1: Title & Description**
- Overview of the pipeline
- Architecture details

#### **Cell 2: Setup & Device Check**
- Auto-reload modules
- Import libraries
- Check GPU availability
- Display GPU info

**Action**: Just run it - no changes needed

#### **Cell 3: Parse Arguments**
- Define argument parser with all parameters
- Use defaults from `config.py`
- Parse arguments (empty list for notebook mode)
- Display all parsed arguments

**To modify parameters**, edit this cell before running:

```python
# Example: Change these inside parse_args function
parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--learning-rate", type=float, default=0.0005)
parser.add_argument("--epochs", type=int, default=200)
```

#### **Cell 4: Training Execution** ⭐ **Main Cell**
- Generate experiment name if not provided
- Execute `train(args)` function
- Train the model
- Save results to `outputs/experiments/exp_name/`

**What it does:**
- Loads data
- Creates model
- Runs training loop with early stopping
- Saves checkpoints and logs
- Prints completion message

**Output:** Model, training history, experiment directory

#### **Cell 5: Plot Training Curves**
- Load training history from JSON
- Plot loss curves (train vs val)
- Plot accuracy curves (train vs val)
- Save visualization to file
- Print best metrics

**What to expect:**
- Loss should decrease over epochs
- Accuracy should increase over epochs
- See if model is overfitting

#### **Cell 6: Load Best Model**
- Load best checkpoint from experiment
- Move to device
- Set to eval mode
- Print confirmation message

**Note**: Best model is determined by lowest validation loss

#### **Cell 7: Test Evaluation**
- Recreate test loader
- Load best model from previous cell
- Evaluate on test set
- Print test loss and accuracy

**Output:** Final test performance metrics

#### **Cell 8: Experiment Comparison**
- List all experiments in `outputs/experiments/`
- Load results from each experiment
- Create comparison table
- Display: Mode, Fusion, Feature, Val Loss, Epochs

**Use this to:**
- Compare different runs
- Find best configuration
- Track experiment history

---

### Step-by-Step Usage Guide

#### **Step 1: Prepare Data**

Ensure you have:
- `embedding.pt` - Pre-trained model embeddings
- `feature.pt` - Handcrafted features

Update paths in **Cell 3** if needed:

```python
parser.add_argument("--embedding-path", type=str, default="/path/to/embedding.pt")
parser.add_argument("--feature-path", type=str, default="/path/to/feature.pt")
```

#### **Step 2: Configure Training Parameters**

Edit **Cell 3** to set parameters:

```python
# Mode selection
parser.add_argument("--mode", type=int, default=3)  # 1, 2, or 3

# For Mode 3 - Fusion
parser.add_argument("--fusion-method", default="gating")  # or "concat", "cross_attention"

# Handcrafted features (Mode 2, 3)
parser.add_argument("--feature-mode", default="mfbe_pitch")

# Training parameters
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--learning-rate", type=float, default=0.0005)
parser.add_argument("--epochs", type=int, default=200)

# Optional: Give experiment a meaningful name
parser.add_argument("--exp-name", default="my_first_run")

# Reproducibility
parser.add_argument("--seed", type=int, default=42)
```

#### **Step 3: Run Training**

Execute cells in order:

1. **Cell 1**: Read title (no action)
2. **Cell 2**: Run setup & device check
3. **Cell 3**: Run argument parsing (see your config)
4. **Cell 4**: Run training ⭐ (this takes time!)
5. Wait for training to complete (see progress bar)

During training, you should see:
```
STARTING TRAINING
Experiment: mode3_fusion_gating_feat_mfbe_pitch_20260128_120000

Training started at: 2026-01-28 12:00:00
Loading data...
✓ Loaded 200 speakers
  Train: 8000, Val: 2000, Test: 2000

Creating model...
✓ Model summary saved to outputs/...

Starting training...

Epoch   1 | Train Loss: 3.2145, Acc: 0.1234 | Val Loss: 2.8934, Acc: 0.2145 | LR: 0.000500
Epoch   2 | Train Loss: 2.1456, Acc: 0.3456 | Val Loss: 1.9234, Acc: 0.4123 | LR: 0.000500
...
✓ Training completed!
  Best validation loss: 0.1234
  Results saved to: outputs/experiments/mode3_...
```

#### **Step 4: View Training Results**

After training completes:

1. **Cell 5**: Plot training curves
   - See loss/accuracy progression
   - Check for convergence
   - Detect overfitting

2. **Cell 6**: Load best model
   - Automatically loads checkpoint with lowest val loss
   - Ready for evaluation

3. **Cell 7**: Evaluate on test set
   - See final performance
   - Compare train/val/test accuracy

#### **Step 5: Compare Experiments**

Run **Cell 8** to compare all previous runs:

```
==================================================
EXPERIMENT COMPARISON
==================================================
Experiment                              Mode  Fusion              Feature      Best Val Loss  Epochs
mode1_fusion_N/A_feat_N/A_120000          1   N/A                 N/A               0.2345     100
mode2_fusion_N/A_feat_mfbe_120100         2   N/A                 mfbe_pitch        0.1234     95
mode3_fusion_concat_feat_mfbe_120200      3   concat              mfbe_pitch        0.0989     87
mode3_fusion_gating_feat_mfbe_120300      3   gating              mfbe_pitch        0.0876     92
==================================================
```

---

### Common Workflows

#### **Workflow 1: Quick Training (Default)**

```python
# Cell 2: Run setup
# Cell 3: Run with defaults
# Cell 4: Run training
# Cell 5: Plot curves
# Cell 7: Check test results
```

#### **Workflow 2: Mode Comparison**

Run notebook 3 times with different modes:

**First run - Mode 1:**
```python
# Cell 3: Change to mode=1
# Cell 4: Run training
```

**Second run - Mode 2:**
```python
# Cell 3: Change to mode=2, feature_mode="mfbe_pitch"
# Cell 4: Run training
```

**Third run - Mode 3:**
```python
# Cell 3: Change to mode=3, fusion_method="gating"
# Cell 4: Run training
```

**Compare all:**
```python
# Cell 8: View comparison table
```

#### **Workflow 3: Hyperparameter Tuning**

```python
# Cell 3: Change learning rate, batch size, epochs
# Cell 4: Run training
# Cell 5-7: Evaluate
# Repeat with different parameters
```

#### **Workflow 4: Statistical Significance**

Run with multiple seeds:

```python
# Cell 3: seed=42
# Cell 4: Run training
# (Restart kernel - Kernel menu > Restart)
# Cell 3: seed=123
# Cell 4: Run training
# (Repeat for seeds: 456, 789, 999)
# Cell 8: Compare all runs
```

---

### Tips & Tricks

#### **Modify Arguments Without Editing Cell Code**

You can modify arguments AFTER parsing:

```python
# After Cell 3 runs:
args.batch_size = 64
args.learning_rate = 0.0001
args.epochs = 300

# Then run Cell 4 with new values
```

#### **Run Multiple Experiments Without Restarting**

```python
# Modify arguments
args.mode = 1
args.exp_name = "exp_mode1_v1"
# Run Cell 4

# Modify again
args.mode = 3
args.fusion_method = "gating"
args.exp_name = "exp_mode3_gating"
# Run Cell 4 again
```

#### **Monitor GPU During Training**

Open another terminal:

```bash
watch -n 1 nvidia-smi
```

While notebook is training, this shows real-time GPU usage.

#### **Stop Training Early**

If training takes too long or isn't going well:

1. Click "Stop" button in notebook (or press Ctrl+C)
2. Training stops, best model is still saved
3. Run cells 5-7 to check results

#### **Restart & Clear Output**

If notebook becomes slow:

1. **Kernel** menu → **Restart Kernel**
2. Click notebook cells again from top

This clears memory while keeping code.

---

### Troubleshooting in Notebook

| Problem | Solution |
|---------|----------|
| "File not found" | Update `--embedding-path` and `--feature-path` in Cell 3 |
| "CUDA out of memory" | Reduce `--batch-size` in Cell 3 (try 16 or 32) |
| Training very slow | Enable mixed precision: Add `parser.add_argument("--mixed-precision", action="store_true", default=True)` in Cell 3 |
| Plot not showing (Cell 5) | Check history file exists: `ls outputs/` |
| No test results (Cell 7) | Ensure Cell 4 (training) completed successfully |
| Comparison table empty (Cell 8) | Run Cell 4 at least once to create experiments |

---

### Output Files

After each training run, check results:

```bash
outputs/experiments/mode{M}_fusion_{F}_feat_{F}_HHMMSS/
├── config.json              # Hyperparameters used
├── results.json             # Final metrics
├── training_history.json    # Loss & accuracy per epoch
├── training_log.txt         # Detailed log
├── model_summary.txt        # Model architecture
├── best_model.pth           # Best checkpoint
├── final_model.pth          # Final checkpoint
└── training_curves.png      # Plot from Cell 5
```

View results:

```bash
cat outputs/experiments/*/config.json       # See configuration
cat outputs/experiments/*/results.json      # See final metrics
tail outputs/experiments/*/training_log.txt # See training logs
```

---

### Example Full Session

```bash
# 1. Open notebook
cd train
jupyter notebook main.ipynb

# 2. Run cells 1-3 (setup & arguments)
# 3. Modify arguments in Cell 3 if needed
# 4. Run Cell 4 (training - wait 30-60 minutes)
# 5. Run Cell 5 (plot curves)
# 6. Run Cells 6-7 (test results)
# 7. Run Cell 8 (comparison)

# 8. Try different configuration
# Modify Cell 3, run Cell 4 again
# Run Cells 5-8 to see comparison

# 9. View all results
ls outputs/experiments/
cat outputs/experiments/*/results.json
```

---

## 📊 Results & Output Structure

After training, results are organized automatically:

```
outputs/
└── experiments/
    └── mode{M}_fusion_{F}_feat_{F}_HHMMSS/
        ├── config.json              # All hyperparameters
        ├── results.json             # Final metrics (loss, accuracy, etc.)
        ├── training_history.json    # Loss & accuracy per epoch
        ├── training_log.txt         # Detailed training log
        ├── model_summary.txt        # Model architecture
        ├── best_model.pth           # Best checkpoint
        ├── final_model.pth          # Final checkpoint
        ├── confusion_matrices/      # Confusion matrix plots (if saved)
        └── gating_analysis/         # Gate distribution (Mode 3 + gating only)

checkpoints/
├── best_model.pth
└── final_model.pth
```

### View Results

```bash
# List experiments
ls outputs/experiments/

# View config
cat outputs/experiments/exp_name/config.json

# View metrics
cat outputs/experiments/exp_name/results.json

# View training log
tail -100 outputs/experiments/exp_name/training_log.txt
```

---

## 🔧 Common Use Cases

### Ablation Study - Feature Modes

```bash
for feature in mfbe_pitch mfcc_pitch mfbe_only mfcc_only pitch_only; do
  python train.py --mode 2 --feature-mode $feature \
    --exp-name ablation_feature_$feature &
done
```

### Ablation Study - Fusion Methods

```bash
for fusion in concat cross_attention gating; do
  python train.py --mode 3 --fusion-method $fusion \
    --exp-name ablation_fusion_$fusion &
done
```

### Learning Rate Search

```bash
for lr in 0.0001 0.0005 0.001 0.005; do
  python train.py --learning-rate $lr \
    --exp-name lr_search_${lr} &
done
```

### Batch Size Study

```bash
for bs in 16 32 64 128 256; do
  python train.py --batch-size $bs \
    --exp-name batch_study_$bs &
done
```

### Statistical Significance (Multiple Seeds)

```bash
for seed in 42 123 456 789 999; do
  python train.py --seed $seed \
    --exp-name seed_study_$seed &
done
```

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found (torch, etc.) | `pip install -r requirements.txt` or install manually |
| CUDA out of memory | Reduce `--batch-size` (e.g., 16 or 32) |
| Files not found | Update `--embedding-path` and `--feature-path` |
| Training not converging | Lower learning rate, increase epochs, change scheduler |
| Very slow training | Enable mixed precision: `--mixed-precision` |
| Loss/acc not improving | Check data format, verify file paths, try different optimizer |

### Check Training Logs

```bash
# View detailed log
cat outputs/experiments/*/training_log.txt

# Real-time monitoring
tail -f outputs/experiments/*/training_log.txt
```

### Monitor GPU

```bash
# In another terminal
watch -n 1 nvidia-smi
```

---

## 💡 Best Practices

1. **Start simple**: Mode 1 first, then Mode 2, then Mode 3
2. **Compare modes**: Run all three to understand tradeoffs
3. **Use meaningful names**: `--exp-name "mode3_gating_mfbe_lr0001_bs32"`
4. **Multiple seeds**: Run each config with 3-5 different seeds
5. **Ablate systematically**: Change only one parameter at a time
6. **Save logs**: Redirect output to file: `... 2>&1 | tee exp.log`
7. **Monitor progress**: Use `nvidia-smi` to check GPU usage

---

## 📈 Expected Performance

Typical results (may vary by dataset):

| Mode | Fusion | Feature | Acc | EER |
|------|--------|---------|-----|-----|
| 1 | N/A | N/A | 95-98% | 2-5% |
| 2 | N/A | mfbe_pitch | 90-95% | 5-10% |
| 3 | concat | mfbe_pitch | 96-98% | 3-5% |
| 3 | cross_attention | mfbe_pitch | 97-99% | 1-3% |
| 3 | gating | mfbe_pitch | 97-99% | 1-3% |

---

## 📦 Requirements

```
torch >= 1.10
torchvision
torchaudio
scikit-learn
matplotlib
seaborn
tqdm
torchinfo
transformers
numpy
```

Install: `pip install torch scikit-learn matplotlib seaborn tqdm torchinfo transformers`

---

## 🎓 Advanced Topics

### Custom Training Loop

See `train.py` for utility functions:
- `train_epoch()` - One training epoch
- `validate()` - Validation loop  
- `EarlyStopping` - Early stopping callback
- `load_checkpoint()` - Load saved model
- `save_checkpoint()` - Save model state
- `compute_metrics()` - Accuracy computation
- `compute_eer()` - Equal Error Rate (speaker verification metric)
- `analyze_gating_behavior()` - Visualize gating weights

Example custom loop:

```python
from train import train_epoch, validate, EarlyStopping
from dataset import create_data_loaders
from model import get_model, AAMSoftmaxLoss
import torch

# Setup
train_loader, val_loader, test_loader, speaker_to_idx, num_speakers = \
    create_data_loaders(embedding_path, feature_path, mode, batch_size)

model = get_model(num_speakers, mode=mode, fusion_method=fusion_method)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = AAMSoftmaxLoss(num_speakers=num_speakers)
early_stopping = EarlyStopping(patience=10)

# Training loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, criterion, scaler=None, 
        epoch=epoch, device=device
    )
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
```

---

## 📞 Support

For issues:

1. **Check logs**: `outputs/experiments/*/training_log.txt`
2. **Verify data**: embedding.pt and feature.pt shapes must match config
3. **Install packages**: `pip install torch scikit-learn matplotlib seaborn tqdm torchinfo`
4. **GPU problems**: Use `--device cpu` or check `nvidia-smi`


