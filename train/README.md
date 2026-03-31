# 🎤 Speaker Verification Training (Current Repo State)

This README documents the **current** behavior of `train/`.

## 📁 File Structure

```
train/
├── config.py           # Default constants
├── model.py            # Model architectures + loss + get_model(...)
├── dataset.py          # Data loading and collate utilities
├── metrics.py          # EER / MinDCF metrics
├── inference.py        # Verification evaluation helpers
├── train.py            # Core training loop (train(args))
├── main.ipynb          # Notebook for training/evaluation
└── README.md
```

## 🎯 Training Modes

| Mode | Input | Description |
|------|------|------|
| `1` | `embedding` | PTM only |
| `2` | `feature` | Handcrafted only |
| `3` | `embedding` + `feature` | Fusion |

### Feature Modes

| Feature Mode | Dim |
|--------------|-----|
| `mfbe_pitch` | 81 |
| `mfcc_pitch` | 41 |
| `mfbe_only`  | 80 |
| `mfcc_only`  | 40 |
| `pitch_only` | 1  |

## 🚀 How to Run

### 1) Notebook (recommended)

```bash
cd train
jupyter notebook main.ipynb
```

### 2) Python API (`train(args)`)

From repository root:

```python
from types import SimpleNamespace
from train.train import train

args = SimpleNamespace(
    embedding_path="path/to/embedding_shards_or_pt",
    feature_path="path/to/feature_shards_or_pt",
    mode=3,
    fusion_method="concat",   # concat | gating | film
    feature_mode="mfbe_pitch",
    use_gating=True,
    use_augment=False,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-4,
    num_epochs=100,
    optimizer="adam",
    lr_scheduler="plateau",
    early_stop_patience=10,
    mixed_precision=True,
    embedding_dim=512,
    output_dir="train/outputs",
    exp_name="Mode3_concat_train_raw_wavlm_mfbe_pitch",
    seed=42,
    duration="train_raw",
    pretrained_model="wavlm",
)

model, history, exp_dir = train(args)
print(exp_dir)
```

## 📦 Outputs

Training artifacts are saved to:

- `train/outputs/experiments/<exp_name>/config.json`
- `train/outputs/experiments/<exp_name>/training_log.txt`
- `train/outputs/experiments/<exp_name>/training_history.json`
- `train/outputs/experiments/<exp_name>/best_model.pth`
- `train/outputs/experiments/<exp_name>/final_model.pth`
- `train/outputs/experiments/<exp_name>/results.json`

## 🧪 Input Data Notes

- `embedding_path`: PTM data (`speaker_ids`, `filenames`, `embeddings`).
- `feature_path`: handcrafted data (`features`) for mode 2/3.
- In mode 3, embedding and feature samples must be aligned in the same order.
