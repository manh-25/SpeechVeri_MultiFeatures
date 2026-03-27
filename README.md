# SpeechVeri_MultiFeatures

This repository provides a full **Speaker Verification** pipeline with:
- PTM embeddings (WavLM / HuBERT / Wav2Vec2)
- Handcrafted features (MFBE / MFCC / Pitch)
- Training for mode 1/2/3 in `train/`
- A Streamlit demo app in `app/`

## 1) Project Structure

```text
SpeechVeri_MultiFeatures/
├── data_preparation/       # Notebooks for audio conversion/splitting
├── embedding/              # PTM embedding extraction
├── extract_feature_model/  # Handcrafted feature extraction notebook
├── train/                  # Training and evaluation pipeline
├── app/                    # Streamlit demo (register/compare/identify)
├── requirements.txt
└── README.md
```

## 2) Environment Setup

From repository root:

```bash
pip install -r requirements.txt
```

Recommended Python version: **3.10–3.12**.

## 3) End-to-End Run Guide

### Step A — Prepare Audio Data

Use notebooks in `data_preparation/` to convert/cut/split audio.
Expected downstream input is normalized `.wav` files.

### Step B — Extract PTM Embeddings

Use `embedding/main.ipynb` (or APIs in `embedding/embedding.py`).
Output `.pt` files include:
- `embeddings` with shape `(N, 13, 768)`
- `speaker_ids`
- `filenames`

### Step C — Extract Handcrafted Features

Use notebooks in `extract_feature_model/` to generate handcrafted features (`mfbe_pitch`, `mfcc_pitch`, etc.).
Feature sample order must match PTM embedding sample order.

### Step D — Train Model

`train/train.py` is currently used via notebook/API (no argparse CLI entrypoint).

Recommended:

```bash
cd train
jupyter notebook main.ipynb
```

Or run with Python API:

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

### Step E — Run Streamlit Demo

From repository root:

```bash
streamlit run app/streamlit_app.py
```

Default app checkpoint:
- `train/outputs/experiments/Mode3_concat_train_raw_wavlm_mfbe_pitch/best_model.pth`

If you hit a Streamlit watcher issue with `torch.classes`:

```bash
streamlit run app/streamlit_app.py --server.fileWatcherType none
```

## 4) Input Naming Convention

For embedding extraction scripts, speaker ID is parsed from filename prefix before `_`.

Example:
- `speaker001_sample01.wav` -> speaker ID `speaker001`

## 5) Module Documentation

- `embedding/README.md`
- `extract_feature_model/README.md`
- `train/README.md`
- `app/README.md`

## 6) Current Mode 3 Notes

- Mode 3 consumes both `embedding` and `feature` inputs.
- Valid mode 3 fusion methods in current code: `concat`, `gating`, `film`.
- `cross_attention` has been removed for mode 3 and raises `ValueError`.
