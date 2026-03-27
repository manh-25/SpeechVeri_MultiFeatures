# Handcrafted Feature Extraction Module

This module contains notebooks for extracting handcrafted acoustic features used by the training pipeline.

## Scope

Current folder content:

```text
extract_feature_model/
├── code_extract_features.ipynb
└── README.md
```

The notebook is used to generate handcrafted feature tensors (for example `mfbe_pitch`, `mfcc_pitch`, and related variants) consumed by `train/` mode 2 and mode 3.

## Expected output

The training pipeline expects handcrafted feature payloads compatible with `train/dataset.py`, typically including:

- `features`: list/tensor of shape `(C, T)` per sample
- sample order aligned with PTM embedding samples when using mode 3

## How to run

```bash
cd extract_feature_model
jupyter notebook code_extract_features.ipynb
```

Run cells in order and export the feature artifacts to your configured output directory.

## Integration with training

- Mode 2 uses only handcrafted features.
- Mode 3 requires both PTM embeddings and handcrafted features aligned by sample order.
- Feature channel dimension must match `feature_mode` in `train/config.py`:
  - `mfbe_pitch` -> 81
  - `mfcc_pitch` -> 41
  - `mfbe_only` -> 80
  - `mfcc_only` -> 40
  - `pitch_only` -> 1
