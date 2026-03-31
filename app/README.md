# Demo App - Vietnamese Speaker Recognition

## Features

- Register a new speaker using multiple audio samples
- Compare two audio files using cosine similarity + accept/reject threshold
- Identify one audio sample against the registered speaker list
- Supports both file upload and direct microphone recording

## Default Model

- Default checkpoint:
  - `train/outputs/experiments/Mode3_concat_train_raw_wavlm_mfbe_pitch/best_model.pth`
- Runtime configuration:
  - `mode=3`, `fusion=concat`, `feature_mode=mfbe_pitch`

## Feature Pipeline Used by the App

- Sample rate: 16kHz
- PTM embedding:
  - WavLM base (`microsoft/wavlm-base`), hidden states mean-pooled over time -> `(13, 768)`
- Handcrafted branch: `MFBE + Pitch`
  - MFBE: log-mel (`n_mels=80`, `n_fft=400`, `hop=160`, `center=False`)
  - Pitch: `librosa.pyin` (`fmin=60`, `fmax=500`) with fallback to `torchaudio.detect_pitch_frequency`
  - Time alignment + CMVN after concatenation

## Run the App

From repository root:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

If you see watcher errors related to `torch.classes`, use:

```bash
streamlit run app/streamlit_app.py --server.fileWatcherType none
```

## Registered Speaker Storage

- Default path: `app/data/speaker_registry.pt`
- You can change this path in the app sidebar.
