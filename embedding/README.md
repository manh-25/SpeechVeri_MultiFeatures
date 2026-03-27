# Embedding Module

This module extracts **PTM speaker embeddings** from WAV audio using Hugging Face models.

## What it does

- Recursively scans WAV files from a folder
- Loads audio and normalizes to 16 kHz mono
- Extracts hidden states from PTM models (WavLM / HuBERT / Wav2Vec2)
- Applies mean pooling over time
- Saves output in shard `.pt` files

## Files

```text
embedding/
├── embedding.py           # Extraction pipeline
├── resume_embeeding.py    # Resume-capable extraction pipeline
├── main.ipynb             # Notebook workflow
└── README.md
```

## Output format

Saved `.pt` payload contains:

```python
{
    "embeddings": Tensor,   # shape: (N, 13, 768)
    "speaker_ids": List[str],
    "filenames": List[str],
    "model_name": str
}
```

## Supported models

- `wavlm` -> `microsoft/wavlm-base`
- `hubert` -> `facebook/hubert-base-ls960`
- `wav2vec2` -> `facebook/wav2vec2-base-960h`

## Run from notebook

```bash
cd embedding
jupyter notebook main.ipynb
```

## Run from Python

```python
from embedding.embedding import run_extraction

run_extraction(
    model_key="wavlm",
    folder_path="/path/to/wav_root",
    save_dir="/path/to/output_dir",
    batch_size=8,
    shard_size=10000,
)
```

## Naming convention

Speaker ID is parsed from filename prefix before `_`.

Example:
- `speaker001_sample01.wav` -> speaker ID `speaker001`

## Notes

- Use smaller `batch_size` if GPU memory is limited.
- The first run downloads model weights from Hugging Face.
- For very large datasets, use the resume pipeline in `resume_embeeding.py`.
