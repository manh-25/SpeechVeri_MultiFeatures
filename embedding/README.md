# Embedding Module

Mô-đun này chứa mã trích xuất **speaker embedding** từ các tệp âm thanh WAV bằng cách sử dụng các mô hình pre-trained từ Hugging Face.

## 📋 Tổng quan

Module embedding cung cấp tính năng để:
- **Tải và xử lý dữ liệu âm thanh**: Tự động load file WAV từ thư mục con, resample về 16kHz, xử lý stereo
- **Trích xuất embedding**: Sử dụng 3 mô hình khác nhau (WavLM, HuBERT, Wav2Vec2)
- **Lưu kết quả**: Lưu embedding, speaker ID, tên tệp dưới dạng PyTorch tensor

## 📁 Cấu trúc tập tin

```
embedding/
├── README.md                 # Tài liệu này
├── embedding.py              # Mô-đun Python chính
└── embed_folder.ipynb        # Notebook ví dụ
```

## 🔧 Thành phần chính

### `embedding.py`

#### Class: `SpeakerDataset`
Lớp Dataset cho phép tải và xử lý dữ liệu âm thanh:
- **Input**: Đường dẫn thư mục chứa các file WAV (hỗ trợ cấu trúc thư mục con)
- **Output**: Waveform, Speaker ID, tên tệp
- **Xử lý tự động**:
  - Resample về 16 kHz
  - Chuyển stereo → mono (nếu cần)
  - Trích xuất Speaker ID từ tên tệp (phần trước dấu `_`)

#### Hàm: `collate_fn`
Hàm ghép dữ liệu cho batch, trả về danh sách waveform, speaker IDs, và tên tệp.

#### Hàm: `run_extraction`
Hàm chính để trích xuất embedding.

**Tham số:**
- `model_key` (str): Mô hình sử dụng - `"wavlm"`, `"hubert"`, hoặc `"wav2vec2"`
- `folder_path` (str): Đường dẫn thư mục chứa file WAV
- `save_path` (str): Đường dẫn tệp để lưu kết quả (`.pt` file)
- `batch_size` (int, mặc định=8): Kích thước batch xử lý

**Đầu ra:**
Lưu file `.pt` chứa dictionary với các khóa:
```python
{
    'embeddings': torch.Tensor,      # Shape: (N_samples, 13_layers, 768_dim)
    'speaker_ids': List[str],         # Danh sách speaker ID
    'filenames': List[str],           # Danh sách tên tệp
    'model_name': str                 # Tên mô hình được sử dụng
}
```

## 🚀 Cách sử dụng

### Cách 1: Sử dụng Notebook (Khuyến nghị)

Mở `embed_folder.ipynb` và chạy các cell:

```python
from embedding import run_extraction

# Cấu hình đường dẫn
DATA_DIR = r"E:\speech_data\train_raw"  # Thay bằng đường dẫn của bạn
MODELS = ["wavlm", "hubert", "wav2vec2"]

# Trích xuất embedding cho từng mô hình
for m in MODELS:
    output_name = f"{m}_all_layers.pt"
    run_extraction(
        model_key=m, 
        folder_path=DATA_DIR, 
        save_path=output_name, 
        batch_size=16
    )
```

### Cách 2: Sử dụng từ Python script

```python
from embedding import run_extraction

# Trích xuất embedding cho 1 mô hình
run_extraction(
    model_key="wavlm",
    folder_path="/path/to/audio/folder",
    save_path="wavlm_embeddings.pt",
    batch_size=16
)
```

### Cách 3: Kiểm tra kết quả

```python
import torch

# Load embedding đã lưu
data = torch.load("wavlm_all_layers.pt")

print(f"Mô hình: {data['model_name']}")
print(f"Số mẫu: {len(data['filenames'])}")
print(f"Shape embedding: {data['embeddings'].shape}")  # (N, 13, 768)

# Kiểm tra mẫu đầu tiên
print(f"File: {data['filenames'][0]}")
print(f"Speaker ID: {data['speaker_ids'][0]}")
print(f"Embedding layer 12: {data['embeddings'][0, 12]}")  # Lớp cuối cùng
```

## 📊 Mô hình được hỗ trợ

| Mô hình | Model Key | Repo | Kích thước | Mô tả |
|---------|-----------|------|-----------|--------|
| **WavLM** | `wavlm` | microsoft/wavlm-base | 300MB | Microsoft SEAL, phù hợp speaker verification |
| **HuBERT** | `hubert` | facebook/hubert-base-ls960 | 360MB | Meta/Facebook SELF-supervised Learning, tốt cho understanding |
| **Wav2Vec 2.0** | `wav2vec2` | facebook/wav2vec2-base-960h | 360MB | Meta self-supervised, phù hợp ASR |

## ⚙️ Yêu cầu kỹ thuật

### Thư viện Python
```
torch>=1.9.0
torchaudio>=0.9.0
transformers>=4.20.0
tqdm
```

### Cài đặt
```bash
pip install torch torchaudio transformers tqdm
```

### GPU (Tùy chọn nhưng Khuyến nghị)
- CUDA 11.x hoặc cao hơn
- GPU với VRAM ≥ 8GB (cho batch_size=16)
- Để chạy trên CPU, để `batch_size=4` hoặc nhỏ hơn

## 📝 Chi tiết kỹ thuật

### Xử lý Audio
1. Load file WAV với torchaudio
2. Kiểm tra sample rate và resample nếu cần (target: 16kHz)
3. Chuyển stereo → mono (lấy trung bình)

### Trích xuất Embedding
1. Sử dụng `Wav2Vec2FeatureExtractor` để chuẩn bị audio input
2. Đưa vào mô hình pre-trained với `output_hidden_states=True`
3. Stack tất cả 13 lớp hidden states: `(13, Batch, Time, 768)`
4. Mean pooling theo chiều time: `(Batch, 13, 768)`
5. Chuyển về CPU và lưu

### Output Format
- **Shape**: `(N_samples, 13_layers, 768_dimensions)`
- **Dòng mỗi lớp**: Biểu diễn vector từ mỗi lớp mô hình
- **Lớp cuối (index 12)**: Thường được sử dụng cho speaker verification tasks

## ⏱️ Thời gian xử lý ước tính

| Dữ liệu | GPU (RTX 3060) | GPU (RTX 4090) | CPU (i7) |
|---------|----------------|----------------|----------|
| 100 samples | ~30s | ~15s | ~5min |
| 1000 samples | ~3min | ~1.5min | ~50min |
| 10000 samples | ~30min | ~15min | ~8h |

## 🔍 Ghi chú quan trọng

- **Tên file**: Speaker ID được trích từ phần trước dấu `_` trong tên file (ví dụ: `speaker001_sample1.wav` → `speaker001`)
- **GPU Memory**: Sử dụng `batch_size` nhỏ hơn nếu gặp lỗi "CUDA out of memory"
- **Model Download**: Lần đầu chạy sẽ tự động download mô hình (~300-360MB mỗi mô hình)
- **Autoreload**: Notebook sử dụng `%autoreload 2` để tự động cập nhật code mà không cần restart kernel

## 🐛 Troubleshooting

### Lỗi: "CUDA out of memory"
```python
# Giảm batch_size
run_extraction(..., batch_size=4)
```

### Lỗi: "Model not supported"
```python
# Chỉ sử dụng các model sau:
# - "wavlm"
# - "hubert"  
# - "wav2vec2"
```

### Lỗi: Không tìm thấy file WAV
- Kiểm tra đường dẫn folder_path
- Đảm bảo folder chứa file `.wav` (có thể trong thư mục con)

## 📚 Tài liệu tham khảo

- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [WavLM Paper](https://arxiv.org/abs/2110.01852)
- [HuBERT Paper](https://arxiv.org/abs/2106.07447)
- [Wav2Vec 2.0 Paper](https://arxiv.org/abs/2006.11477)


