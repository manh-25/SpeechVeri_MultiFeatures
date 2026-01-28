import torch
import torchaudio
import os
import gc
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, HubertModel, Wav2Vec2Model
from optimum.bettertransformer import BetterTransformer

class SpeakerDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = glob.glob(os.path.join(folder_path, "**", "*.wav"), recursive=True)
        # Sắp xếp file theo dung lượng (proxy cho độ dài âm thanh)
        self.file_paths.sort(key=lambda x: os.path.getsize(x))
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        filename = os.path.basename(path)
        speaker_id = filename.split('_')[0]
        return waveform, speaker_id, filename

def collate_fn(batch):
    waveforms, ids, names = zip(*batch)
    return list(waveforms), list(ids), list(names)

@torch.inference_mode()
def run_extraction(model_key, folder_path, save_path, batch_size=8):
    """
    Hàm chính để gọi từ Notebook.
    model_key: "wavlm", "hubert", hoặc "wav2vec2"
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Map model key với các class và repo tương ứng
    model_map = {
        "wavlm": (WavLMModel, "microsoft/wavlm-base"),
        "hubert": (HubertModel, "facebook/hubert-base-ls960"),
        "wav2vec2": (Wav2Vec2Model, "facebook/wav2vec2-base-960h")
    }
    
    if model_key not in model_map:
        raise ValueError(f"Model {model_key} không được hỗ trợ. Chọn: {list(model_map.keys())}")
    
    model_class, repo = model_map[model_key]
    processor = Wav2Vec2FeatureExtractor.from_pretrained(repo)
    model = model_class.from_pretrained(repo, output_hidden_states=True).to(device).eval()

    try:
        model = BetterTransformer.transform(model)
        print("Đã kích hoạt BetterTransformer.")
    except Exception as e:
        print(f"Không thể kích hoạt BetterTransformer: {e}. Chạy chế độ thường.")

    # Chuyển sang Half Precision (FP16) để tăng tốc 2x và giảm VRAM 2x
    model = model.half().eval()
    
    dataset = SpeakerDataset(folder_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=False, 
        num_workers=4,      # Thử với 4 hoặc 8 tùy số nhân CPU
        pin_memory=True     # Luôn bật cái này khi dùng CUDA
    )
    
    all_embeddings = []
    all_speaker_ids = []
    all_filenames = []

    print(f"--- Đang chạy {model_key} trên {device} ---")
    # Cơ chế "Streaming" cho .pt (Lưu nháp nếu cần, ở đây tối ưu hóa RAM list)
    for i, (waveforms, ids, names) in enumerate(tqdm(dataloader)):
        waveforms_numpy = [w.numpy() for w in waveforms]
        
        # Ép kiểu input về half ngay khi vào model
        inputs = processor(
            waveforms_numpy, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        inputs['input_values'] = inputs['input_values'].half()
        
        outputs = model(**inputs, output_hidden_states=True)
        
        # Stack & Mean pooling
        stacked = torch.stack(outputs.hidden_states) # (13, B, T, D)
        
        # Chuyển về CPU và xóa tensor trung gian ngay lập tức để giải phóng RAM
        pooled = stacked.mean(dim=2).permute(1, 0, 2).cpu().float() 
        
        all_embeddings.append(pooled)
        all_speaker_ids.extend(ids)
        all_filenames.extend(names)

        # Giải phóng VRAM định kỳ
        if i % 100 == 0:
            torch.cuda.empty_cache()

    # Giải phóng model trước khi thực hiện cat() cuối cùng để tránh đỉnh điểm RAM
    del model
    gc.collect()
    torch.cuda.empty_cache()

    final_data = {
        'embeddings': torch.cat(all_embeddings, dim=0),
        'speaker_ids': all_speaker_ids,
        'filenames': all_filenames,
        'model_name': model_key
    }

    torch.save(final_data, save_path)
    print(f"Lưu thành công: {save_path}")
    return save_path