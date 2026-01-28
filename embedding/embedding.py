import torch
import torchaudio
import os
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, HubertModel, Wav2Vec2Model

class SpeakerDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = glob.glob(os.path.join(folder_path, "**", "*.wav"), recursive=True)
        
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

@torch.no_grad()
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
    
    dataset = SpeakerDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    all_embeddings = []
    all_speaker_ids = []
    all_filenames = []

    print(f"--- Đang chạy {model_key} trên {device} ---")
    for waveforms, ids, names in tqdm(dataloader):
        inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        
        # Stack & Mean pooling: (Batch, Layers, Dim)
        # 13 layers cho bản base
        stacked = torch.stack(outputs.hidden_states) # (13, B, T, D)
        pooled = stacked.mean(dim=2).permute(1, 0, 2).cpu() # (B, 13, D)

        all_embeddings.append(pooled)
        all_speaker_ids.extend(ids)
        all_filenames.extend(names)

    final_data = {
        'embeddings': torch.cat(all_embeddings, dim=0),
        'speaker_ids': all_speaker_ids,
        'filenames': all_filenames,
        'model_name': model_key
    }

    torch.save(final_data, save_path)
    print(f"Lưu thành công: {save_path}")
    
    # Dọn dẹp GPU
    del model
    torch.cuda.empty_cache()
    return save_path