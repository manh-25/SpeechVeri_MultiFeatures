import torch
import torchaudio
import os
import gc
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, HubertModel, Wav2Vec2Model

# Class Dataset giữ nguyên như của bạn để đảm bảo thứ tự file không đổi
class SpeakerDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = glob.glob(os.path.join(folder_path, "**", "*.wav"), recursive=True)
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
def run_extraction_resume(model_key, folder_path, save_dir, batch_size=16, shard_size=10000):
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dataset = SpeakerDataset(folder_path)
    total_files = len(dataset)
    num_shards = (total_files + shard_size - 1) // shard_size
    
    model_map = {"wavlm": WavLMModel, "hubert": HubertModel, "wav2vec2": Wav2Vec2Model}
    repo_map = {"wavlm": "microsoft/wavlm-base", "hubert": "facebook/hubert-base-ls960", "wav2vec2": "facebook/wav2vec2-base-960h"}
    
    model_class, repo = model_map[model_key], repo_map[model_key]
    processor = Wav2Vec2FeatureExtractor.from_pretrained(repo)
    
    # Khởi tạo biến model là None để Lazy Loading (chỉ load khi thực sự cần trích xuất)
    model = None

    print(f"--- Kiểm tra tiến độ cho {model_key.upper()} ---")
    
    for s_idx in range(num_shards):
        shard_path = os.path.join(save_dir, f"{model_key}_shard_{s_idx}.pt")
        
        # Ý 1: KIỂM TRA SHARD ĐÃ TỒN TẠI
        if os.path.exists(shard_path):
            print(f"⏩ Shard {s_idx} đã xong. Bỏ qua.")
            continue

        # Nếu chưa có shard, bắt đầu load model
        if model is None:
            model = model_class.from_pretrained(
                repo, 
                output_hidden_states=True,
                torch_dtype=torch.float32, 
                attn_implementation="eager"
            ).to(device).eval()

        print(f"🚀 Bắt đầu trích xuất Shard {s_idx}/{num_shards-1}...")
        
        # Xác định phạm vi file cho shard này
        start_idx = s_idx * shard_size
        end_idx = min(start_idx + shard_size, total_files)
        subset_indices = list(range(start_idx, end_idx))
        
        subset = Subset(dataset, subset_indices)
        dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn, num_workers=16, pin_memory=True)
        
        shard_embeddings, shard_ids, shard_names = [], [], []

        for i, (waveforms, ids, names) in enumerate(tqdm(dataloader, desc=f"Shard {s_idx}")):
            waveforms_numpy = [w.numpy() for w in waveforms]
            
            # Ý 2: CƠ CHẾ OOM RECOVERY
            try:
                inputs = processor(waveforms_numpy, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
                # inputs['input_values'] = inputs['input_values'].half()
                
                outputs = model(**inputs, output_hidden_states=True)
                stacked = torch.stack(outputs.hidden_states)
                pooled = stacked.mean(dim=2).permute(1, 0, 2).cpu().float()
                
            except torch.cuda.OutOfMemoryError:
                # Nếu batch bị lỗi OOM (thường là batch cuối), xử lý từng file một
                torch.cuda.empty_cache()
                temp_pooled = []
                for w in waveforms_numpy:
                    inp = processor([w], sampling_rate=16000, return_tensors="pt").to(device)
                    # inp['input_values'] = inp['input_values'].half()
                    with torch.no_grad():
                        out = model(**inp, output_hidden_states=True)
                        p = torch.stack(out.hidden_states).mean(dim=2).permute(1, 0, 2).cpu().float()
                        temp_pooled.append(p)
                    torch.cuda.empty_cache()
                pooled = torch.cat(temp_pooled, dim=0)

            shard_embeddings.append(pooled)
            shard_ids.extend(ids)
            shard_names.extend(names)

        # Lưu mảnh vừa xong
        torch.save({
            'embeddings': torch.cat(shard_embeddings, dim=0),
            'speaker_ids': shard_ids,
            'filenames': shard_names,
            'model_name': model_key
        }, shard_path)
        
        # Giải phóng RAM sau mỗi mảnh
        del shard_embeddings, shard_ids, shard_names
        gc.collect()
        torch.cuda.empty_cache()

    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ Hoàn thành toàn bộ dữ liệu cho {model_key}!")