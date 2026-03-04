import torch
import torchaudio
import os
import gc
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import WavLMModel, HubertModel, Wav2Vec2Model

class SpeakerDataset(Dataset):
    def __init__(self, folder_path):
        self.file_paths = glob.glob(os.path.join(folder_path, "**", "*.wav"), recursive=True)
        # Giữ nguyên việc sort theo dung lượng để tối ưu padding trong batch
        self.file_paths.sort(key=lambda x: os.path.getsize(x))
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # Data đã được chuẩn hóa 16kHz nên ta loại bỏ bước Resample gây chậm
        waveform, _ = torchaudio.load(path)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
            
        # Tự làm công việc của HuggingFace Processor (Chuẩn hóa Zero-mean, Unit-variance)
        mean = waveform.mean()
        var = waveform.var(unbiased=False)
        waveform = (waveform - mean) / torch.sqrt(var + 1e-7)

        filename = os.path.basename(path)
        speaker_id = filename.split('_')[0]
        return waveform, speaker_id, filename

def collate_fn(batch):
    waveforms, ids, names = zip(*batch)
    
    # Pad sequence siêu tốc bằng PyTorch, chuẩn bị sẵn sàng cho GPU
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    
    # Tạo Attention Mask để model bỏ qua các vùng padding, giữ độ chính xác cao nhất
    lengths = torch.tensor([len(w) for w in waveforms])
    max_len = padded_waveforms.shape[1]
    attention_mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    
    return padded_waveforms, attention_mask.long(), list(ids), list(names)

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
    model = None

    print(f"--- Kiểm tra tiến độ cho {model_key.upper()} ---")
    
    for s_idx in range(num_shards):
        shard_path = os.path.join(save_dir, f"{model_key}_shard_{s_idx}.pt")
        
        if os.path.exists(shard_path):
            print(f"⏩ Shard {s_idx} đã xong. Bỏ qua.")
            continue

        if model is None:
            model = model_class.from_pretrained(
                repo, 
                output_hidden_states=True,
                torch_dtype=torch.bfloat16, # Tuyệt chiêu chống lỗi NaN và giảm 50% VRAM
                attn_implementation="sdpa"  # Kích hoạt Flash Attention
            ).to(device).eval()

        print(f"🚀 Bắt đầu trích xuất Shard {s_idx}/{num_shards-1}...")
        
        start_idx = s_idx * shard_size
        end_idx = min(start_idx + shard_size, total_files)
        subset = Subset(dataset, list(range(start_idx, end_idx)))
        
        # Giảm num_workers xuống 8 để tối ưu I/O cho ổ cứng, tránh overhead
        dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn, num_workers=8, pin_memory=True)
        
        shard_embeddings, shard_ids, shard_names = [], [], []

        for waveforms, attention_mask, ids, names in tqdm(dataloader, desc=f"Shard {s_idx}"):
            # Đẩy thẳng tensor lên GPU
            waveforms = waveforms.to(device, dtype=torch.bfloat16, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            
            try:
                outputs = model(input_values=waveforms, attention_mask=attention_mask, output_hidden_states=True)
                stacked = torch.stack(outputs.hidden_states)
                pooled = stacked.mean(dim=2).permute(1, 0, 2).cpu().float()
                
            except torch.cuda.OutOfMemoryError:
                # OOM Recovery vẫn giữ lại phòng ngừa file âm thanh dài "bất thường"
                torch.cuda.empty_cache()
                temp_pooled = []
                for i in range(len(waveforms)):
                    w = waveforms[i].unsqueeze(0)
                    mask = attention_mask[i].unsqueeze(0)
                    out = model(input_values=w, attention_mask=mask, output_hidden_states=True)
                    p = torch.stack(out.hidden_states).mean(dim=2).permute(1, 0, 2).cpu().float()
                    temp_pooled.append(p)
                    torch.cuda.empty_cache()
                pooled = torch.cat(temp_pooled, dim=0)

            shard_embeddings.append(pooled)
            shard_ids.extend(ids)
            shard_names.extend(names)

        torch.save({
            'embeddings': torch.cat(shard_embeddings, dim=0),
            'speaker_ids': shard_ids,
            'filenames': shard_names,
            'model_name': model_key
        }, shard_path)
        
        del shard_embeddings, shard_ids, shard_names
        gc.collect()
        torch.cuda.empty_cache()

    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ Hoàn thành toàn bộ dữ liệu cho {model_key}!")