import torch
import torchaudio
import os
import gc
import glob
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import WavLMModel, HubertModel, Wav2Vec2Model

class SpeakerDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
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
        rel_path = os.path.relpath(path, self.folder_path).replace("\\", "/")
        speaker_id = filename.split('_')[0]
        return waveform, speaker_id, filename, rel_path

def collate_fn(batch):
    waveforms, ids, names, rel_paths = zip(*batch)
    
    # Pad sequence siêu tốc bằng PyTorch, chuẩn bị sẵn sàng cho GPU
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    
    # Tạo Attention Mask để model bỏ qua các vùng padding, giữ độ chính xác cao nhất
    lengths = torch.tensor([len(w) for w in waveforms])
    max_len = padded_waveforms.shape[1]
    attention_mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    
    return padded_waveforms, attention_mask.long(), list(ids), list(names), list(rel_paths)


def _normalize_rel_path(path_value: str) -> str:
    return str(path_value).replace("\\", "/").lower().strip()


def _crop_or_pad_temporal(x: torch.Tensor, valid_len: int, target_t: int, train_crop_mode: str):
    # x: (L, T, D)
    valid_len = int(max(1, min(valid_len, x.shape[1])))
    x = x[:, :valid_len, :]

    if valid_len > target_t:
        if train_crop_mode == "random":
            start = random.randint(0, valid_len - target_t)
        else:
            start = 0
        x = x[:, start:start + target_t, :]
        valid_len = target_t
    elif valid_len < target_t:
        pad_t = target_t - valid_len
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_t))

    return x, valid_len

@torch.inference_mode()
def run_extraction_resume(
    model_key,
    folder_path,
    save_dir,
    batch_size=16,
    shard_size=10000,
    time_cap=350,
    train_crop_mode="random",
):
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
        
        shard_embeddings, shard_ids, shard_names, shard_rel_paths, shard_lengths = [], [], [], [], []

        for waveforms, attention_mask, ids, names, rel_paths in tqdm(dataloader, desc=f"Shard {s_idx}"):
            # Đẩy thẳng tensor lên GPU
            waveforms = waveforms.to(device, dtype=torch.bfloat16, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            
            try:
                outputs = model(input_values=waveforms, attention_mask=attention_mask, output_hidden_states=True)
                hidden = torch.stack(outputs.hidden_states, dim=1).float().cpu()  # (B, L, T, D)
                input_lengths = attention_mask.sum(dim=1)
                if hasattr(model, "_get_feat_extract_output_lengths"):
                    out_lengths = model._get_feat_extract_output_lengths(input_lengths).to("cpu")
                else:
                    out_lengths = torch.full(
                        (hidden.shape[0],),
                        hidden.shape[2],
                        dtype=torch.long,
                    )

                packed_samples = []
                packed_lengths = []
                for b in range(hidden.shape[0]):
                    x_b, len_b = _crop_or_pad_temporal(
                        hidden[b],
                        int(out_lengths[b].item()),
                        target_t=time_cap,
                        train_crop_mode=train_crop_mode,
                    )
                    packed_samples.append(x_b.unsqueeze(0))
                    packed_lengths.append(len_b)

                temporal = torch.cat(packed_samples, dim=0)  # (B, L, T_cap, D)
                lengths = torch.tensor(packed_lengths, dtype=torch.long)
                
            except torch.cuda.OutOfMemoryError:
                # OOM Recovery vẫn giữ lại phòng ngừa file âm thanh dài "bất thường"
                torch.cuda.empty_cache()
                temp_temporal = []
                temp_lengths = []
                for i in range(len(waveforms)):
                    w = waveforms[i].unsqueeze(0)
                    mask = attention_mask[i].unsqueeze(0)
                    out = model(input_values=w, attention_mask=mask, output_hidden_states=True)
                    h = torch.stack(out.hidden_states, dim=1).float().cpu().squeeze(0)  # (L, T, D)
                    if hasattr(model, "_get_feat_extract_output_lengths"):
                        out_len = int(model._get_feat_extract_output_lengths(mask.sum(dim=1)).item())
                    else:
                        out_len = h.shape[1]

                    x_i, len_i = _crop_or_pad_temporal(
                        h,
                        out_len,
                        target_t=time_cap,
                        train_crop_mode=train_crop_mode,
                    )
                    temp_temporal.append(x_i.unsqueeze(0))
                    temp_lengths.append(len_i)
                    torch.cuda.empty_cache()
                temporal = torch.cat(temp_temporal, dim=0)
                lengths = torch.tensor(temp_lengths, dtype=torch.long)

            shard_embeddings.append(temporal)
            shard_lengths.append(lengths)
            shard_ids.extend(ids)
            shard_names.extend(names)
            shard_rel_paths.extend(rel_paths)

        torch.save({
            'embeddings': torch.cat(shard_embeddings, dim=0),
            'lengths': torch.cat(shard_lengths, dim=0),
            'speaker_ids': shard_ids,
            'filenames': shard_names,
            'relative_paths': shard_rel_paths,
            'time_cap': int(time_cap),
            'pooling': 'none_keep_time',
            'model_name': model_key
        }, shard_path)
        
        del shard_embeddings, shard_ids, shard_names, shard_rel_paths, shard_lengths
        gc.collect()
        torch.cuda.empty_cache()

    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ Hoàn thành toàn bộ dữ liệu cho {model_key}!")


def build_subset_shards_from_existing(
    source_shard_dir,
    subset_folder_path,
    subset_save_dir,
    model_key,
    shard_size=10000,
):
    """
    Tạo shards cho tập subset bằng cách copy embeddings đã có từ source shards.
    Match theo relative path, KHÔNG chạy model lại.
    """
    os.makedirs(subset_save_dir, exist_ok=True)

    source_shards = sorted(glob.glob(os.path.join(source_shard_dir, "*.pt")))
    if not source_shards:
        raise FileNotFoundError(f"Không tìm thấy source shard trong: {source_shard_dir}")

    print(f"🔎 Index source shards từ: {source_shard_dir}")
    rel_to_location = {}
    for shard_path in tqdm(source_shards, desc="Index source shards"):
        payload = torch.load(shard_path, map_location="cpu")
        rels = payload.get("relative_paths", [])
        if not rels:
            raise ValueError(
                f"Shard {shard_path} không có 'relative_paths'. "
                "Hãy extract source bằng phiên bản mới trước."
            )
        for idx, rel in enumerate(rels):
            rel_to_location[_normalize_rel_path(rel)] = (shard_path, idx)

    subset_wavs = glob.glob(os.path.join(subset_folder_path, "**", "*.wav"), recursive=True)
    subset_wavs.sort(key=lambda x: os.path.getsize(x))

    current_embeddings, current_lengths = [], []
    current_ids, current_names, current_rels = [], [], []
    shard_count = 0
    missing = 0

    cache_path = None
    cache_payload = None

    for wav_path in tqdm(subset_wavs, desc="Build subset shards"):
        rel = os.path.relpath(wav_path, subset_folder_path).replace("\\", "/")
        rel_key = _normalize_rel_path(rel)

        loc = rel_to_location.get(rel_key)
        if loc is None:
            missing += 1
            continue

        src_shard, src_idx = loc
        if cache_path != src_shard:
            cache_payload = torch.load(src_shard, map_location="cpu")
            cache_path = src_shard

        emb = cache_payload["embeddings"][src_idx].float().unsqueeze(0)
        if "lengths" in cache_payload:
            emb_len = cache_payload["lengths"][src_idx].long().view(1)
        else:
            emb_len = torch.tensor([emb.shape[2]], dtype=torch.long)

        filename = os.path.basename(wav_path)
        speaker_id = filename.split("_")[0]

        current_embeddings.append(emb)
        current_lengths.append(emb_len)
        current_ids.append(speaker_id)
        current_names.append(filename)
        current_rels.append(rel)

        if len(current_names) >= shard_size:
            subset_shard_path = os.path.join(subset_save_dir, f"{model_key}_shard_{shard_count}.pt")
            torch.save(
                {
                    "embeddings": torch.cat(current_embeddings, dim=0),
                    "lengths": torch.cat(current_lengths, dim=0),
                    "speaker_ids": current_ids,
                    "filenames": current_names,
                    "relative_paths": current_rels,
                    "time_cap": int(torch.cat(current_lengths, dim=0).max().item()),
                    "pooling": "none_keep_time",
                    "model_name": model_key,
                },
                subset_shard_path,
            )
            current_embeddings, current_lengths = [], []
            current_ids, current_names, current_rels = [], [], []
            shard_count += 1

    if current_names:
        subset_shard_path = os.path.join(subset_save_dir, f"{model_key}_shard_{shard_count}.pt")
        torch.save(
            {
                "embeddings": torch.cat(current_embeddings, dim=0),
                "lengths": torch.cat(current_lengths, dim=0),
                "speaker_ids": current_ids,
                "filenames": current_names,
                "relative_paths": current_rels,
                "time_cap": int(torch.cat(current_lengths, dim=0).max().item()),
                "pooling": "none_keep_time",
                "model_name": model_key,
            },
            subset_shard_path,
        )
        shard_count += 1

    print(
        f"✅ Tạo xong subset shards tại {subset_save_dir}. "
        f"Số shard: {shard_count} | Missing theo relative_path: {missing}"
    )


def run_extraction_raw_and_subset(
    model_key,
    train_raw_folder,
    train_raw_save_dir,
    train_vi_folder,
    train_vi_save_dir,
    batch_size=16,
    shard_size=10000,
    time_cap=350,
    train_crop_mode="random",
):
    """
    1) Extract train_raw (giữ T, có lengths, relative_paths)
    2) Build train_vi_full shards bằng cách copy từ train_raw shards theo relative path.
    """
    run_extraction_resume(
        model_key=model_key,
        folder_path=train_raw_folder,
        save_dir=train_raw_save_dir,
        batch_size=batch_size,
        shard_size=shard_size,
        time_cap=time_cap,
        train_crop_mode=train_crop_mode,
    )

    build_subset_shards_from_existing(
        source_shard_dir=train_raw_save_dir,
        subset_folder_path=train_vi_folder,
        subset_save_dir=train_vi_save_dir,
        model_key=model_key,
        shard_size=shard_size,
    )