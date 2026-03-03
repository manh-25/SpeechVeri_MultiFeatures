"""
Dataset loader for Speaker Verification
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import random
import os
import glob
from config import RANDOM_SEED, TRAIN_RATIO, VAL_RATIO
from functools import partial


class SpeakerDataset(Dataset):
    """
    Dataset for loading PTM embeddings and handcrafted features.
    Supports 3 modes:
    - Mode 1: PTM embeddings only
    - Mode 2: Handcrafted features only
    - Mode 3: Both PTM and handcrafted features
    """

    def __init__(self, embedding_data, feature_data=None, speaker_to_idx=None, mode=1):
        """
        Args:
            embedding_data: Dict chứa PTM embeddings (đã load từ shard)
            handcrafted_mapping: Dict map từ 'filename' sang 'đường dẫn file .pt'
            mode: 1 (PTM), 2 (Handcrafted), 3 (Both)
        """
        self.mode = mode
        self.embedding_data = embedding_data
        self.feature_data = feature_data
        self.speaker_to_idx = speaker_to_idx or {}

        # Build speaker_to_idx if not provided
        if not self.speaker_to_idx:
            unique_speakers = sorted(set(embedding_data["speaker_ids"]))
            self.speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}

        self.num_speakers = len(self.speaker_to_idx)
        self.num_samples = len(embedding_data["speaker_ids"])

        self.feature_cache = {}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        speaker_id = self.embedding_data["speaker_ids"][idx]
        speaker_label = self.speaker_to_idx[speaker_id]
        wav_filename = self.embedding_data["filenames"][idx]

        data = {"label": speaker_label}

        # 1. PTM Embedding (Thường đã pooling sẵn từ khâu extract)
        if self.mode in [1, 3]:
            data["embedding"] = self.embedding_data["embeddings"][idx].float()

        # 2. Handcrafted Feature (Giữ nguyên C, T để cho ECAPA-TDNN)
        if self.mode in [2, 3]:
            data["feature"] = self.feature_data["features"][idx].float()

        return data


def collate_fn_general(batch, mode, is_train=True, max_frames=200):
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    output = {"label": labels}

    if mode in [1,3]:
        output["embedding"] = torch.stack([item["embedding"] for item in batch])

    # Gom Handcrafted Features (Dynamic Padding chiều T bằng Replicate)
    if mode in [2,3]:
        features = [item["feature"] for item in batch]
        processed_features = []
        
        # 1. BƯỚC LỌC CHUẨN: Ép tất cả feature về 2D [C, T] ngay từ đầu
        safe_features = []
        for f in features:
            if f.dim() == 1:
                f = f.unsqueeze(0)  # [T] thành [1, T] (VD: pitch)
            safe_features.append(f)
            
        # 2. XỬ LÝ THEO TRAIN/VAL
        if is_train:
            # TỐI ƯU TỐC ĐỘ: Cắt ngẫu nhiên max_frames lúc Train
            for f in safe_features:
                c, t = f.shape
                if t > max_frames:
                    start = random.randint(0, t - max_frames)
                    f = f[:, start:start + max_frames]
                elif t < max_frames:
                    pad_len = max_frames - t
                    f = F.pad(f.unsqueeze(0), (0, pad_len), mode='replicate').squeeze(0)
                processed_features.append(f)
        else:
            # Lúc Val/Test: Giữ nguyên độ dài, nhưng CẮT BỎ phần thừa nếu quá dài
            max_t = max([f.shape[-1] for f in safe_features])
            max_t = min(max_t, 1000) # <-- SAFETY CAP CHỐNG TRÀN VRAM
            
            for f in safe_features:
                if f.shape[-1] > max_t:
                    f = f[:, :max_t] # Cắt cụt nếu dài hơn max_t
                
                pad_len = max_t - f.shape[-1]
                if pad_len > 0:
                    f = F.pad(f.unsqueeze(0), (0, pad_len), mode='replicate').squeeze(0)
                processed_features.append(f)
            
        output["feature"] = torch.stack(processed_features)

    return output


def load_data(embedding_path, feature_dir=None, mode=1):
    embedding_data = {"speaker_ids": [], "filenames": [], "embeddings": []}
    feature_data = {"features": []}

    # 1. LOAD PTM EMBEDDINGS (Hỗ trợ nhiều file Shard)
    if mode in [1, 3]:
        if os.path.isdir(embedding_path):
            print(f"🔍 Đang quét các file shard PTM tại: {embedding_path}...")
            shard_files = glob.glob(os.path.join(embedding_path, "*.pt"))
            
            if not shard_files:
                raise FileNotFoundError(f"Không tìm thấy file .pt nào trong thư mục {embedding_path}")
            
            all_embeddings = []
            for shard in shard_files:
                shard_data = torch.load(shard, map_location='cpu')
                embedding_data["speaker_ids"].extend(shard_data["speaker_ids"])
                embedding_data["filenames"].extend(shard_data["filenames"])
                all_embeddings.append(shard_data["embeddings"])
                
            # Gộp tất cả tensor embedding lại theo chiều dọc (chiều Batch - dim 0)
            embedding_data["embeddings"] = torch.cat(all_embeddings, dim=0)
            # embedding_data["embeddings"].share_memory_()
            print(f"✅ Đã load gộp {len(shard_files)} file shards. Tổng số sample PTM: {len(embedding_data['speaker_ids'])}")
        else:
            # Fallback nếu truyền vào đường dẫn của 1 file duy nhất
            embedding_data = torch.load(embedding_path, map_location='cpu')
            print(f"✅ Đã load 1 file PTM tổng. Tổng số sample PTM: {len(embedding_data['speaker_ids'])}")

    # Quét thư mục Handcrafted để tạo mapping
    handcrafted_mapping = {}
    if mode in [2, 3]:
        if feature_dir is None or not os.path.isdir(feature_dir):
            raise ValueError(f"Mode {mode} yêu cầu feature_dir là đường dẫn thư mục chứa Shards")
        
        print(f"🔍 Đang nạp các file shard Handcrafted từ: {feature_dir}...")
        hc_shards = sorted(glob.glob(os.path.join(feature_dir, "*.pt")))
        
        for shard in hc_shards:
            shard_data = torch.load(shard, map_location='cpu')
            feature_data["features"].extend(shard_data["features"])
            
            # Nếu chạy Mode 2, cần mượn speaker_ids từ tập Handcrafted
            if mode == 2:
                embedding_data["speaker_ids"].extend(shard_data["speaker_ids"])
                embedding_data["filenames"].extend(shard_data["filenames"])
                
        print(f"✅ Đã nạp xong {len(hc_shards)} HC shards vào RAM.")

    unique_speakers = sorted(set(embedding_data["speaker_ids"]))
    speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}

    return embedding_data, feature_data, speaker_to_idx


def create_train_val_loaders(embedding_path, feature_path, mode, batch_size, num_workers=0):
    # Nạp dữ liệu
    embedding_data, feature_data, speaker_to_idx = load_data(embedding_path, feature_path, mode)
    
    # --- LỌC INDEX THEO SPEAKER ID (TRÁNH DATA LEAKAGE) ---
    speaker_ids = embedding_data["speaker_ids"]
    unique_speakers = sorted(set(speaker_ids))
    
    # Xáo trộn danh sách người nói (cố định seed để dễ tái lập)
    shuffled_speakers = sorted(list(unique_speakers))
    
    # Cắt 85% NGƯỜI NÓI cho Train, 15% NGƯỜI NÓI cho Val
    num_train_spk = int(len(shuffled_speakers) * TRAIN_RATIO)
    train_speakers = set(shuffled_speakers[:num_train_spk])
    val_speakers = set(shuffled_speakers[num_train_spk:])
    
    train_indices = []
    val_indices = []
    
    # Phân loại từng mẫu âm thanh về đúng tập dựa trên speaker
    for i, spk in enumerate(speaker_ids):
        if spk in train_speakers:
            train_indices.append(i)
        else:
            val_indices.append(i)
            
    print(f"\n🎤 CHIA DỮ LIỆU THEO OPEN-SET (Unseen Speakers):")
    print(f"   - Tập Train: {len(train_speakers)} speakers ({len(train_indices)} samples)")
    print(f"   - Tập Val:   {len(val_speakers)} speakers ({len(val_indices)} samples)\n")
    
    # Xáo trộn ngẫu nhiên thứ tự sample trong mỗi tập
    random.shuffle(train_indices)
    random.shuffle(val_indices)

    # Khởi tạo Full Dataset (Vẫn giữ số lượng class tổng để không bị lỗi out-of-bounds index)
    full_dataset = SpeakerDataset(embedding_data, feature_data, speaker_to_idx, mode)

    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=partial(collate_fn_general, mode=mode, is_train=True), 
        pin_memory=False
    )
    
    val_loader = DataLoader(
        Subset(full_dataset, val_indices),
        batch_size=32, shuffle=False, num_workers=num_workers,
        collate_fn=partial(collate_fn_general, mode=mode, is_train=False), 
        pin_memory=False
    )
    
    return train_loader, val_loader, speaker_to_idx, len(speaker_to_idx)

def create_test_loader(
    test_embedding_path, test_feature_path=None, mode=1, batch_size=64, num_workers=0
):
    """CHỈ DÙNG LÚC TEST: Nhận data của Unseen Speakers và ném tất cả vào 1 Loader"""
    embedding_data, feature_data, speaker_to_idx = load_data(test_embedding_path, test_feature_path, mode)
    test_dataset = SpeakerDataset(embedding_data, feature_data, speaker_to_idx, mode)
    
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=num_workers,
        # Thay lambda bằng partial
        collate_fn=partial(collate_fn_general, mode=mode, is_train=False), 
        pin_memory=False
    )
    return test_loader, len(speaker_to_idx)