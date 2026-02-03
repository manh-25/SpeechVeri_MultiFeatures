"""
Dataset loader for Speaker Verification
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import random
import os
import glob
from config import RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO


class SpeakerDataset(Dataset):
    """
    Dataset for loading PTM embeddings and handcrafted features.
    Supports 3 modes:
    - Mode 1: PTM embeddings only
    - Mode 2: Handcrafted features only
    - Mode 3: Both PTM and handcrafted features
    """

    def __init__(self, embedding_data, handcrafted_mapping=None, speaker_to_idx=None, mode=1):
        """
        Args:
            embedding_data: Dict chứa PTM embeddings (đã load từ shard)
            handcrafted_mapping: Dict map từ 'filename' sang 'đường dẫn file .pt'
            mode: 1 (PTM), 2 (Handcrafted), 3 (Both)
        """
        self.mode = mode
        self.embedding_data = embedding_data
        self.handcrafted_mapping = handcrafted_mapping
        self.speaker_to_idx = speaker_to_idx or {}

        # Build speaker_to_idx if not provided
        if not self.speaker_to_idx:
            unique_speakers = sorted(set(embedding_data["speaker_ids"]))
            self.speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}

        self.num_speakers = len(self.speaker_to_idx)
        self.num_samples = len(embedding_data["speaker_ids"])

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
            pt_filename = os.path.splitext(wav_filename)[0] + ".pt"
            if pt_filename not in self.handcrafted_mapping:
                raise FileNotFoundError(f"Không tìm thấy file feature cho {wav_filename}")

            feature_path = self.handcrafted_mapping[pt_filename]
            
            # Load tensor shape (C, T)
            feature = torch.load(feature_path, map_location='cpu').float()
            
            # Đảm bảo có chiều C nếu là 1D
            if feature.dim() == 1:
                feature = feature.unsqueeze(0)
                
            data["feature"] = feature 

        return data


def collate_fn_general(batch, mode):
    """
    Hàm gom batch thông minh: Tự động pad chiều T cho Handcrafted features
    """
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    output = {"label": labels}

    # Gom PTM Embeddings (Fix size: Num_layers, Dim)
    if mode in [1, 3]:
        output["embedding"] = torch.stack([item["embedding"] for item in batch])

    # Gom Handcrafted Features (Dynamic Padding chiều T)
    if mode in [2, 3]:
        features = [item["feature"] for item in batch]
        # Tìm độ dài T lớn nhất trong batch
        max_t = max([f.shape[-1] for f in features])
        
        padded_features = [F.pad(f, (0, max_t - f.shape[-1])) for f in features]
        output["feature"] = torch.stack(padded_features) # Shape: (B, C, T)

    return output


def load_data(embedding_path, feature_dir=None, mode=1):
    """
    Load embedding and feature data from .pt files.

    Args:
        embedding_path: Path to embedding.pt
        feature_path: Path to feature.pt (required for mode 2,3)
        mode: 1, 2, or 3

    Returns:
        embedding_data, feature_data, speaker_to_idx
    """
    embedding_data = torch.load(embedding_path)

    # Quét thư mục Handcrafted để tạo mapping
    handcrafted_mapping = {}
    if mode in [2, 3]:
        if feature_dir is None or not os.path.isdir(feature_dir):
            raise ValueError(f"Mode {mode} yêu cầu feature_dir là đường dẫn thư mục")
        
        print(f"🔍 Đang quét đặc trưng tại: {feature_dir}...")
        all_pt_files = glob.glob(os.path.join(feature_dir, "**", "*.pt"), recursive=True)
        for path in all_pt_files:
            handcrafted_mapping[os.path.basename(path)] = path
        print(f"✅ Đã tìm thấy {len(handcrafted_mapping)} file đặc trưng.")

    # Build speaker mapping
    unique_speakers = sorted(set(embedding_data["speaker_ids"]))
    speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}

    return embedding_data, handcrafted_mapping, speaker_to_idx


def create_data_loaders(
    embedding_path, feature_path=None, mode=1, batch_size=64, num_workers=0
):
    """
    Create train, val, test dataloaders.

    Args:
        embedding_path: Path to embedding.pt
        feature_path: Path to feature.pt (for mode 2,3)
        mode: 1, 2, or 3
        batch_size: Batch size
        num_workers: Number of workers for DataLoader

    Returns:
        train_loader, val_loader, test_loader, speaker_to_idx, num_speakers
    """
    # Load data
    embedding_data, handcrafted_mapping, speaker_to_idx = load_data(embedding_path, feature_path, mode)

    num_samples = len(embedding_data["speaker_ids"])

    # Create indices
    indices = list(range(num_samples))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    # Split indices
    train_end = int(num_samples * TRAIN_RATIO)
    val_end = train_end + int(num_samples * VAL_RATIO)

    full_dataset = SpeakerDataset(embedding_data, handcrafted_mapping, speaker_to_idx, mode)

    train_loader = DataLoader(
        Subset(full_dataset, indices[:train_end]),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=lambda b: collate_fn_general(b, mode), pin_memory=True
    )
    val_loader = DataLoader(
        Subset(full_dataset, indices[train_end:val_end]),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=lambda b: collate_fn_general(b, mode), pin_memory=True
    )
    test_loader = DataLoader(
        Subset(full_dataset, indices[val_end:]),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=lambda b: collate_fn_general(b, mode), pin_memory=True
    )

    return train_loader, val_loader, test_loader, speaker_to_idx, len(speaker_to_idx)
