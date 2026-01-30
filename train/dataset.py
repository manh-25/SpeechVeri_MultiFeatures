"""
Dataset loader for Speaker Verification
"""

import torch
from torch.utils.data import Dataset, DataLoader
import random
from config import RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO


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
            embedding_data: Dict with keys 'embeddings', 'speaker_ids', 'filenames'
                           embeddings shape: (N, num_layers, dim)
            feature_data: Dict with same keys, features shape: (N, channels, time_steps)
                         Required for mode 3
            speaker_to_idx: Dict mapping speaker_id to class index
            mode: 1 (PTM only), 2 (Handcrafted only), 3 (Both)
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

        # Validate mode and data
        if mode == 1:
            pass  # Only embeddings needed
        elif mode == 2:
            if feature_data is None:
                raise ValueError("Mode 2 requires feature_data")
        elif mode == 3:
            if feature_data is None:
                raise ValueError("Mode 3 requires feature_data")
            # Validate matching between embeddings and features
            if len(embedding_data["speaker_ids"]) != len(feature_data["speaker_ids"]):
                raise ValueError("Embedding and feature data have different lengths")
            if embedding_data["filenames"] != feature_data["filenames"]:
                raise ValueError("Embedding and feature filenames don't match")
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 1, 2, or 3")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        speaker_id = self.embedding_data["speaker_ids"][idx]
        speaker_label = self.speaker_to_idx[speaker_id]

        if self.mode == 1:
            # PTM only: (num_layers, dim)
            embedding = self.embedding_data["embeddings"][idx].float()
            return {"embedding": embedding, "label": speaker_label}

        elif self.mode == 2:
            # Handcrafted only: (channels, time_steps) -> average pooling across time
            feature = self.feature_data["embeddings"][idx].float()
            # Mean pooling: (channels, time_steps) -> (channels,)
            if feature.dim() > 1:
                feature = feature.mean(dim=-1)
            return {"feature": feature, "label": speaker_label}

        elif self.mode == 3:
            # Both: PTM + Handcrafted
            embedding = self.embedding_data["embeddings"][idx].float()
            feature = self.feature_data["embeddings"][idx].float()
            # Mean pooling for feature
            if feature.dim() > 1:
                feature = feature.mean(dim=-1)
            return {"embedding": embedding, "feature": feature, "label": speaker_label}


def collate_fn_mode1(batch):
    """Collate function for Mode 1 (PTM only)"""
    embeddings = torch.stack([item["embedding"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"embedding": embeddings, "label": labels}


def collate_fn_mode2(batch):
    """Collate function for Mode 2 (Handcrafted only)"""
    features = torch.stack([item["feature"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"feature": features, "label": labels}


def collate_fn_mode3(batch):
    """Collate function for Mode 3 (Both)"""
    embeddings = torch.stack([item["embedding"] for item in batch])
    features = torch.stack([item["feature"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"embedding": embeddings, "feature": features, "label": labels}


def load_data(embedding_path, feature_path=None, mode=1):
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

    feature_data = None
    if mode in [2, 3]:
        if feature_path is None:
            raise ValueError(f"Mode {mode} requires feature_path")
        feature_data = torch.load(feature_path)

    # Build speaker mapping
    unique_speakers = sorted(set(embedding_data["speaker_ids"]))
    speaker_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}

    return embedding_data, feature_data, speaker_to_idx


def create_data_loaders(
    embedding_path, feature_path=None, mode=1, batch_size=64, num_workers=4
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
    embedding_data, feature_data, speaker_to_idx = load_data(
        embedding_path, feature_path, mode
    )

    num_samples = len(embedding_data["speaker_ids"])
    num_speakers = len(speaker_to_idx)

    # Create indices
    indices = list(range(num_samples))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)

    # Split indices
    train_end = int(num_samples * TRAIN_RATIO)
    val_end = train_end + int(num_samples * VAL_RATIO)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create subsets
    train_dataset = SpeakerDataset(
        embedding_data, feature_data, speaker_to_idx, mode
    )
    val_dataset = SpeakerDataset(embedding_data, feature_data, speaker_to_idx, mode)
    test_dataset = SpeakerDataset(embedding_data, feature_data, speaker_to_idx, mode)

    # Create samplers (manual indices)
    from torch.utils.data import Subset

    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)
    test_set = Subset(test_dataset, test_indices)

    # Select collate function based on mode
    if mode == 1:
        collate_fn = collate_fn_mode1
    elif mode == 2:
        collate_fn = collate_fn_mode2
    else:
        collate_fn = collate_fn_mode3

    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, speaker_to_idx, num_speakers
