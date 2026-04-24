"""
Training script for Speaker Verification
Includes: Training loop, validation, early stopping, LR scheduling, metrics computation
Features: Experiment management, model summary, TensorBoard, gating analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import numpy as np
import os
import json
import csv
import shutil
import hashlib
import time
from datetime import datetime
from tqdm import tqdm
from torchinfo import summary
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from dataset import create_train_val_loaders
import torch.nn.functional as F
from metrics import compute_eer, compute_mindcf
import math

from config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    MIN_LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOP_PATIENCE,
    EARLY_STOP_DELTA,
    LR_SCHEDULER,
    COSINE_T_MAX,
    PLATEAU_PATIENCE,
    PLATEAU_FACTOR,
    OPTIMIZER,
    MOMENTUM,
    NESTEROV,
    DEVICE,
    MIXED_PRECISION,
    LOG_INTERVAL,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
    FINAL_MODEL_NAME,
    LOG_FILE,
    MODE,
    FUSION_METHOD,
    FEATURE_MODE,
    AAM_MARGIN,
    AAM_SCALE,
    PTM_DIM,
    PTM_NUM_LAYERS,
    HANDCRAFTED_DIM,
    DIM_MAP,
    USE_SPK_BALANCED_SAMPLER,
    SPK_PER_BATCH,
    UTT_PER_SPK,
    HARD_NEGATIVE_MINING,
    HARD_NEGATIVE_TOPK,
    HARD_NEGATIVE_WEIGHT,
    HARD_NEGATIVE_MARGIN,
    AUGMENT_PROB,
    FEATURE_NOISE_STD,
    EMBEDDING_NOISE_STD,
)
from model import SpeakerVerificationModel, AAMSoftmaxLoss, get_model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def apply_spec_augment(x, freq_mask_param=15, time_mask_param=30):
    """Áp dụng SpecAugment trực tiếp trên GPU để tăng cường dữ liệu."""
    b, c, t = x.shape
    x_aug = x.clone()
    
    t_mask = torch.randint(0, time_mask_param, (1,)).item()
    if t_mask > 0 and t > t_mask:
        t0 = torch.randint(0, t - t_mask, (1,)).item()
        x_aug[:, :, t0:t0+t_mask] = 0

    f_mask = torch.randint(0, freq_mask_param, (1,)).item()
    if f_mask > 0 and c > f_mask:
        f0 = torch.randint(0, c - f_mask, (1,)).item()
        x_aug[:, f0:f0+f_mask, :] = 0

    return x_aug


def apply_domain_feature_augment(
    x,
    augment_prob=0.5,
    feature_noise_std=0.003,
):
    """Augment trên feature-domain để tăng robust khi không còn waveform gốc."""
    if torch.rand(1).item() >= augment_prob:
        return x

    x_aug = apply_spec_augment(x)

    if feature_noise_std > 0:
        x_aug = x_aug + torch.randn_like(x_aug) * feature_noise_std

    if x_aug.shape[-1] > 16 and torch.rand(1).item() < 0.35:
        drop = max(1, int(x_aug.shape[-1] * 0.08))
        start = torch.randint(0, x_aug.shape[-1] - drop + 1, (1,)).item()
        x_aug[:, :, start:start + drop] = 0

    return x_aug


def apply_embedding_augment(x, augment_prob=0.5, embedding_noise_std=0.002):
    """Nhiễu nhẹ cho PTM embedding để tăng ổn định decision boundary."""
    if torch.rand(1).item() >= augment_prob:
        return x

    x_aug = x
    if embedding_noise_std > 0:
        x_aug = x_aug + torch.randn_like(x_aug) * embedding_noise_std

    if torch.rand(1).item() < 0.25:
        dropout_mask = (torch.rand_like(x_aug) > 0.02).float()
        x_aug = x_aug * dropout_mask

    return x_aug


def compute_hard_negative_loss(embeddings, labels, topk=20, margin=0.1):
    """Hard-negative loss đơn giản trên cosine similarity trong batch."""
    if embeddings.shape[0] < 4:
        return embeddings.new_tensor(0.0)

    emb_norm = F.normalize(embeddings, p=2, dim=1)
    sim = torch.matmul(emb_norm, emb_norm.t())

    labels = labels.view(-1)
    same = labels.unsqueeze(1).eq(labels.unsqueeze(0))
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    pos_mask = same & (~eye)
    neg_mask = ~same

    if not pos_mask.any() or not neg_mask.any():
        return embeddings.new_tensor(0.0)

    pos_count = pos_mask.sum(dim=1)
    pos_mean = (sim * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1).float()

    neg_sim = sim.masked_fill(~neg_mask, -1e4)
    k = min(max(1, int(topk)), neg_sim.shape[1])
    hard_neg = torch.topk(neg_sim, k=k, dim=1).values.mean(dim=1)

    valid_anchor = pos_count > 0
    if not valid_anchor.any():
        return embeddings.new_tensor(0.0)

    loss_vec = F.relu(margin + hard_neg[valid_anchor] - pos_mean[valid_anchor])
    return loss_vec.mean()

def update_aam_margin(criterion, epoch, final_margin=AAM_MARGIN, warmup_epochs=15):
    """Tăng dần Margin từ 0.0 lên mức tối đa để mô hình không bị 'sốc' loss ở những epoch đầu."""
    if epoch >= warmup_epochs:
        current_margin = final_margin
    else:
        current_margin = final_margin * (epoch / warmup_epochs)
    
    criterion.margin = current_margin
    criterion.cos_m = math.cos(current_margin)
    criterion.sin_m = math.sin(current_margin)
    criterion.th = math.cos(math.pi - current_margin)
    criterion.mm = math.sin(math.pi - current_margin) * current_margin
    
    return current_margin


def save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]
    return model, optimizer, epoch, best_loss


def _count_params_and_memory(model: nn.Module):
    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total_param_bytes = int(sum(p.numel() * p.element_size() for p in model.parameters()))
    trainable_param_bytes = int(sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad))
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_param_memory_bytes": total_param_bytes,
        "total_param_memory_mb": float(total_param_bytes / (1024 ** 2)),
        "trainable_param_memory_bytes": trainable_param_bytes,
        "trainable_param_memory_mb": float(trainable_param_bytes / (1024 ** 2)),
    }


def _build_inputs_from_batch(batch_data, device, non_blocking=False):
    labels = batch_data["label"].to(device, non_blocking=non_blocking)
    inputs = {
        k: v.to(device, non_blocking=non_blocking)
        for k, v in batch_data.items()
        if isinstance(v, torch.Tensor) and k != "label"
    }
    return inputs, labels


def _profile_gflops_per_sample(model, criterion, batch_data, device, use_mixed_precision=False):
    """Profile actual FLOPs for one representative forward pass and normalize per sample."""
    if batch_data is None:
        return 0.0

    try:
        from torch.profiler import profile, ProfilerActivity
    except Exception:
        return 0.0

    model_was_training = model.training
    model.eval()

    inputs, labels = _build_inputs_from_batch(batch_data, device, non_blocking=False)
    batch_size = int(labels.shape[0]) if labels is not None else 0
    if batch_size <= 0:
        if model_was_training:
            model.train()
        return 0.0

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    try:
        with torch.no_grad():
            with profile(activities=activities, with_flops=True, record_shapes=False) as prof:
                if use_mixed_precision and torch.cuda.is_available() and str(device).startswith("cuda"):
                    with autocast('cuda'):
                        _, embeddings = model(**inputs)
                    _ = criterion(None, labels, embeddings=embeddings.float())
                else:
                    _, embeddings = model(**inputs)
                    _ = criterion(None, labels, embeddings=embeddings)

        total_flops = 0.0
        for evt in prof.key_averages():
            evt_flops = getattr(evt, "flops", 0) or 0
            total_flops += float(evt_flops)

        gflops_per_sample = (total_flops / 1e9) / float(batch_size)
        return float(max(0.0, gflops_per_sample))
    except Exception:
        return 0.0
    finally:
        if model_was_training:
            model.train()


def _write_results_csv(exp_dir, final_results):
    csv_path = os.path.join(exp_dir, "results.csv")
    flat_row = {
        "exp_name": final_results.get("exp_name"),
        "timestamp": final_results.get("timestamp"),
        "mode": final_results.get("config", {}).get("mode"),
        "fusion_method": final_results.get("config", {}).get("fusion_method"),
        "feature_mode": final_results.get("config", {}).get("feature_mode"),
        "duration": final_results.get("config", {}).get("duration"),
        "pretrained_model": final_results.get("config", {}).get("pretrained_model"),
        "best_val_eer": final_results.get("best_val_eer"),
        "best_val_mindcf": final_results.get("best_val_mindcf"),
        "epochs_trained": final_results.get("epochs_trained"),
        "total_train_time_sec": final_results.get("performance", {}).get("total_train_time_sec"),
        "avg_train_epoch_time_sec": final_results.get("performance", {}).get("avg_train_epoch_time_sec"),
        "avg_val_epoch_time_sec": final_results.get("performance", {}).get("avg_val_epoch_time_sec"),
        "gflops_per_sample": final_results.get("performance", {}).get("gflops_per_sample"),
        "gflops_total": final_results.get("performance", {}).get("gflops_total"),
        "peak_gpu_memory_allocated_mb": final_results.get("performance", {}).get("peak_gpu_memory_allocated_mb"),
        "total_params": final_results.get("model_stats", {}).get("total_params"),
        "trainable_params": final_results.get("model_stats", {}).get("trainable_params"),
        "total_param_memory_mb": final_results.get("model_stats", {}).get("total_param_memory_mb"),
        "trainable_param_memory_mb": final_results.get("model_stats", {}).get("trainable_param_memory_mb"),
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
        writer.writeheader()
        writer.writerow(flat_row)

    return csv_path


def _update_experiment_mode_summary(experiment_root, final_results, exp_dir):
    """Maintain a summary file for all mode runs under one experiment root."""
    os.makedirs(experiment_root, exist_ok=True)
    summary_row = {
        "exp_name": final_results.get("exp_name"),
        "exp_dir": exp_dir,
        "timestamp": final_results.get("timestamp"),
        "mode": final_results.get("config", {}).get("mode"),
        "fusion_method": final_results.get("config", {}).get("fusion_method"),
        "feature_mode": final_results.get("config", {}).get("feature_mode"),
        "duration": final_results.get("config", {}).get("duration"),
        "pretrained_model": final_results.get("config", {}).get("pretrained_model"),
        "best_val_eer": final_results.get("best_val_eer"),
        "best_val_mindcf": final_results.get("best_val_mindcf"),
        "epochs_trained": final_results.get("epochs_trained"),
        "total_train_time_sec": final_results.get("performance", {}).get("total_train_time_sec"),
        "avg_train_epoch_time_sec": final_results.get("performance", {}).get("avg_train_epoch_time_sec"),
        "avg_val_epoch_time_sec": final_results.get("performance", {}).get("avg_val_epoch_time_sec"),
        "gflops_per_sample": final_results.get("performance", {}).get("gflops_per_sample"),
        "gflops_total": final_results.get("performance", {}).get("gflops_total"),
        "peak_gpu_memory_allocated_mb": final_results.get("performance", {}).get("peak_gpu_memory_allocated_mb"),
        "total_params": final_results.get("model_stats", {}).get("total_params"),
        "trainable_params": final_results.get("model_stats", {}).get("trainable_params"),
        "total_param_memory_mb": final_results.get("model_stats", {}).get("total_param_memory_mb"),
        "trainable_param_memory_mb": final_results.get("model_stats", {}).get("trainable_param_memory_mb"),
    }

    summary_json = os.path.join(experiment_root, "summary_all_modes.json")
    summary_csv = os.path.join(experiment_root, "summary_all_modes.csv")

    rows = []
    if os.path.exists(summary_json):
        try:
            with open(summary_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
                if isinstance(payload, list):
                    rows = payload
        except Exception:
            rows = []

    rows = [r for r in rows if r.get("exp_dir") != exp_dir]
    rows.append(summary_row)
    rows = sorted(rows, key=lambda x: (str(x.get("mode", "")), str(x.get("exp_name", ""))))

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if len(rows) > 0:
        fieldnames = list(rows[0].keys())
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return summary_json, summary_csv


def _map_stageA_to_mode3_state_dict(stageA_state_dict, backbone_target_prefix: str):
    """Map a Mode1/Mode2 state_dict into Mode3 branch prefixes.

    Mode1 keys: ptm_encoder.*, backbone.*
    Mode2 keys: handcrafted_encoder.*, backbone.*

    For Mode3, backbone is split into ptm_backbone.* and hc_backbone.*.
    """
    mapped = {}
    for key, value in stageA_state_dict.items():
        if key.startswith("backbone."):
            mapped[f"{backbone_target_prefix}." + key[len("backbone."):]] = value
        else:
            mapped[key] = value
    return mapped


def _augment_mode1_to_mode3_temporal_mapping(model, stageA_state_dict, mapped_sd):
    """Best-effort key remap from Mode1 PTM head to Mode3 temporal PTM encoder.

    Current architectures differ:
    - Mode1: ptm_encoder.pool.score.*, ptm_encoder.in_norm.*
    - Mode3: ptm_temporal_encoder.layer_pool.score.*, ptm_temporal_encoder.out_norm.*

    We remap only exact shape-compatible tensors.
    """
    try:
        model_sd = model.state_dict()
    except Exception:
        return mapped_sd

    # Decouple destination from source to avoid in-loop mutation on the iterated dict.
    mapped_sd = dict(mapped_sd)

    alias_rules = (
        ("ptm_encoder.pool.score.", "ptm_temporal_encoder.layer_pool.score."),
        ("ptm_encoder.in_norm.", "ptm_temporal_encoder.out_norm."),
    )

    for src_prefix, dst_prefix in alias_rules:
        for key, value in list(stageA_state_dict.items()):
            if not key.startswith(src_prefix):
                continue
            dst_key = dst_prefix + key[len(src_prefix):]
            if dst_key not in model_sd:
                continue

            src_shape = tuple(value.shape) if hasattr(value, "shape") else None
            dst_shape = tuple(model_sd[dst_key].shape) if hasattr(model_sd[dst_key], "shape") else None
            if src_shape is None or dst_shape is None or src_shape != dst_shape:
                continue

            mapped_sd[dst_key] = value

    return mapped_sd


def _model_has_prefix(model, prefix: str) -> bool:
    try:
        for key in model.state_dict().keys():
            if key.startswith(prefix):
                return True
    except Exception:
        return False
    return False


def try_initialize_mode3_from_stageA(model, args, device, exp_dir):
    """Optional warm-start for Mode3 from already-trained Mode1 + Mode2 checkpoints.

    This is best-effort: if checkpoints aren't found, training proceeds from scratch.
    """
    if getattr(args, "mode", None) != 3:
        return None

    if not bool(getattr(args, "use_mode3_warmstart", True)):
        print("ℹ Mode3 warm-start disabled by args.use_mode3_warmstart=False")
        return {
            "mode1_ckpt": None,
            "mode2_ckpt": None,
            "loaded": {"mode1": False, "mode2": False},
            "disabled": True,
        }

    init_report = {
        "mode1_ckpt": None,
        "mode2_ckpt": None,
        "loaded": {
            "mode1": False,
            "mode2": False,
        },
    }

    output_dir = getattr(args, "output_dir", "./outputs")
    duration = getattr(args, "duration", "")
    pretrained_model = getattr(args, "pretrained_model", "")
    feature_mode = getattr(args, "feature_mode", "")

    # Allow explicit override paths
    mode1_ckpt = getattr(args, "init_mode1_ckpt", None)
    mode2_ckpt = getattr(args, "init_mode2_ckpt", None)

    allow_auto_fallback = bool(getattr(args, "use_mode3_auto_fallback_ckpt", False))
    allow_auto_fallback = allow_auto_fallback or (os.getenv("SV_MODE3_AUTO_WARMSTART", "0") == "1")

    if allow_auto_fallback and (not mode1_ckpt) and bool(getattr(args, "use_mode3_init_mode1", True)):
        exp_name = f"Mode1_PTM_{duration}_{pretrained_model}"
        mode1_ckpt = os.path.join(output_dir, "experiments", exp_name, BEST_MODEL_NAME)
    if allow_auto_fallback and (not mode2_ckpt) and bool(getattr(args, "use_mode3_init_mode2", True)):
        exp_name = f"Mode2_HC_{duration}_{feature_mode}"
        mode2_ckpt = os.path.join(output_dir, "experiments", exp_name, BEST_MODEL_NAME)

    init_report["mode1_ckpt"] = mode1_ckpt
    init_report["mode2_ckpt"] = mode2_ckpt

    # Load Mode1 -> (ptm_encoder, ptm_backbone)
    if mode1_ckpt and os.path.exists(mode1_ckpt):
        checkpoint = torch.load(mode1_ckpt, map_location=device)
        stageA_sd = checkpoint.get("model_state_dict", checkpoint)

        stageA_sd = {k: v for k, v in stageA_sd.items() if k.startswith("ptm_encoder.") or k.startswith("backbone.")}
        if _model_has_prefix(model, "ptm_backbone."):
            mapped_sd = _map_stageA_to_mode3_state_dict(stageA_sd, backbone_target_prefix="ptm_backbone")
        else:
            # Mode3 PTM branch may not have a backbone (e.g., PTM head only)
            mapped_sd = stageA_sd

        incompatible = model.load_state_dict(mapped_sd, strict=False)
        init_report["loaded"]["mode1"] = True
        init_report["mode1_incompatible_keys"] = {
            "missing_keys": list(getattr(incompatible, "missing_keys", [])),
            "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
        }
        print(f"✓ Initialized Mode3 PTM branch from: {mode1_ckpt}")
    else:
        print(f"ℹ Mode1 checkpoint not found (skip init): {mode1_ckpt}")

    # Load Mode2 -> (handcrafted_encoder, hc_backbone)
    if mode2_ckpt and os.path.exists(mode2_ckpt):
        checkpoint = torch.load(mode2_ckpt, map_location=device)
        stageA_sd = checkpoint.get("model_state_dict", checkpoint)

        stageA_sd = {k: v for k, v in stageA_sd.items() if k.startswith("handcrafted_encoder.") or k.startswith("backbone.")}
        if _model_has_prefix(model, "hc_backbone."):
            mapped_sd = _map_stageA_to_mode3_state_dict(stageA_sd, backbone_target_prefix="hc_backbone")
        else:
            mapped_sd = stageA_sd

        incompatible = model.load_state_dict(mapped_sd, strict=False)
        init_report["loaded"]["mode2"] = True
        init_report["mode2_incompatible_keys"] = {
            "missing_keys": list(getattr(incompatible, "missing_keys", [])),
            "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
        }
        print(f"✓ Initialized Mode3 HC branch from: {mode2_ckpt}")
    else:
        print(f"ℹ Mode2 checkpoint not found (skip init): {mode2_ckpt}")

    try:
        with open(os.path.join(exp_dir, "init_report.json"), "w", encoding="utf-8") as f:
            json.dump(init_report, f, indent=2)
    except Exception as e:
        print(f"⚠ Could not write init_report.json: {e}")

    return init_report


def _build_fixed_val_trials(val_labels, max_pos_pairs=20000, random_seed=42):
    """Tạo trial list cố định chỉ từ tập validation (không dùng sample train)."""
    from collections import defaultdict
    import random

    val_speaker_indices = defaultdict(list)
    for idx, label in enumerate(val_labels):
        val_speaker_indices[int(label)].append(idx)

    labels_unique = list(val_speaker_indices.keys())
    if len(labels_unique) < 2:
        raise ValueError("Validation set cần >= 2 speakers để tạo positive/negative trials.")

    rng = random.Random(random_seed)
    pos_pairs, neg_pairs = [], []

    # Avoid O(n^2) full pair materialization for large speakers in validation.
    pos_speakers = [spk for spk, idxs in val_speaker_indices.items() if len(idxs) > 1]
    if not pos_speakers:
        raise ValueError("Validation set không có speaker nào đủ >= 2 sample để tạo positive pairs.")

    spk_weights = []
    for spk in pos_speakers:
        n = len(val_speaker_indices[spk])
        spk_weights.append((n * (n - 1)) // 2)

    pos_target = max(1, int(max_pos_pairs))
    pos_seen = set()
    max_attempts = max(pos_target * 30, 10000)
    attempts = 0

    while len(pos_pairs) < pos_target and attempts < max_attempts:
        attempts += 1
        spk = rng.choices(pos_speakers, weights=spk_weights, k=1)[0]
        idx1, idx2 = rng.sample(val_speaker_indices[spk], 2)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        key = (idx1, idx2)
        if key in pos_seen:
            continue
        pos_seen.add(key)
        pos_pairs.append(key)

    if len(pos_pairs) == 0:
        raise ValueError("Không tạo được positive validation pairs.")

    neg_seen = set()
    neg_target = len(pos_pairs)
    neg_attempts = 0
    max_neg_attempts = max(neg_target * 40, 10000)
    while len(neg_pairs) < neg_target and neg_attempts < max_neg_attempts:
        neg_attempts += 1
        spk1, spk2 = rng.sample(labels_unique, 2)
        idx1 = rng.choice(val_speaker_indices[spk1])
        idx2 = rng.choice(val_speaker_indices[spk2])
        key = (idx1, idx2) if idx1 <= idx2 else (idx2, idx1)
        if key in neg_seen:
            continue
        neg_seen.add(key)
        neg_pairs.append((idx1, idx2))

    if len(neg_pairs) < len(pos_pairs):
        # Fallback: allow duplicate negatives to preserve balanced trial list size.
        while len(neg_pairs) < len(pos_pairs):
            spk1, spk2 = rng.sample(labels_unique, 2)
            idx1 = rng.choice(val_speaker_indices[spk1])
            idx2 = rng.choice(val_speaker_indices[spk2])
            neg_pairs.append((idx1, idx2))

    return {"pos_pairs": pos_pairs, "neg_pairs": neg_pairs}


def _extract_val_filenames_from_loader(val_loader):
    """Lấy danh sách filename theo đúng thứ tự sample của val_loader (phục vụ export trial text)."""
    val_dataset = val_loader.dataset
    if not hasattr(val_dataset, "indices") or not hasattr(val_dataset, "dataset"):
        raise ValueError("val_loader.dataset phải là torch.utils.data.Subset để trích xuất filename ổn định.")

    base_dataset = val_dataset.dataset
    if not hasattr(base_dataset, "embedding_data") or "filenames" not in base_dataset.embedding_data:
        raise ValueError("Không tìm thấy embedding_data['filenames'] trong dataset gốc.")

    all_filenames = base_dataset.embedding_data["filenames"]
    return [str(all_filenames[idx]) for idx in val_dataset.indices]


def _write_trials_protocol_csv(trials_csv_file, val_trials, val_filenames):
    """
    Ghi trial list theo format CSV 1 cột:
    "<label> <path_1> <path_2>"
    label=1: cùng speaker, label=0: khác speaker
    """
    with open(trials_csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for idx1, idx2 in val_trials["pos_pairs"]:
            writer.writerow([f"1 {val_filenames[idx1]} {val_filenames[idx2]}"])
        for idx1, idx2 in val_trials["neg_pairs"]:
            writer.writerow([f"0 {val_filenames[idx1]} {val_filenames[idx2]}"])


def get_or_create_fixed_val_trials(val_loader, args, max_pos_pairs=10000, random_seed=42):
    """
    Tạo/lưu trial list cố định theo signature của validation set và tái sử dụng cho mọi lần train.
    Đảm bảo không leak train->val vì trial chỉ sinh từ val_loader.
    """
    val_labels = None

    # Fast path: derive validation labels directly from Subset metadata
    # to avoid iterating val_loader (which triggers heavy feature disk I/O).
    val_subset = getattr(val_loader, "dataset", None)
    if hasattr(val_subset, "indices") and hasattr(val_subset, "dataset"):
        base_dataset = val_subset.dataset
        speaker_ids = getattr(getattr(base_dataset, "embedding_data", {}), "get", lambda *_: [])("speaker_ids", [])
        speaker_to_idx = getattr(base_dataset, "speaker_to_idx", None)
        if speaker_ids and isinstance(speaker_to_idx, dict) and len(speaker_to_idx) > 0:
            val_labels = [
                int(speaker_to_idx[speaker_ids[idx]])
                for idx in val_subset.indices
                if speaker_ids[idx] in speaker_to_idx
            ]

    # Fallback for non-Subset / unexpected dataset wrappers.
    if val_labels is None:
        val_labels = []
        for batch in val_loader:
            val_labels.extend([int(x) for x in batch["label"].tolist()])

    label_bytes = json.dumps(val_labels, separators=(",", ":")).encode("utf-8")
    val_signature = hashlib.sha1(label_bytes).hexdigest()[:12]

    duration_tag = getattr(args, "duration", "unknown") or "unknown"
    num_samples = len(val_labels)
    num_speakers = len(set(val_labels))
    val_filenames = _extract_val_filenames_from_loader(val_loader)

    trials_dir = os.path.join(args.output_dir, "fixed_val_trials")
    os.makedirs(trials_dir, exist_ok=True)

    trials_file = os.path.join(
        trials_dir,
        f"val_trials_{duration_tag}_{num_samples}s_{num_speakers}spk_{val_signature}.json",
    )
    trials_csv_file = os.path.join(
        trials_dir,
        f"val_trials_{duration_tag}_{num_samples}s_{num_speakers}spk_{val_signature}.csv",
    )

    if os.path.exists(trials_file):
        with open(trials_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        saved_max_pairs = payload.get("meta", {}).get("max_pos_pairs")
        if saved_max_pairs == max_pos_pairs:
            val_trials = payload["trials"]
            print(f"✓ Đã nạp fixed validation trials: {trials_file}")
            _write_trials_protocol_csv(trials_csv_file, val_trials, val_filenames)
            print(f"✓ Đã đồng bộ trial protocol csv: {trials_csv_file}")
        else:
            print(
                f"ℹ Fixed trials cũ dùng max_pos_pairs={saved_max_pairs}, "
                f"đang tạo lại theo cấu hình mới={max_pos_pairs}."
            )
            val_trials = _build_fixed_val_trials(
                val_labels=val_labels,
                max_pos_pairs=max_pos_pairs,
                random_seed=random_seed,
            )
            payload = {
                "meta": {
                    "duration": duration_tag,
                    "num_samples": num_samples,
                    "num_speakers": num_speakers,
                    "max_pos_pairs": max_pos_pairs,
                    "random_seed": random_seed,
                    "val_signature": val_signature,
                    "created_at": datetime.now().isoformat(),
                },
                "trials": val_trials,
            }
            with open(trials_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            _write_trials_protocol_csv(trials_csv_file, val_trials, val_filenames)
            print(f"✓ Đã cập nhật fixed validation trials: {trials_file}")
            print(f"✓ Đã cập nhật trial protocol csv: {trials_csv_file}")
    else:
        val_trials = _build_fixed_val_trials(
            val_labels=val_labels,
            max_pos_pairs=max_pos_pairs,
            random_seed=random_seed,
        )

        payload = {
            "meta": {
                "duration": duration_tag,
                "num_samples": num_samples,
                "num_speakers": num_speakers,
                "max_pos_pairs": max_pos_pairs,
                "random_seed": random_seed,
                "val_signature": val_signature,
                "created_at": datetime.now().isoformat(),
            },
            "trials": val_trials,
        }
        with open(trials_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        _write_trials_protocol_csv(trials_csv_file, val_trials, val_filenames)

        print(f"✓ Đã tạo và lưu fixed validation trials: {trials_file}")
        print(f"✓ Đã tạo trial protocol csv: {trials_csv_file}")

    print(
        f"  Positive pairs: {len(val_trials['pos_pairs'])} | "
        f"Negative pairs: {len(val_trials['neg_pairs'])}"
    )
    return val_trials


def compute_metrics(logits, labels):
    """
    Compute classification metrics.

    Args:
        logits: (batch_size, num_speakers)
        labels: (batch_size,)

    Returns:
        accuracy: float
    """
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).float().mean().item()
    return accuracy


def plot_confusion_matrix(cm_tensor, epoch, stage, exp_dir):
    """Create and save confusion matrix visualization"""
    cm = cm_tensor.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {stage} - Epoch {epoch}')
    plt.tight_layout()
    
    os.makedirs(os.path.join(exp_dir, "confusion_matrices"), exist_ok=True)
    plt.savefig(os.path.join(exp_dir, "confusion_matrices", f"{stage}_epoch_{epoch}.png"))
    plt.close(fig)
    return fig


# ============================================================================
# TRAINING & VALIDATION
# ============================================================================
class EarlyStopping:
    """Early stopping callback"""

    def __init__(self, patience=EARLY_STOP_PATIENCE, delta=EARLY_STOP_DELTA, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    scaler,
    epoch,
    device,
    log_interval=LOG_INTERVAL,
    use_augment=True,
    augment_prob=AUGMENT_PROB,
    feature_noise_std=FEATURE_NOISE_STD,
    embedding_noise_std=EMBEDDING_NOISE_STD,
    hard_negative_mining=HARD_NEGATIVE_MINING,
    hard_negative_topk=HARD_NEGATIVE_TOPK,
    hard_negative_weight=HARD_NEGATIVE_WEIGHT,
    hard_negative_margin=HARD_NEGATIVE_MARGIN,
    use_mixed_precision=False,
    non_blocking_transfer=False,
    max_steps_per_epoch=None,
):
    """
    Train for one epoch.

    Args:
        model: nn.Module
        train_loader: DataLoader
        optimizer: Optimizer
        criterion: Loss function
        scaler: GradScaler for mixed precision
        epoch: Current epoch number
        device: Device to train on
        log_interval: Logging interval

    Returns:
        avg_loss: Average training loss
        avg_accuracy: Average training accuracy
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    all_logits = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)

    max_steps_per_epoch = None if max_steps_per_epoch is None else max(1, int(max_steps_per_epoch))

    for batch_idx, batch_data in enumerate(progress_bar):
        if max_steps_per_epoch is not None and batch_idx >= max_steps_per_epoch:
            break

        # Move data to device
        labels = batch_data["label"].to(device, non_blocking=non_blocking_transfer)
        inputs = {
            k: v.to(device, non_blocking=non_blocking_transfer)
            for k, v in batch_data.items()
            if isinstance(v, torch.Tensor) and k != "label"
        }

        if use_augment and "feature" in inputs:
            inputs["feature"] = apply_domain_feature_augment(
                inputs["feature"],
                augment_prob=augment_prob,
                feature_noise_std=feature_noise_std,
            )
        if use_augment and "embedding" in inputs:
            inputs["embedding"] = apply_embedding_augment(
                inputs["embedding"],
                augment_prob=augment_prob * 0.8,
                embedding_noise_std=embedding_noise_std,
            )

        if "embedding" in inputs and torch.isnan(inputs["embedding"]).any():
            print(f"\n🚨 PHÁT HIỆN DỮ LIỆU PTM BỊ NaN Ở BATCH {batch_idx}! Đã bỏ qua batch này.")
            continue
        if "feature" in inputs and torch.isnan(inputs["feature"]).any():
            print(f"\n🚨 PHÁT HIỆN FEATURE HC BỊ NaN Ở BATCH {batch_idx}! Đã bỏ qua batch này.")
            continue
        
        optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision
        if use_mixed_precision and scaler is not None:
            with autocast('cuda'):
                _, embeddings = model(**inputs)
            loss, logits = criterion(None, labels, embeddings=embeddings.float())
        else:
            _, embeddings = model(**inputs)
            loss, logits = criterion(None, labels, embeddings=embeddings)

        if (not torch.isfinite(embeddings).all()) or (not torch.isfinite(logits).all()) or (not torch.isfinite(loss)):
            print(f"\n[WARN] Non-finite forward output at batch {batch_idx}. Skip this batch to avoid NaN poisoning.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if hard_negative_mining:
            hnm_loss = compute_hard_negative_loss(
                embeddings=embeddings.float(),
                labels=labels,
                topk=hard_negative_topk,
                margin=hard_negative_margin,
            )
            if not torch.isfinite(hnm_loss):
                print(f"\n[WARN] Non-finite hard-negative loss at batch {batch_idx}. Skip this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss = loss + hard_negative_weight * hnm_loss

        if not torch.isfinite(loss):
            print(f"\n[WARN] Non-finite total loss at batch {batch_idx}. Skip this batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        # Backward pass
        if use_mixed_precision and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Metrics
        accuracy = compute_metrics(logits, labels)
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1
        
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            avg_acc = total_accuracy / num_batches
            progress_bar.set_postfix(
                {"loss": f"{avg_loss:.4f}", "accuracy": f"{avg_acc:.4f}"}
            )

    if num_batches == 0:
        print("[WARN] No valid training batch in this epoch (all skipped). Return NaN loss.")
        return float("nan"), 0.0

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def validate(model, val_loader, device, val_trials):
    """
    Validate Open-set với Trial List cố định
    """
    model.eval()
    total_samples = len(val_loader.dataset)
    all_embeddings = None
    write_offset = 0
    
    # 1. Trích xuất embedding theo đúng thứ tự của val_loader
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation (EER)", leave=False)
        for batch_data in progress_bar:
            inputs = {
                k: v.to(device, non_blocking=True)
                for k, v in batch_data.items()
                if isinstance(v, torch.Tensor) and k != "label"
            }
            _, embeddings = model(**inputs)

            # Normalize and move per-batch to avoid torch.cat peak RAM duplication.
            emb_np = (
                F.normalize(embeddings, p=2, dim=1)
                .detach()
                .to("cpu")
                .numpy()
                .astype(np.float32, copy=False)
            )

            if all_embeddings is None:
                emb_dim = int(emb_np.shape[1])
                all_embeddings = np.empty((total_samples, emb_dim), dtype=np.float32)

            batch_n = int(emb_np.shape[0])
            end = min(write_offset + batch_n, total_samples)
            copy_n = max(0, end - write_offset)
            if copy_n > 0:
                all_embeddings[write_offset:end, :] = emb_np[:copy_n]
                write_offset = end

    if all_embeddings is None or write_offset == 0:
        raise RuntimeError("Validation failed: no embeddings extracted from val_loader.")

    if write_offset < total_samples:
        all_embeddings = all_embeddings[:write_offset, :]

    # 2. Tính Cosine Similarity dựa trên danh sách cặp cố định (vectorized)
    pos_pairs = np.asarray(val_trials["pos_pairs"], dtype=np.int64)
    neg_pairs = np.asarray(val_trials["neg_pairs"], dtype=np.int64)

    pos_scores = (
        np.sum(all_embeddings[pos_pairs[:, 0]] * all_embeddings[pos_pairs[:, 1]], axis=1)
        if pos_pairs.size > 0
        else np.empty((0,), dtype=np.float32)
    )
    neg_scores = (
        np.sum(all_embeddings[neg_pairs[:, 0]] * all_embeddings[neg_pairs[:, 1]], axis=1)
        if neg_pairs.size > 0
        else np.empty((0,), dtype=np.float32)
    )

    scores = np.concatenate([pos_scores, neg_scores], axis=0)
    y_true = np.concatenate(
        [
            np.ones((len(pos_scores),), dtype=np.int32),
            np.zeros((len(neg_scores),), dtype=np.int32),
        ],
        axis=0,
    )

    # 3. Tính toán metrics
    eer_out = compute_eer(y_true, scores)
    mindcf_out = compute_mindcf(y_true, scores, p_target=0.05)

    eer = eer_out[0] if isinstance(eer_out, tuple) else eer_out
    mindcf = mindcf_out[0] if isinstance(mindcf_out, tuple) else mindcf_out

    return float(eer * 100), float(mindcf)

# ============================================================================
# GATING ANALYSIS
# ============================================================================
def analyze_gating_behavior(model, loader, device, exp_dir):
    """
    Analyze gating mechanism - understand how model weights PTM vs Handcrafted
    """
    if model.mode != 3 or model.fusion_method != "gating":
        return None, None, None
    
    model.eval()
    all_gates = []
    all_labels = []
    
    print("\nAnalyzing gating weights...")
    with torch.no_grad():
        for batch_data in tqdm(loader, leave=False):
            labels = batch_data["label"].to(device)
            inputs = {
                k: v.to(device)
                for k, v in batch_data.items()
                if isinstance(v, torch.Tensor) and k != "label"
            }

            _, speaker_embedding, gate_weights = model(return_gates=True, **inputs)

            if gate_weights is None:
                continue

            # Robust reduction: convert any gate tensor shape to one scalar per sample.
            gate_tensor = gate_weights.detach().float()
            batch_n = int(labels.shape[0])

            if gate_tensor.ndim == 0:
                gate_per_sample = gate_tensor.view(1).repeat(batch_n)
            elif gate_tensor.shape[0] == batch_n:
                if gate_tensor.ndim == 1:
                    gate_per_sample = gate_tensor
                else:
                    reduce_dims = tuple(range(1, gate_tensor.ndim))
                    gate_per_sample = gate_tensor.mean(dim=reduce_dims)
            else:
                # Fallback for unexpected layout: flatten then match batch size when possible.
                flat = gate_tensor.reshape(-1)
                if int(flat.numel()) == batch_n:
                    gate_per_sample = flat
                else:
                    gate_per_sample = flat.mean().view(1).repeat(batch_n)

            all_gates.extend(gate_per_sample.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_gates = np.asarray(all_gates, dtype=np.float32)
    all_labels = np.asarray(all_labels, dtype=np.int64)

    if all_gates.size == 0:
        print("  [WARN] Gating analysis skipped: no valid gate values collected.")
        return None, None
    
    # Plot gate distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_gates.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Neutral (0.5)')
    ax.set_xlabel('Gate Value (PTM Weight)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Gating Behavior: PTM vs Handcrafted\n(>0.5: Trust PTM, <0.5: Trust Handcrafted)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.join(exp_dir, "gating_analysis"), exist_ok=True)
    plt.savefig(os.path.join(exp_dir, "gating_analysis", "gate_distribution.png"))
    plt.close()
    
    ptm_priority = np.sum(all_gates > 0.5)
    hc_priority = np.sum(all_gates <= 0.5)
    
    total_elements = all_gates.size
    print(f"  PTM Priority (g > 0.5): {ptm_priority} / {total_elements} ({100*ptm_priority/total_elements:.1f}%)")
    print(f"  HC Priority (g <= 0.5): {hc_priority} / {total_elements} ({100*hc_priority/total_elements:.1f}%)")
    
    return all_gates, all_labels


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train(args):
    """
    Main training function with experiment management.
    
    Args:
        args: argparse.Namespace object with training configuration
    """
    # Setup
    cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device(DEVICE)
    use_mixed_precision = bool(getattr(args, "mixed_precision", MIXED_PRECISION))

    # Auto-generate experiment name if not provided
    if args.exp_name is None:
        args.exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Conservative auto-bump for under-utilized runs on CUDA.
    if str(device).startswith("cuda") and os.getenv("SV_DISABLE_AUTO_BATCH_BUMP", "0") != "1":
        mode_int = int(getattr(args, "mode", 3))
        target_train_bs = {1: 8, 2: 12, 3: 8}.get(mode_int, int(getattr(args, "batch_size", 8)))
        target_val_bs = {1: 12, 2: 12, 3: 8}.get(mode_int, int(getattr(args, "val_batch_size", 8)))

        if int(getattr(args, "batch_size", 1)) < target_train_bs:
            print(f"[AUTO-TUNE] Increase batch_size: {args.batch_size} -> {target_train_bs}")
            args.batch_size = int(target_train_bs)

        if int(getattr(args, "val_batch_size", 1)) < target_val_bs:
            print(f"[AUTO-TUNE] Increase val_batch_size: {getattr(args, 'val_batch_size', 1)} -> {target_val_bs}")
            args.val_batch_size = int(target_val_bs)
    
    # Create experiment directory
    exp_dir = os.path.join(args.output_dir, "experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Khởi tạo TensorBoard Writer
    tb_log_dir = os.path.join(exp_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs will be saved to: {tb_log_dir}")

    # Print configuration pretty
    print("\n" + "="*80)
    print("SPEAKER VERIFICATION TRAINING - CONFIGURATION")
    print("="*80)
    print(f"{'Experiment Name':<30} {args.exp_name}")
    print(f"{'Device':<30} {device}")
    print(f"{'Mode':<30} {args.mode} (1=PTM, 2=Handcrafted, 3=Fusion)")
    print(f"{'Fusion Method':<30} {args.fusion_method if args.mode == 3 else 'N/A'}")
    print(f"{'Feature Mode':<30} {args.feature_mode if args.mode in [2, 3] else 'N/A'}")
    mode_int = int(getattr(args, "mode", 3))
    if mode_int in [1, 3]:
        if bool(getattr(args, "use_ptm_on_the_fly", False)):
            ptm_input_source = "on-the-fly"
        elif os.path.isdir(getattr(args, "embedding_path", "")):
            ptm_input_source = "precomputed_shards_dir"
        elif str(getattr(args, "embedding_path", "")).lower().endswith(".pt"):
            ptm_input_source = "precomputed_single_pt"
        else:
            ptm_input_source = "unknown"
    else:
        ptm_input_source = "N/A"
    print(f"{'PTM Input Source':<30} {ptm_input_source}")
    print(f"{'Use Gating':<30} {args.use_gating if args.mode == 3 else 'N/A'}")
    print(f"{'Use Augment':<30} {args.use_augment if args.mode in [2, 3] else 'N/A'}")
    print(f"{'PTM On-the-fly':<30} {bool(getattr(args, 'use_ptm_on_the_fly', False)) if args.mode in [1, 3] else 'N/A'}")
    if args.mode in [1, 3] and bool(getattr(args, 'use_ptm_on_the_fly', False)):
        print(f"{'PTM Runtime Model':<30} {getattr(args, 'ptm_model_id', 'facebook/wav2vec2-base')}")
        print(f"{'Audio Base Dir':<30} {getattr(args, 'audio_base_dir', '') or os.getenv('SV_AUDIO_BASE_DIR', '')}")
        print(f"{'Audio Sample Rate':<30} {int(getattr(args, 'audio_sample_rate', 16000))}")
    print(f"\n{'Learning Rate':<30} {args.learning_rate}")
    print(f"{'Optimizer':<30} {args.optimizer.upper()}")
    print(f"{'Batch Size':<30} {args.batch_size}")
    print(f"{'Epochs':<30} {args.num_epochs}")
    print(f"{'Weight Decay':<30} {args.weight_decay}")
    print(f"{'Mixed Precision':<30} {use_mixed_precision}")
    print(f"\n{'Early Stop Patience':<30} {args.early_stop_patience}")
    print(f"{'LR Scheduler':<30} {args.lr_scheduler}")
    print(f"{'AAM Margin':<30} {AAM_MARGIN}")
    print(f"{'AAM Scale':<30} {AAM_SCALE}")
    print(f"\n{'Experiment Dir':<30} {exp_dir}")
    print("="*80 + "\n")

    use_speaker_balanced_sampler = getattr(args, "use_speaker_balanced_sampler", USE_SPK_BALANCED_SAMPLER)
    speakers_per_batch = int(getattr(args, "speakers_per_batch", SPK_PER_BATCH))
    utt_per_speaker = int(getattr(args, "utt_per_speaker", UTT_PER_SPK))

    print(f"{'Spk-Balanced Sampler':<30} {use_speaker_balanced_sampler}")
    if use_speaker_balanced_sampler:
        print(f"{'Speakers per Batch':<30} {speakers_per_batch}")
        print(f"{'Utterances per Speaker':<30} {utt_per_speaker}")
    print(f"{'Hard Negative Mining':<30} {HARD_NEGATIVE_MINING}")
    print(f"{'Hard Neg Weight/TopK':<30} {HARD_NEGATIVE_WEIGHT} / {HARD_NEGATIVE_TOPK}")
    print(f"{'Augment Prob':<30} {AUGMENT_PROB}")

    config_snapshot = {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "duration": getattr(args, 'duration', 'unknown'),
        "pretrained_model": getattr(args, 'pretrained_model', 'N/A'),
        "mode": args.mode,
        "fusion_method": getattr(args, 'fusion_method', 'N/A'),
        "feature_mode": args.feature_mode,
        "use_gating": getattr(args, 'use_gating', False),
        "use_augment": getattr(args, 'use_augment', False),
        "use_speaker_balanced_sampler": use_speaker_balanced_sampler,
        "speakers_per_batch": speakers_per_batch,
        "utt_per_speaker": utt_per_speaker,
        "hard_negative_mining": HARD_NEGATIVE_MINING,
        "hard_negative_topk": HARD_NEGATIVE_TOPK,
        "hard_negative_weight": HARD_NEGATIVE_WEIGHT,
        "hard_negative_margin": HARD_NEGATIVE_MARGIN,
        "augment_prob": AUGMENT_PROB,
        "feature_noise_std": FEATURE_NOISE_STD,
        "embedding_noise_std": EMBEDDING_NOISE_STD,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "weight_decay": args.weight_decay,
        "mixed_precision": use_mixed_precision,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "validate_every": validate_every,
        "max_train_steps_per_epoch": max_train_steps_per_epoch,
        "train_subset_fraction": train_subset_fraction,
        "max_train_samples": max_train_samples,
        "val_max_pos_pairs": val_max_pos_pairs,
        "early_stop_patience": args.early_stop_patience,
        "lr_scheduler": args.lr_scheduler,
        "aam_margin": AAM_MARGIN,
        "aam_scale": AAM_SCALE,
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print("Loading Train/Val data...")
    train_loader, val_loader, train_speaker_to_idx, num_speakers = create_train_val_loaders(
        args.embedding_path,
        args.feature_path,
        args.mode,
        args.batch_size,
        num_workers=num_workers,
        use_speaker_balanced=use_speaker_balanced_sampler,
        speakers_per_batch=speakers_per_batch,
        utt_per_speaker=utt_per_speaker,
        max_frames=int(getattr(args, "max_frames", 350)),
        val_batch_size=val_batch_size,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        use_ptm_on_the_fly=bool(getattr(args, 'use_ptm_on_the_fly', False)),
        audio_base_dir=getattr(args, 'audio_base_dir', ''),
        audio_sample_rate=int(getattr(args, 'audio_sample_rate', 16000)),
        max_audio_seconds=float(getattr(args, 'max_audio_seconds', 8.0)),
        train_subset_fraction=float(getattr(args, 'train_subset_fraction', 1.0)),
        max_train_samples=int(getattr(args, 'max_train_samples', 0)),
    )
    print(f"✓ Loaded {num_speakers} speakers")
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}\n")

    train_dataset_size = int(len(train_loader.dataset))
    val_dataset_size = int(len(val_loader.dataset))

    # Create model
    print("Creating model...")
    model = get_model(
        num_speakers,
        device=str(device),
        mode=args.mode,
        fusion_method=args.fusion_method,
        feature_mode=args.feature_mode,
        use_gating=args.use_gating
    )

    # Stage A -> Stage B warm-start (best-effort)
    init_report = try_initialize_mode3_from_stageA(model, args, device, exp_dir)

    # Save model summary
    print("\nGenerating model summary...")
    skip_model_summary = bool(getattr(args, "skip_model_summary", int(args.mode) == 3))
    # Default behavior: always generate and save model summary unless explicitly overridden.
    force_model_summary = os.getenv("SV_FORCE_MODEL_SUMMARY", "1") == "1"
    if force_model_summary:
        skip_model_summary = False
    if skip_model_summary:
        print("ℹ Skip model summary để giảm peak memory.")
    else:
        try:
            actual_input_dim = DIM_MAP.get(args.feature_mode, 81)
            # Tạo dummy tensors theo mode thay vì dict shape
            use_ptm_on_the_fly = bool(getattr(args, "use_ptm_on_the_fly", False))
            dummy_audio_len = max(1, int(float(getattr(args, "max_audio_seconds", 8.0)) * int(getattr(args, "audio_sample_rate", 16000))))
            if args.mode == 3:
                if use_ptm_on_the_fly:
                    dummy_inputs = {
                        "audio": torch.randn(args.batch_size, dummy_audio_len).to(device),
                        "audio_lengths": torch.full((args.batch_size,), dummy_audio_len, dtype=torch.long).to(device),
                        "feature": torch.randn(args.batch_size, actual_input_dim, 200).to(device),
                    }
                else:
                    dummy_inputs = {
                        "embedding": torch.randn(args.batch_size, PTM_NUM_LAYERS, PTM_DIM).to(device),
                        "feature": torch.randn(args.batch_size, actual_input_dim, 200).to(device)
                    }
            elif args.mode == 1:
                if use_ptm_on_the_fly:
                    dummy_inputs = {
                        "audio": torch.randn(args.batch_size, dummy_audio_len).to(device),
                        "audio_lengths": torch.full((args.batch_size,), dummy_audio_len, dtype=torch.long).to(device),
                    }
                else:
                    dummy_inputs = {"embedding": torch.randn(args.batch_size, PTM_NUM_LAYERS, PTM_DIM).to(device)}
            else:
                dummy_inputs = {"feature": torch.randn(args.batch_size, actual_input_dim, 200).to(device)}

            # Truyền thẳng dummy_inputs dưới dạng **kwargs vào torchinfo
            model_summary = summary(model, **dummy_inputs, verbose=0)

            with open(os.path.join(exp_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
                f.write(str(model_summary))
            print(f"✓ Model summary saved to {os.path.join(exp_dir, 'model_summary.txt')}")
        except Exception as e:
            print(f"⚠ Could not save model summary: {e}")

    # Loss and optimizer
    criterion = AAMSoftmaxLoss(num_speakers=num_speakers, embedding_dim=args.embedding_dim)
    criterion = criterion.to(device)

    if args.optimizer.lower() == "adam":
        params = list(model.parameters()) + list(criterion.parameters())
        opt = optim.AdamW(
            params, lr=args.learning_rate, weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == "sgd":
        params = list(model.parameters()) + list(criterion.parameters())
        opt = optim.SGD(
            params,
            lr=args.learning_rate,
            momentum=MOMENTUM,
            nesterov=NESTEROV,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Learning rate scheduler
    if args.lr_scheduler.lower() == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=COSINE_T_MAX, eta_min=MIN_LEARNING_RATE)
    elif args.lr_scheduler.lower() == "plateau":
        scheduler = ReduceLROnPlateau(
            opt,
            mode="min",
            factor=PLATEAU_FACTOR,
            patience=PLATEAU_PATIENCE,
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_scheduler}")

    # Mixed precision
    scaler = GradScaler('cuda') if (use_mixed_precision and str(device).startswith("cuda")) else None

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stop_patience, delta=EARLY_STOP_DELTA)

    # Logging
    log_file = os.path.join(exp_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Training started: {datetime.now()}\n")
        f.write(json.dumps(config_snapshot, indent=2) + "\n\n")

    # Training loop
    best_val_eer = float("inf") 
    
    training_history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
        "val_eer": [], "val_mindcf": [], # Thêm 2 field mới
        "train_epoch_time_sec": [],
        "val_epoch_time_sec": [],
        "epoch_total_time_sec": [],
    }

    # ==========================================================
    # TẠO/NẠP DANH SÁCH CẶP VALIDATION CỐ ĐỊNH (FIXED TRIAL LIST)
    # ==========================================================
    print("Đang chuẩn bị fixed validation trials...")
    val_trials = get_or_create_fixed_val_trials(
        val_loader=val_loader,
        args=args,
        max_pos_pairs=val_max_pos_pairs,
        random_seed=args.seed,
    )
    # ==========================================================

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    run_start_time = time.perf_counter()

    print("Starting training...\n")
    for epoch in range(args.num_epochs):
        #current_margin = update_aam_margin(criterion, epoch, final_margin=AAM_MARGIN, warmup_epochs=5)
        #print(f"\n[Info] Epoch {epoch + 1}: AAM-Softmax Margin set to {current_margin:.4f}")
        
        # Train
        train_start_time = time.perf_counter()
        train_loss, train_acc = train_epoch(
            model, train_loader, opt, criterion, scaler, epoch, device,
            use_augment=getattr(args, 'use_augment', False),
            augment_prob=AUGMENT_PROB,
            feature_noise_std=FEATURE_NOISE_STD,
            embedding_noise_std=EMBEDDING_NOISE_STD,
            hard_negative_mining=HARD_NEGATIVE_MINING,
            hard_negative_topk=HARD_NEGATIVE_TOPK,
            hard_negative_weight=HARD_NEGATIVE_WEIGHT,
            hard_negative_margin=HARD_NEGATIVE_MARGIN,
            use_mixed_precision=use_mixed_precision,
            non_blocking_transfer=non_blocking_transfer,
            max_steps_per_epoch=max_train_steps_per_epoch,
        )
        train_epoch_time = float(time.perf_counter() - train_start_time)

        # Validation can be reduced to every N epochs for faster long runs.
        should_validate = ((epoch + 1) % validate_every == 0) or (epoch == args.num_epochs - 1)
        if should_validate:
            val_start_time = time.perf_counter()
            val_eer, val_mindcf = validate(model, val_loader, device, val_trials)
            val_epoch_time = float(time.perf_counter() - val_start_time)
        else:
            val_eer = float(training_history["val_eer"][-1]) if training_history["val_eer"] else float("inf")
            val_mindcf = float(training_history["val_mindcf"][-1]) if training_history["val_mindcf"] else float("inf")
            val_epoch_time = 0.0
            print(f"[VAL] Skip epoch {epoch + 1}: validate_every={validate_every}")
        epoch_total_time = float(time.perf_counter() - epoch_start_time)

        # Update scheduler
        if args.lr_scheduler.lower() == "cosine":
            scheduler.step()
        elif args.lr_scheduler.lower() == "plateau":
            # Scheduler Plateau cũng nên ưu tiên dựa trên EER thay vì Loss
            if should_validate:
                scheduler.step(val_eer)

        # Update history
        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(train_acc)
        training_history["val_eer"].append(float(val_eer))
        training_history["val_mindcf"].append(float(val_mindcf))
        training_history["train_epoch_time_sec"].append(train_epoch_time)
        training_history["val_epoch_time_sec"].append(val_epoch_time)
        training_history["epoch_total_time_sec"].append(epoch_total_time)

        # Ghi log vào TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Metrics/EER_Validation", val_eer, epoch)       # ĐỒ THỊ MỚI
        writer.add_scalar("Metrics/MinDCF_Validation", val_mindcf, epoch) # ĐỒ THỊ MỚI
        writer.add_scalar("LearningRate", opt.param_groups[0]['lr'], epoch)

        # Logging
        log_msg = (
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"EER: {val_eer:.2f}%, MinDCF: {val_mindcf:.4f} | "
            f"LR: {opt.param_groups[0]['lr']:.6f} | "
            f"TrainTime: {train_epoch_time:.2f}s, ValTime: {val_epoch_time:.2f}s, EpochTime: {epoch_total_time:.2f}s"
        )
        print(log_msg)
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

        # ĐIỂM CỐT LÕI: Save checkpoint dựa trên EER thay vì Loss
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            checkpoint_path = os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)
            # Tái sử dụng hàm save (vẫn lưu best_val_eer vào field best_loss cho tương thích)
            save_checkpoint(model, opt, epoch, best_val_eer, checkpoint_path)
            shutil.copy(checkpoint_path, os.path.join(exp_dir, BEST_MODEL_NAME))

        # Early stopping (bây giờ sẽ ngừng train nếu EER không giảm)
        if should_validate:
            early_stopping(val_eer)
            if early_stopping.early_stop:
                print("\n✓ Early stopping triggered!")
                with open(log_file, "a") as f:
                    f.write("Early stopping triggered!\n")
                break

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, FINAL_MODEL_NAME)
    save_checkpoint(model, opt, epoch, best_val_eer, final_path)
    shutil.copy(final_path, os.path.join(exp_dir, FINAL_MODEL_NAME))

    # Save history
    history_path = os.path.join(exp_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=4)

    # Load best model
    model, _, _, _ = load_checkpoint(os.path.join(exp_dir, BEST_MODEL_NAME), model)

    # Phân tích Gating trên tập VAL
    if args.mode == 3 and args.fusion_method == "gating":
        gates, labels = analyze_gating_behavior(model, val_loader, device, exp_dir)
    else:
        gates, labels = None, None

    # Save final results
    total_train_time_sec = float(time.perf_counter() - run_start_time)
    epochs_trained = int(epoch + 1)
    avg_train_epoch_time_sec = float(np.mean(training_history["train_epoch_time_sec"])) if training_history["train_epoch_time_sec"] else 0.0
    avg_val_epoch_time_sec = float(np.mean(training_history["val_epoch_time_sec"])) if training_history["val_epoch_time_sec"] else 0.0
    peak_gpu_memory_allocated_mb = 0.0
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        peak_gpu_memory_allocated_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    total_seen_samples = int((train_dataset_size + val_dataset_size) * epochs_trained)
    gflops_total = float(gflops_per_sample * total_seen_samples)

    final_results = {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "config": config_snapshot,
        "best_val_eer": float(best_val_eer), 
        "best_val_mindcf": float(training_history["val_mindcf"][training_history["val_eer"].index(best_val_eer)]),
        "final_train_loss": float(training_history["train_loss"][-1]),
        "final_train_accuracy": float(training_history["train_accuracy"][-1]),
        "epochs_trained": epochs_trained,
        "model_stats": model_stats,
        "performance": {
            "total_train_time_sec": total_train_time_sec,
            "avg_train_epoch_time_sec": avg_train_epoch_time_sec,
            "avg_val_epoch_time_sec": avg_val_epoch_time_sec,
            "train_epoch_time_sec": training_history["train_epoch_time_sec"],
            "val_epoch_time_sec": training_history["val_epoch_time_sec"],
            "epoch_total_time_sec": training_history["epoch_total_time_sec"],
            "gflops_per_sample": float(gflops_per_sample),
            "gflops_total": gflops_total,
            "total_samples_accounted": total_seen_samples,
            "peak_gpu_memory_allocated_mb": peak_gpu_memory_allocated_mb,
        },
    }

    if gates is not None:
        final_results["gating_analysis"] = {
            "ptm_priority_count": int(np.sum(gates > 0.5)),
            "hc_priority_count": int(np.sum(gates <= 0.5)),
            "mean_gate_weight": float(np.mean(gates)),
        }

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    results_csv_path = _write_results_csv(exp_dir, final_results)
    experiment_root = os.path.join(args.output_dir, experiments_dirname)
    summary_json_path, summary_csv_path = _update_experiment_mode_summary(experiment_root, final_results, exp_dir)

    print(f"\n✓ Training completed!")
    print(f"  Best validation EER: {best_val_eer:.4f}")
    print(f"  Experiment dir: {exp_dir}")
    print(f"  Config: {os.path.join(exp_dir, 'config.json')}")
    print(f"  Results: {os.path.join(exp_dir, 'results.json')}")
    print(f"  Results CSV: {results_csv_path}")
    print(f"  Summary JSON (all modes): {summary_json_path}")
    print(f"  Summary CSV (all modes): {summary_csv_path}")
    print(f"  Model: {os.path.join(exp_dir, BEST_MODEL_NAME)}")

    writer.close()
    
    return model, training_history, exp_dir