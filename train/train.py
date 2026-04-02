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


def _model_has_prefix(model, prefix: str) -> bool:
    try:
        for key in model.state_dict().keys():
            if key.startswith(prefix):
                return True
    except Exception:
        return False
    return False


def _summarize_warmstart_load(model, mapped_sd, incompatible, relevant_prefixes):
    model_sd = model.state_dict()

    candidate_keys = [k for k in mapped_sd.keys() if k in model_sd]
    shape_matched_keys = [
        k for k in candidate_keys
        if hasattr(mapped_sd[k], "shape") and hasattr(model_sd[k], "shape") and tuple(mapped_sd[k].shape) == tuple(model_sd[k].shape)
    ]

    raw_missing = list(getattr(incompatible, "missing_keys", []))
    raw_unexpected = list(getattr(incompatible, "unexpected_keys", []))

    filtered_missing = [
        k for k in raw_missing
        if any(k.startswith(prefix) for prefix in relevant_prefixes)
    ]
    filtered_unexpected = [
        k for k in raw_unexpected
        if any(k.startswith(prefix) for prefix in relevant_prefixes)
    ]

    return {
        "candidate_keys": int(len(candidate_keys)),
        "shape_matched_keys": int(len(shape_matched_keys)),
        "coverage_ratio": float(len(shape_matched_keys) / max(1, len(candidate_keys))),
        "missing_keys": filtered_missing,
        "unexpected_keys": filtered_unexpected,
        "raw_missing_key_count": int(len(raw_missing)),
        "raw_unexpected_key_count": int(len(raw_unexpected)),
    }


def try_initialize_mode3_from_stageA(model, args, device, exp_dir):
    """Optional warm-start for Mode3 from already-trained Mode1 + Mode2 checkpoints.

    This is best-effort: if checkpoints aren't found, training proceeds from scratch.
    """
    if getattr(args, "mode", None) != 3:
        return None

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

    if not mode1_ckpt:
        exp_name = f"Mode1_PTM_{duration}_{pretrained_model}"
        mode1_ckpt = os.path.join(output_dir, "experiments", exp_name, BEST_MODEL_NAME)
    if not mode2_ckpt:
        exp_name = f"Mode2_HC_{duration}_{feature_mode}"
        mode2_ckpt = os.path.join(output_dir, "experiments", exp_name, BEST_MODEL_NAME)

    init_report["mode1_ckpt"] = mode1_ckpt
    init_report["mode2_ckpt"] = mode2_ckpt

    # Load Mode1 -> (ptm_encoder, ptm_backbone)
    if os.path.exists(mode1_ckpt):
        checkpoint = torch.load(mode1_ckpt, map_location=device)
        stageA_sd = checkpoint.get("model_state_dict", checkpoint)

        stageA_sd = {k: v for k, v in stageA_sd.items() if k.startswith("ptm_encoder.") or k.startswith("backbone.")}
        if _model_has_prefix(model, "ptm_backbone."):
            mapped_sd = _map_stageA_to_mode3_state_dict(stageA_sd, backbone_target_prefix="ptm_backbone")
        else:
            # Mode3 PTM branch may not have a backbone (e.g., PTM head only)
            mapped_sd = stageA_sd

        incompatible = model.load_state_dict(mapped_sd, strict=False)
        mode1_summary = _summarize_warmstart_load(
            model=model,
            mapped_sd=mapped_sd,
            incompatible=incompatible,
            relevant_prefixes=("ptm_encoder.", "ptm_backbone.", "ptm_emb_ln."),
        )
        init_report["loaded"]["mode1"] = True
        init_report["mode1_incompatible_keys"] = {
            "missing_keys": mode1_summary["missing_keys"],
            "unexpected_keys": mode1_summary["unexpected_keys"],
        }
        init_report["mode1_load_summary"] = {
            "candidate_keys": mode1_summary["candidate_keys"],
            "shape_matched_keys": mode1_summary["shape_matched_keys"],
            "coverage_ratio": mode1_summary["coverage_ratio"],
            "raw_missing_key_count": mode1_summary["raw_missing_key_count"],
            "raw_unexpected_key_count": mode1_summary["raw_unexpected_key_count"],
        }
        print(f"✓ Initialized Mode3 PTM branch from: {mode1_ckpt}")
        print(
            f"  ↳ Mode1 warm-start coverage: "
            f"{mode1_summary['shape_matched_keys']}/{mode1_summary['candidate_keys']} "
            f"({mode1_summary['coverage_ratio'] * 100:.1f}%)"
        )
    else:
        print(f"ℹ Mode1 checkpoint not found (skip init): {mode1_ckpt}")

    # Load Mode2 -> (handcrafted_encoder, hc_backbone)
    if os.path.exists(mode2_ckpt):
        checkpoint = torch.load(mode2_ckpt, map_location=device)
        stageA_sd = checkpoint.get("model_state_dict", checkpoint)

        stageA_sd = {k: v for k, v in stageA_sd.items() if k.startswith("handcrafted_encoder.") or k.startswith("backbone.")}
        if _model_has_prefix(model, "hc_backbone."):
            mapped_sd = _map_stageA_to_mode3_state_dict(stageA_sd, backbone_target_prefix="hc_backbone")
        else:
            mapped_sd = stageA_sd

        incompatible = model.load_state_dict(mapped_sd, strict=False)
        mode2_summary = _summarize_warmstart_load(
            model=model,
            mapped_sd=mapped_sd,
            incompatible=incompatible,
            relevant_prefixes=("handcrafted_encoder.", "hc_backbone.", "hc_emb_ln."),
        )
        init_report["loaded"]["mode2"] = True
        init_report["mode2_incompatible_keys"] = {
            "missing_keys": mode2_summary["missing_keys"],
            "unexpected_keys": mode2_summary["unexpected_keys"],
        }
        init_report["mode2_load_summary"] = {
            "candidate_keys": mode2_summary["candidate_keys"],
            "shape_matched_keys": mode2_summary["shape_matched_keys"],
            "coverage_ratio": mode2_summary["coverage_ratio"],
            "raw_missing_key_count": mode2_summary["raw_missing_key_count"],
            "raw_unexpected_key_count": mode2_summary["raw_unexpected_key_count"],
        }
        print(f"✓ Initialized Mode3 HC branch from: {mode2_ckpt}")
        print(
            f"  ↳ Mode2 warm-start coverage: "
            f"{mode2_summary['shape_matched_keys']}/{mode2_summary['candidate_keys']} "
            f"({mode2_summary['coverage_ratio'] * 100:.1f}%)"
        )
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

    for _, indices in val_speaker_indices.items():
        n = len(indices)
        if n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    pos_pairs.append((indices[i], indices[j]))

    rng.shuffle(pos_pairs)
    pos_pairs = pos_pairs[:max_pos_pairs]

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

    def reset(self, best_value=None, keep_best=False):
        self.counter = 0
        if best_value is not None:
            self.best_loss = float(best_value)
        elif not keep_best:
            self.best_loss = None
        self.early_stop = False


def _set_module_trainable(module, trainable: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = bool(trainable)


def _build_optimizer_and_scheduler(args, model, criterion, stage="single"):
    if stage == "stage2" and getattr(args, "mode", None) == 3:
        stage2_ptm_lr_scale = float(getattr(args, "stage2_ptm_lr_scale", 0.35))
        stage2_ptm_lr_scale = max(0.0, stage2_ptm_lr_scale)

        ptm_params = []
        if hasattr(model, "ptm_encoder"):
            ptm_params.extend([p for p in model.ptm_encoder.parameters() if p.requires_grad])
        if hasattr(model, "ptm_emb_ln"):
            ptm_params.extend([p for p in model.ptm_emb_ln.parameters() if p.requires_grad])

        ptm_ids = {id(p) for p in ptm_params}
        other_model_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in ptm_ids
        ]
        criterion_params = [p for p in criterion.parameters() if p.requires_grad]

        param_groups = []
        if len(other_model_params) > 0:
            param_groups.append({"params": other_model_params, "lr": args.learning_rate})
        if len(ptm_params) > 0:
            param_groups.append({"params": ptm_params, "lr": args.learning_rate * stage2_ptm_lr_scale})
        if len(criterion_params) > 0:
            param_groups.append({"params": criterion_params, "lr": args.learning_rate})
    else:
        params = [p for p in model.parameters() if p.requires_grad] + [
            p for p in criterion.parameters() if p.requires_grad
        ]
        param_groups = params

    if args.optimizer.lower() == "adam":
        optimizer = optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=args.learning_rate,
            momentum=MOMENTUM,
            nesterov=NESTEROV,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    if args.lr_scheduler.lower() == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=COSINE_T_MAX, eta_min=MIN_LEARNING_RATE)
    elif args.lr_scheduler.lower() == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=PLATEAU_FACTOR,
            patience=PLATEAU_PATIENCE,
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_scheduler}")

    return optimizer, scheduler


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

    for batch_idx, batch_data in enumerate(progress_bar):
        # Move data to device
        labels = batch_data["label"].to(device)
        inputs = {
            k: v.to(device)
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
        
        optimizer.zero_grad()

        # Forward pass with mixed precision
        if MIXED_PRECISION:
            with autocast('cuda'):
                _, embeddings = model(**inputs)
            loss, logits = criterion(None, labels, embeddings=embeddings.float())
        else:
            _, embeddings = model(**inputs)
            loss, logits = criterion(None, labels, embeddings=embeddings)

        if hard_negative_mining:
            hnm_loss = compute_hard_negative_loss(
                embeddings=embeddings.float(),
                labels=labels,
                topk=hard_negative_topk,
                margin=hard_negative_margin,
            )
            loss = loss + hard_negative_weight * hnm_loss

        # Backward pass
        if MIXED_PRECISION:
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

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def validate(model, val_loader, device, val_trials):
    """
    Validate Open-set với Trial List cố định
    """
    model.eval()
    all_embeddings_list = []
    
    # 1. Trích xuất embedding theo đúng thứ tự của val_loader
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation (EER)", leave=False)
        for batch_data in progress_bar:
            inputs = {
                k: v.to(device)
                for k, v in batch_data.items()
                if isinstance(v, torch.Tensor) and k != "label"
            }
            _, embeddings = model(**inputs) 
            all_embeddings_list.append(embeddings.cpu())
    
    all_embeddings = torch.cat(all_embeddings_list, dim=0)
    all_embeddings = F.normalize(all_embeddings, p=2, dim=1).numpy()

    # 2. Tính Cosine Similarity dựa trên danh sách cặp cố định
    scores = []
    y_true = []

    for idx1, idx2 in val_trials["pos_pairs"]:
        scores.append(np.dot(all_embeddings[idx1], all_embeddings[idx2]))
        y_true.append(1)

    for idx1, idx2 in val_trials["neg_pairs"]:
        scores.append(np.dot(all_embeddings[idx1], all_embeddings[idx2]))
        y_true.append(0)

    scores = np.array(scores)
    y_true = np.array(y_true)

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
            
            # Average gate weights across embedding dimension
            gate_avg = gate_weights.mean(dim=-1).cpu().numpy()
            all_gates.extend(gate_avg)
            all_labels.extend(labels.cpu().numpy())
    
    all_gates = np.array(all_gates)
    all_labels = np.array(all_labels)
    
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device(DEVICE)

    # Auto-generate experiment name if not provided
    if args.exp_name is None:
        args.exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create experiment directory
    experiments_dirname = getattr(args, "experiments_dirname", "experiments")
    exp_dir = os.path.join(args.output_dir, experiments_dirname, args.exp_name)
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
    print(f"{'Use Gating':<30} {args.use_gating if args.mode == 3 else 'N/A'}")
    print(f"{'Use Augment':<30} {args.use_augment if args.mode in [2, 3] else 'N/A'}")
    print(f"\n{'Learning Rate':<30} {args.learning_rate}")
    print(f"{'Optimizer':<30} {args.optimizer.upper()}")
    print(f"{'Batch Size':<30} {args.batch_size}")
    print(f"{'Epochs':<30} {args.num_epochs}")
    print(f"{'Weight Decay':<30} {args.weight_decay}")
    print(f"{'Mixed Precision':<30} {args.mixed_precision}")
    print(f"\n{'Early Stop Patience':<30} {args.early_stop_patience}")
    print(f"{'LR Scheduler':<30} {args.lr_scheduler}")
    print(f"{'AAM Margin':<30} {args.aam_margin}")
    print(f"{'AAM Scale':<30} {args.aam_scale}")
    print(f"{'Branch Dropout (Mode3)':<30} {getattr(args, 'branch_dropout_prob', 0.0) if args.mode == 3 else 'N/A'}")
    print(f"{'Use 2-Stage FT (Mode3)':<30} {getattr(args, 'use_two_stage_ft', False) if args.mode == 3 else 'N/A'}")
    print(f"{'Stage1 Epochs':<30} {getattr(args, 'stage1_epochs', 0) if args.mode == 3 else 'N/A'}")
    print(f"{'Stage2 PTM LR Scale':<30} {getattr(args, 'stage2_ptm_lr_scale', 0.35) if args.mode == 3 else 'N/A'}")
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
    augment_prob = float(getattr(args, 'augment_prob', AUGMENT_PROB))
    feature_noise_std = float(getattr(args, 'feature_noise_std', FEATURE_NOISE_STD))
    embedding_noise_std = float(getattr(args, 'embedding_noise_std', EMBEDDING_NOISE_STD))

    print(f"{'Augment Prob':<30} {augment_prob}")
    print(f"{'Feature Noise Std':<30} {feature_noise_std}")
    print(f"{'Embedding Noise Std':<30} {embedding_noise_std}")

    config_snapshot = {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "duration": getattr(args, 'duration', 'unknown'),
        "pretrained_model": getattr(args, 'pretrained_model', 'N/A'),
        "experiments_dirname": experiments_dirname,
        "mode": args.mode,
        "fusion_method": getattr(args, 'fusion_method', 'N/A'),
        "feature_mode": args.feature_mode,
        "use_gating": getattr(args, 'use_gating', False),
        "branch_dropout_prob": float(getattr(args, 'branch_dropout_prob', 0.0)),
        "use_two_stage_ft": bool(getattr(args, 'use_two_stage_ft', False)),
        "stage1_epochs": int(getattr(args, 'stage1_epochs', 0)),
        "stage2_ptm_lr_scale": float(getattr(args, 'stage2_ptm_lr_scale', 0.35)),
        "use_augment": getattr(args, 'use_augment', False),
        "use_speaker_balanced_sampler": use_speaker_balanced_sampler,
        "speakers_per_batch": speakers_per_batch,
        "utt_per_speaker": utt_per_speaker,
        "hard_negative_mining": HARD_NEGATIVE_MINING,
        "hard_negative_topk": HARD_NEGATIVE_TOPK,
        "hard_negative_weight": HARD_NEGATIVE_WEIGHT,
        "hard_negative_margin": HARD_NEGATIVE_MARGIN,
        "augment_prob": augment_prob,
        "feature_noise_std": feature_noise_std,
        "embedding_noise_std": embedding_noise_std,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "weight_decay": args.weight_decay,
        "mixed_precision": args.mixed_precision,
        "early_stop_patience": args.early_stop_patience,
        "lr_scheduler": args.lr_scheduler,
        "aam_margin": float(args.aam_margin),
        "aam_scale": float(args.aam_scale),
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print("Loading Train/Val data...")
    train_loader, val_loader, train_speaker_to_idx, num_speakers = create_train_val_loaders(
        args.embedding_path,
        args.feature_path,
        args.mode,
        args.batch_size,
        num_workers=0,
        use_speaker_balanced=use_speaker_balanced_sampler,
        speakers_per_batch=speakers_per_batch,
        utt_per_speaker=utt_per_speaker,
    )
    print(f"✓ Loaded {num_speakers} speakers")
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}\n")

    # Create model
    print("Creating model...")
    model = get_model(
        num_speakers,
        device=str(device),
        mode=args.mode,
        fusion_method=args.fusion_method,
        feature_mode=args.feature_mode,
        use_gating=args.use_gating,
        branch_dropout_prob=float(getattr(args, 'branch_dropout_prob', 0.0)),
    )

    # Stage A -> Stage B warm-start (best-effort)
    init_report = try_initialize_mode3_from_stageA(model, args, device, exp_dir)

    # Save model summary
    print("\nGenerating model summary...")
    try:
        actual_input_dim = DIM_MAP.get(args.feature_mode, 81)
        # Tạo dummy tensors theo mode thay vì dict shape
        if args.mode == 3:
            dummy_inputs = { 
                "embedding": torch.randn(args.batch_size, PTM_NUM_LAYERS, PTM_DIM).to(device), 
                "feature": torch.randn(args.batch_size, actual_input_dim, 200).to(device) 
            }
        elif args.mode == 1:
            dummy_inputs = { "embedding": torch.randn(args.batch_size, PTM_NUM_LAYERS, PTM_DIM).to(device) }
        else:
            dummy_inputs = { "feature": torch.randn(args.batch_size, actual_input_dim, 200).to(device) }

        # Truyền thẳng dummy_inputs dưới dạng **kwargs vào torchinfo
        model_summary = summary(model, **dummy_inputs, verbose=0)
        
        with open(os.path.join(exp_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
            f.write(str(model_summary))
        print(f"✓ Model summary saved to {os.path.join(exp_dir, 'model_summary.txt')}")
    except Exception as e:
        print(f"⚠ Could not save model summary: {e}")

    # Loss and optimizer
    criterion = AAMSoftmaxLoss(
        num_speakers=num_speakers,
        embedding_dim=args.embedding_dim,
        margin=float(args.aam_margin),
        scale=float(args.aam_scale),
    )
    criterion = criterion.to(device)

    use_two_stage_ft = bool(getattr(args, "use_two_stage_ft", False)) and int(args.mode) == 3
    stage1_epochs = max(0, int(getattr(args, "stage1_epochs", 0)))

    if use_two_stage_ft and stage1_epochs > 0:
        print(f"✓ 2-stage FT enabled | Stage1 freeze PTM for {stage1_epochs} epochs")
        _set_module_trainable(getattr(model, "ptm_encoder", None), False)
        _set_module_trainable(getattr(model, "ptm_emb_ln", None), False)

    # Loss and optimizer
    opt, scheduler = _build_optimizer_and_scheduler(args, model, criterion, stage="stage1" if (use_two_stage_ft and stage1_epochs > 0) else "single")

    # Mixed precision
    scaler = GradScaler('cuda') if args.mixed_precision else None

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stop_patience, delta=EARLY_STOP_DELTA)

    # Logging
    log_file = os.path.join(exp_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Training started: {datetime.now()}\n")
        f.write(json.dumps(config_snapshot, indent=2) + "\n\n")

    # Training loop
    best_val_eer_global = float("inf")
    
    training_history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
        "val_eer": [], "val_mindcf": [] # Thêm 2 field mới
    }

    # ==========================================================
    # TẠO/NẠP DANH SÁCH CẶP VALIDATION CỐ ĐỊNH (FIXED TRIAL LIST)
    # ==========================================================
    print("Đang chuẩn bị fixed validation trials...")
    val_trials = get_or_create_fixed_val_trials(
        val_loader=val_loader,
        args=args,
        max_pos_pairs=10000,
        random_seed=args.seed,
    )
    # ==========================================================

    print("Starting training...\n")
    for epoch in range(args.num_epochs):
        if use_two_stage_ft and stage1_epochs > 0 and epoch == stage1_epochs:
            print("\n[2-Stage FT] Switching to Stage 2: unfreeze PTM branch and rebuild optimizer/scheduler...")
            _set_module_trainable(getattr(model, "ptm_encoder", None), True)
            _set_module_trainable(getattr(model, "ptm_emb_ln", None), True)
            opt, scheduler = _build_optimizer_and_scheduler(args, model, criterion, stage="stage2")
            if args.mixed_precision:
                scaler = GradScaler('cuda')
            # Chỉ reset counter cho stage mới, nhưng giữ ngưỡng best toàn cục
            # để không "dễ dãi" hơn stage 1 và tránh ghi đè best checkpoint bởi EER tệ hơn.
            baseline_eer = None if best_val_eer_global == float("inf") else best_val_eer_global
            early_stopping.reset(best_value=baseline_eer)
            if baseline_eer is not None:
                print(f"[2-Stage FT] Keep global-best baseline for early-stop/checkpoint: EER={baseline_eer:.4f}%")

        #current_margin = update_aam_margin(criterion, epoch, final_margin=AAM_MARGIN, warmup_epochs=5)
        #print(f"\n[Info] Epoch {epoch + 1}: AAM-Softmax Margin set to {current_margin:.4f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, opt, criterion, scaler, epoch, device,
            use_augment=getattr(args, 'use_augment', False),
            augment_prob=augment_prob,
            feature_noise_std=feature_noise_std,
            embedding_noise_std=embedding_noise_std,
            hard_negative_mining=HARD_NEGATIVE_MINING,
            hard_negative_topk=HARD_NEGATIVE_TOPK,
            hard_negative_weight=HARD_NEGATIVE_WEIGHT,
            hard_negative_margin=HARD_NEGATIVE_MARGIN,
        )

        # Validate (Nhận thêm eer và mindcf)
        val_eer, val_mindcf = validate(model, val_loader, device, val_trials)

        # Update scheduler
        if args.lr_scheduler.lower() == "cosine":
            scheduler.step()
        elif args.lr_scheduler.lower() == "plateau":
            # Scheduler Plateau cũng nên ưu tiên dựa trên EER thay vì Loss
            scheduler.step(val_eer)

        # Update history
        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(train_acc)
        training_history["val_eer"].append(val_eer)
        training_history["val_mindcf"].append(val_mindcf)

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
            f"LR: {opt.param_groups[0]['lr']:.6f}"
        )
        print(log_msg)
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

        # ĐIỂM CỐT LÕI: Save checkpoint dựa trên EER thay vì Loss
        if val_eer < best_val_eer_global:
            best_val_eer_global = val_eer
            checkpoint_path = os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)
            # Tái sử dụng hàm save (vẫn lưu best_val_eer vào field best_loss cho tương thích)
            save_checkpoint(model, opt, epoch, best_val_eer_global, checkpoint_path)
            shutil.copy(checkpoint_path, os.path.join(exp_dir, BEST_MODEL_NAME))

        # Early stopping (bây giờ sẽ ngừng train nếu EER không giảm)
        early_stopping(val_eer)
        if early_stopping.early_stop:
            print("\n✓ Early stopping triggered!")
            with open(log_file, "a") as f:
                f.write("Early stopping triggered!\n")
            break

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, FINAL_MODEL_NAME)
    save_checkpoint(model, opt, epoch, best_val_eer_global, final_path)
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
    final_results = {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "config": config_snapshot,
        "best_val_eer": float(best_val_eer_global), 
        "best_val_mindcf": float(training_history["val_mindcf"][training_history["val_eer"].index(best_val_eer_global)]),
        "final_train_loss": float(training_history["train_loss"][-1]),
        "final_train_accuracy": float(training_history["train_accuracy"][-1]),
        "epochs_trained": epoch + 1,
    }

    if gates is not None:
        final_results["gating_analysis"] = {
            "ptm_priority_count": int(np.sum(gates > 0.5)),
            "hc_priority_count": int(np.sum(gates <= 0.5)),
            "mean_gate_weight": float(np.mean(gates)),
        }

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✓ Training completed!")
    print(f"  Best validation EER: {best_val_eer_global:.4f}")
    print(f"  Experiment dir: {exp_dir}")
    print(f"  Config: {os.path.join(exp_dir, 'config.json')}")
    print(f"  Results: {os.path.join(exp_dir, 'results.json')}")
    print(f"  Model: {os.path.join(exp_dir, BEST_MODEL_NAME)}")

    writer.close()
    
    return model, training_history, exp_dir