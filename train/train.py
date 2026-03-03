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
import shutil
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


def train_epoch(model, train_loader, optimizer, criterion, scaler, epoch, device, log_interval=LOG_INTERVAL, use_augment=True):
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
        inputs = {k: v.to(device) for k, v in batch_data.items() if k != "label"}

        if use_augment and "feature" in inputs:
            inputs["feature"] = apply_spec_augment(inputs["feature"])

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


def validate(model, val_loader, device):
    """
    Validate Open-set: Chỉ trích xuất embedding và đo EER/MinDCF.
    Không tính Loss và Accuracy vì tập Val chứa Unseen Speakers.
    """
    model.eval()
    all_embeddings_list = []
    all_labels_list = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation (EER)", leave=False)
        for batch_data in progress_bar:
            labels = batch_data["label"].to(device)
            inputs = {k: v.to(device) for k, v in batch_data.items() if k != "label"}

            # Chỉ cần lấy embedding, bỏ qua logits
            _, embeddings = model(**inputs) 
            
            all_embeddings_list.append(embeddings.cpu())
            all_labels_list.append(labels.cpu())
    
    # ---------------------------------------------------------
    # TÍNH TOÁN EER & MinDCF (TỐI ƯU HÓA LẤY MẪU CÂN BẰNG)
    # ---------------------------------------------------------
    all_embeddings = torch.cat(all_embeddings_list, dim=0)
    all_embeddings = F.normalize(all_embeddings, p=2, dim=1).cpu().numpy()
    all_labels = torch.cat(all_labels_list, dim=0).cpu().numpy()

    # 1. Phân nhóm index theo từng speaker
    import random
    from collections import defaultdict
    speaker_indices = defaultdict(list)
    for idx, label in enumerate(all_labels):
        speaker_indices[label].append(idx)

    pos_pairs = []
    # 2. Tạo Positive Pairs (Cùng speaker)
    for label, indices in speaker_indices.items():
        n = len(indices)
        if n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    pos_pairs.append((indices[i], indices[j]))

    random.shuffle(pos_pairs)
    pos_pairs = pos_pairs[:20000] # Giới hạn 20k cặp positive để chạy nhanh

    neg_pairs = []
    # 3. Tạo Negative Pairs (Khác speaker) có số lượng tương đương Positive
    labels_unique = list(speaker_indices.keys())
    while len(neg_pairs) < len(pos_pairs):
        spk1, spk2 = random.sample(labels_unique, 2)
        idx1 = random.choice(speaker_indices[spk1])
        idx2 = random.choice(speaker_indices[spk2])
        neg_pairs.append((idx1, idx2))

    # 4. Tính Cosine Similarity (Dot product của 2 vector đã normalize)
    scores = []
    y_true = []

    for idx1, idx2 in pos_pairs:
        scores.append(np.dot(all_embeddings[idx1], all_embeddings[idx2]))
        y_true.append(1)

    for idx1, idx2 in neg_pairs:
        scores.append(np.dot(all_embeddings[idx1], all_embeddings[idx2]))
        y_true.append(0)

    scores = np.array(scores)
    y_true = np.array(y_true)

    # 5. Lấy kết quả trả về an toàn
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
            for key in batch_data:
                if key != "label":
                    batch_data[key] = batch_data[key].to(device)
            
            _, speaker_embedding, gate_weights = model(return_gates=True, **{k: v for k, v in batch_data.items() if k != "label"})
            
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
    print(f"{'AAM Margin':<30} {AAM_MARGIN}")
    print(f"{'AAM Scale':<30} {AAM_SCALE}")
    print(f"\n{'Experiment Dir':<30} {exp_dir}")
    print("="*80 + "\n")

    # Save config snapshot
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
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "weight_decay": args.weight_decay,
        "mixed_precision": args.mixed_precision,
        "early_stop_patience": args.early_stop_patience,
        "lr_scheduler": args.lr_scheduler,
        "aam_margin": AAM_MARGIN,
        "aam_scale": AAM_SCALE,
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # Create dataloaders
    print("Loading Train/Val data...")
    train_loader, val_loader, speaker_to_idx, num_speakers = create_train_val_loaders(
        args.embedding_path, args.feature_path, args.mode, args.batch_size, num_workers=0
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
        use_gating=args.use_gating
    )

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
    scaler = GradScaler('cuda') if args.mixed_precision else None

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
        "val_eer": [], "val_mindcf": [] # Thêm 2 field mới
    }

    print("Starting training...\n")
    for epoch in range(args.num_epochs):
        #current_margin = update_aam_margin(criterion, epoch, final_margin=AAM_MARGIN, warmup_epochs=5)
        #print(f"\n[Info] Epoch {epoch + 1}: AAM-Softmax Margin set to {current_margin:.4f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, opt, criterion, scaler, epoch, device,
            use_augment=getattr(args, 'use_augment', False)
        )

        # Validate (Nhận thêm eer và mindcf)
        val_eer, val_mindcf = validate(model, val_loader, device)

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
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            checkpoint_path = os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)
            # Tái sử dụng hàm save (vẫn lưu best_val_eer vào field best_loss cho tương thích)
            save_checkpoint(model, opt, epoch, best_val_eer, checkpoint_path)
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
    final_results = {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "config": config_snapshot,
        "best_val_eer": float(best_val_eer), 
        "best_val_mindcf": float(training_history["val_mindcf"][training_history["val_eer"].index(best_val_eer)]),
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
    print(f"  Best validation EER: {best_val_eer:.4f}")
    print(f"  Experiment dir: {exp_dir}")
    print(f"  Config: {os.path.join(exp_dir, 'config.json')}")
    print(f"  Results: {os.path.join(exp_dir, 'results.json')}")
    print(f"  Model: {os.path.join(exp_dir, BEST_MODEL_NAME)}")

    writer.close()
    
    return model, training_history, exp_dir