"""
Training script for Speaker Verification
Includes: Training loop, validation, early stopping, LR scheduling, metrics computation
Features: Experiment management, model summary, TensorBoard, gating analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
import json
import shutil
from datetime import datetime
from tqdm import tqdm
from torchinfo import summary
import matplotlib.pyplot as plt
import seaborn as sns

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
)
from model import SpeakerVerificationModel, AAMSoftmaxLoss, get_model
from dataset import create_data_loaders


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def compute_eer(y_true, y_scores):
    """
    Compute Equal Error Rate (EER) - primary metric for speaker verification
    EER is the threshold where FAR = FRR
    """
    fpr, fnr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = np.min(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_idx]
    return eer, eer_threshold


def compute_mindcf(y_true, y_scores, p_target=0.01, c_miss=1, c_fa=1):
    """
    Compute Minimum Detection Cost Function (MinDCF)
    Used in NIST speaker recognition evaluation
    """
    fpr, fnr, _ = roc_curve(y_true, y_scores, pos_label=1)
    mindcf = np.min(c_miss * fnr * p_target + c_fa * fpr * (1 - p_target))
    return mindcf


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


def train_epoch(model, train_loader, optimizer, criterion, scaler, epoch, device, log_interval=LOG_INTERVAL):
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

        optimizer.zero_grad()

        # Forward pass with mixed precision
        if MIXED_PRECISION:
            with autocast():
                _, embeddings = model(**inputs)
                loss, logits = criterion(None, labels, embeddings=embeddings)
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


def validate(model, val_loader, criterion, device):
    """
    Validate model.

    Args:
        model: nn.Module
        val_loader: DataLoader
        criterion: Loss function
        device: Device to validate on

    Returns:
        avg_loss: Average validation loss
        avg_accuracy: Average validation accuracy
    """
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        for batch_data in progress_bar:
            # Move data to device
            labels = batch_data["label"].to(device)
            inputs = {k: v.to(device) for k, v in batch_data.items() if k != "label"}

            # Forward pass
            _, embeddings = model(**inputs)
            loss, logits = criterion(None, labels, embeddings=embeddings)

            # Metrics
            accuracy = compute_metrics(logits, labels)
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


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
    ax.hist(all_gates, bins=50, alpha=0.7, color='purple', edgecolor='black')
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
    
    print(f"  PTM Priority (g > 0.5): {ptm_priority} / {len(all_gates)} ({100*ptm_priority/len(all_gates):.1f}%)")
    print(f"  HC Priority (g <= 0.5): {hc_priority} / {len(all_gates)} ({100*hc_priority/len(all_gates):.1f}%)")
    
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device(args.device)

    # Auto-generate experiment name if not provided
    if args.exp_name is None:
        args.exp_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create experiment directory
    exp_dir = os.path.join(args.output_dir, "experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

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
        "mode": args.mode,
        "fusion_method": args.fusion_method,
        "feature_mode": args.feature_mode,
        "use_gating": args.use_gating,
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
    print("Loading data...")
    train_loader, val_loader, test_loader, speaker_to_idx, num_speakers = (
        create_data_loaders(
            args.embedding_path, args.feature_path, args.mode, args.batch_size, num_workers=0
        )
    )
    print(f"✓ Loaded {num_speakers} speakers")
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}\n")

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
        # Determine input key and dimension based on mode
        if args.mode == 3:
            input_data = { "embedding": (args.batch_size, PTM_NUM_LAYERS, PTM_DIM), 
                          "feature": (args.batch_size, HANDCRAFTED_DIM, 200) 
                          }
        else:
            input_key = "embedding" if args.mode == 1 else "feature"
            dim = PTM_DIM if args.mode == 1 else HANDCRAFTED_DIM # Nếu là Mode 2 (ECAPA), cần chiều T (ví dụ 200). Mode 1 là vector tĩnh
            input_data = {input_key: (args.batch_size, dim, 200 if args.mode == 2 else None)}

        model_summary = summary(model, input_size=input_data, verbose=0, device=str(device))
        
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
        opt = optim.Adam(
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
            verbose=True,
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_scheduler}")

    # Mixed precision
    scaler = GradScaler() if args.mixed_precision else None

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stop_patience, delta=args.early_stop_delta)

    # Logging
    log_file = os.path.join(exp_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Training started: {datetime.now()}\n")
        f.write(json.dumps(config_snapshot, indent=2) + "\n\n")

    # Training loop
    best_val_loss = float("inf")
    training_history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    print("Starting training...\n")
    for epoch in range(args.num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, opt, criterion, scaler, epoch, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        if args.lr_scheduler.lower() == "cosine":
            scheduler.step()
        elif args.lr_scheduler.lower() == "plateau":
            scheduler.step(val_loss)

        # Update history
        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(train_acc)
        training_history["val_loss"].append(val_loss)
        training_history["val_accuracy"].append(val_acc)

        # Logging
        log_msg = (
            f"Epoch {epoch + 1:3d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
            f"LR: {opt.param_groups[0]['lr']:.6f}"
        )
        print(log_msg)
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, BEST_MODEL_NAME)
            save_checkpoint(model, opt, epoch, best_val_loss, checkpoint_path)
            # Copy to exp folder
            shutil.copy(checkpoint_path, os.path.join(exp_dir, BEST_MODEL_NAME))

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("\n✓ Early stopping triggered!")
            with open(log_file, "a") as f:
                f.write("Early stopping triggered!\n")
            break

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, FINAL_MODEL_NAME)
    save_checkpoint(model, opt, epoch, best_val_loss, final_path)
    shutil.copy(final_path, os.path.join(exp_dir, FINAL_MODEL_NAME))

    # Save history
    history_path = os.path.join(exp_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=4)

    # Load best model for testing
    model, _, _, _ = load_checkpoint(os.path.join(exp_dir, BEST_MODEL_NAME), model)

    # Analyze gating if applicable
    if args.mode == 3 and args.fusion_method == "gating":
        gates, labels = analyze_gating_behavior(model, test_loader, device, exp_dir)
    else:
        gates, labels = None, None

    # Save final results
    final_results = {
        "exp_name": args.exp_name,
        "timestamp": datetime.now().isoformat(),
        "config": config_snapshot,
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(training_history["train_loss"][-1]),
        "final_val_loss": float(training_history["val_loss"][-1]),
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
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Experiment dir: {exp_dir}")
    print(f"  Config: {os.path.join(exp_dir, 'config.json')}")
    print(f"  Results: {os.path.join(exp_dir, 'results.json')}")
    print(f"  Model: {os.path.join(exp_dir, BEST_MODEL_NAME)}")

    return model, training_history, exp_dir