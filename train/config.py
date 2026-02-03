"""
Configuration file for Speaker Verification Training
"""

# ============================================================================
# DATA PATHS
# ============================================================================
EMBEDDING_PATH = "path/to/embedding.pt"  # PTM embeddings (N, 13, 768)
FEATURE_PATH = "path/to/feature.pt"      # Handcrafted features (N, D, T)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Mode selection: 1 (PTM only), 2 (Handcrafted only), 3 (PTM + Handcrafted)
MODE = 1

# Fusion method for MODE 3: "concat", "cross_attention", or "gating"
FUSION_METHOD = "gating"

# Feature mode for handcrafted features: "mfbe_pitch", "mfcc_pitch", "mfbe_only", "mfcc_only", "pitch_only"
FEATURE_MODE = "mfbe_pitch" 

# 2. Định nghĩa mapping kích thước
# Mel-filterbank (MFBE) thường là 80, MFCC là 40, Pitch là 1
DIM_MAP = {
    "mfbe_pitch": 80 + 1,  # 81
    "mfcc_pitch": 40 + 1,  # 41
    "mfbe_only": 80,
    "mfcc_only": 40,
    "pitch_only": 1
}

# 3. Tự động gán dimension
HANDCRAFTED_DIM = DIM_MAP.get(FEATURE_MODE, 81)

# Use gating mechanism for fusion (only for MODE 3)
USE_GATING = True

# Feature dimensions
PTM_DIM = 768  # WavLM/HuBERT/Wav2Vec2 output dimension
PTM_NUM_LAYERS = 13  # Number of layers to use

# ECAPA-TDNN backbone
ECAPA_CHANNELS = 512
ECAPA_BLOCKS = 4
ECAPA_KERNEL_SIZE = 5
ECAPA_DILATION = 1

# Embedding dimension (before speaker classification)
EMBEDDING_DIM = 512

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.0001

# Early stopping
EARLY_STOP_PATIENCE = 10
EARLY_STOP_DELTA = 1e-4  # Minimum change to qualify as improvement

# Learning rate scheduler
LR_SCHEDULER = "cosine"  # "cosine" or "plateau"
COSINE_T_MAX = 50
PLATEAU_PATIENCE = 5
PLATEAU_FACTOR = 0.5

# ============================================================================
# AAM-SOFTMAX LOSS
# ============================================================================
AAM_MARGIN = 0.2  # m in AAM-Softmax
AAM_SCALE = 30    # s in AAM-Softmax

# ============================================================================
# OPTIMIZATION
# ============================================================================
OPTIMIZER = "adam"  # "adam" or "sgd"
MOMENTUM = 0.9
NESTEROV = True

# ============================================================================
# DEVICE & PRECISION
# ============================================================================
DEVICE = "cuda"  # "cuda" or "cpu"
MIXED_PRECISION = True  # Use AMP (Automatic Mixed Precision)

# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================
LOG_INTERVAL = 10  # Log every N batches
CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_NAME = "best_model.pth"
FINAL_MODEL_NAME = "final_model.pth"
LOG_FILE = "training_log.txt"

# ============================================================================
# DATA SPLIT
# ============================================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ============================================================================
# INFERENCE
# ============================================================================
CONFIDENCE_THRESHOLD = 0.5
SAVE_SALIENCY_MAP = True
SALIENCY_OUTPUT_DIR = "./saliency_maps"
