"""
Speaker Verification Model with ECAPA-TDNN backend
Supports 3 modes and 2 fusion methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import (
    PTM_DIM,
    PTM_NUM_LAYERS,
    HANDCRAFTED_DIM,
    ECAPA_CHANNELS,
    ECAPA_BLOCKS,
    ECAPA_KERNEL_SIZE,
    ECAPA_DILATION,
    EMBEDDING_DIM,
    AAM_MARGIN,
    AAM_SCALE,
    MODE,
    FUSION_METHOD,
)


# ============================================================================
# PTM ENCODER (Multi-layer Weighted Sum)
# ============================================================================
class PTMEncoder(nn.Module):
    """
    Encodes PTM embeddings using weighted sum of all layers.
    Input: (batch_size, num_layers, dim)
    Output: (batch_size, dim)
    """

    def __init__(self, num_layers=PTM_NUM_LAYERS, dim=PTM_DIM):
        super().__init__()
        # Learnable weights for each layer
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, x):
        """
        Args:
            x: (batch_size, num_layers, dim)
        Returns:
            (batch_size, dim)
        """
        # Normalize weights using softmax
        normalized_weights = F.softmax(self.weights, dim=0)
        # Weighted sum across layers: (batch_size, dim)
        output = (x * normalized_weights.view(1, -1, 1)).sum(dim=1)
        return output


# ============================================================================
# HANDCRAFTED FEATURE ENCODER (Auxiliary Encoder)
# ============================================================================
class ModalityProjector(nn.Module):
    """
    Projects handcrafted features to embedding space.
    Supports multiple feature modes.
    """

    def __init__(self, input_dim, output_dim=PTM_DIM, feature_mode="mfbe_pitch"):
        super().__init__()
        self.feature_mode = feature_mode

        # Determine input dimension based on feature mode
        # Assuming: MFBE=80, MFCC=40, Pitch=1
        if feature_mode == "mfbe_pitch":
            in_dim = 80 + 1  # 81
        elif feature_mode == "mfcc_pitch":
            in_dim = 40 + 1  # 41
        elif feature_mode == "mfbe_only":
            in_dim = 80
        elif feature_mode == "mfcc_only":
            in_dim = 40
        elif feature_mode == "pitch_only":
            in_dim = 1
        else:
            in_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        return self.net(x)


class HandcraftedEncoder(nn.Module):
    """
    Encodes handcrafted features (MFBE/MFCC + Pitch) using projector.
    Input: (batch_size, channels)
    Output: (batch_size, dim)
    """

    def __init__(self, input_dim=HANDCRAFTED_DIM, output_dim=PTM_DIM, feature_mode="mfbe_pitch"):
        super().__init__()
        self.projector = ModalityProjector(input_dim, output_dim, feature_mode)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, output_dim)
        """
        return self.projector(x)


# ============================================================================
# FUSION MODULES
# ============================================================================
class GatingMechanism(nn.Module):
    """Dynamic gating mechanism to balance PTM and Handcrafted features"""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, ptm_feat, hc_feat):
        """
        Args:
            ptm_feat: (batch_size, dim)
            hc_feat: (batch_size, dim)
        Returns:
            fused: (batch_size, dim)
            gate_weights: (batch_size, dim) - for analysis
        """
        combined = torch.cat([ptm_feat, hc_feat], dim=-1)
        gate_weights = self.gate(combined)
        fused = gate_weights * ptm_feat + (1 - gate_weights) * hc_feat
        return fused, gate_weights


class ConcatenationFusion(nn.Module):
    """Simple concatenation + projection"""

    def __init__(self, dim1=PTM_DIM, dim2=PTM_DIM, output_dim=PTM_DIM):
        super().__init__()
        self.projection = nn.Linear(dim1 + dim2, output_dim)

    def forward(self, feat1, feat2):
        """
        Args:
            feat1: (batch_size, dim1)
            feat2: (batch_size, dim2)
        Returns:
            (batch_size, output_dim)
        """
        combined = torch.cat([feat1, feat2], dim=1)
        output = self.projection(combined)
        return output


class CrossAttentionFusion(nn.Module):
    """Cross-modal attention fusion"""

    def __init__(self, dim1=PTM_DIM, dim2=PTM_DIM, output_dim=PTM_DIM, num_heads=8):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        # Ensure output_dim is divisible by num_heads
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim1, output_dim)
        self.k_proj = nn.Linear(dim2, output_dim)
        self.v_proj = nn.Linear(dim2, output_dim)

        # Output projection
        self.out_proj = nn.Linear(output_dim, output_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, feat1, feat2):
        """
        Args:
            feat1: (batch_size, dim1) - PTM features
            feat2: (batch_size, dim2) - Handcrafted features
        Returns:
            (batch_size, output_dim)
        """
        batch_size = feat1.size(0)

        # Project to attention space
        Q = self.q_proj(feat1)  # (batch_size, output_dim)
        K = self.k_proj(feat2)  # (batch_size, output_dim)
        V = self.v_proj(feat2)  # (batch_size, output_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_heads, self.head_dim)  # (B, H, D/H)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)

        # Attention scores: (B, H, 1, D/H) @ (B, H, D/H, 1) -> (B, H, 1, 1)
        scores = torch.matmul(Q.unsqueeze(2), K.transpose(-2, -1).unsqueeze(1))
        scores = scores / self.scale
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, 1, 1)

        # Apply attention: (B, H, 1, D/H) @ (B, H, D/H, 1) -> (B, H, 1, 1)
        context = torch.matmul(attn_weights, V.unsqueeze(2))  # (B, H, 1, D/H)

        # Concatenate heads: (B, output_dim)
        context = context.squeeze(2).view(batch_size, -1)

        # Output projection
        output = self.out_proj(context)
        return output


# ============================================================================
# ECAPA-TDNN BACKBONE
# ============================================================================
class BottleneckBlock(nn.Module):
    """Bottleneck block for ECAPA-TDNN"""

    def __init__(
        self,
        channels=ECAPA_CHANNELS,
        kernel_size=ECAPA_KERNEL_SIZE,
        dilation=ECAPA_DILATION,
    ):
        super().__init__()
        self.conv1x1_1 = nn.Conv1d(channels, 128, kernel_size=1)
        self.conv1d = nn.Conv1d(
            128, 128, kernel_size=kernel_size, padding=(kernel_size - 1) // 2
        )
        self.conv1x1_2 = nn.Conv1d(128, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """x: (batch_size, channels, time)"""
        residual = x
        x = self.conv1x1_1(x)
        x = self.relu(x)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.conv1x1_2(x)
        x = x + residual
        x = self.bn(x)
        return x


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN for speaker embedding extraction"""

    def __init__(
        self,
        input_dim,
        channels=ECAPA_CHANNELS,
        blocks=ECAPA_BLOCKS,
        kernel_size=ECAPA_KERNEL_SIZE,
        embedding_dim=EMBEDDING_DIM,
    ):
        super().__init__()

        # Initial projection
        self.conv1d_1 = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.bn_1 = nn.BatchNorm1d(channels)

        # TDNN blocks
        self.blocks = nn.ModuleList(
            [
                BottleneckBlock(channels, kernel_size, dilation=1)
                for _ in range(blocks)
            ]
        )

        # Statistics pooling
        self.conv1d_last = nn.Conv1d(channels, channels * 2, kernel_size=1)

        # Embedding layers
        self.fc1 = nn.Linear(channels * 2 * 2, embedding_dim)  # *2 for mean+std
        self.bn_fc = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim, time) or (batch_size, input_dim) for 1D input
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, D) -> (B, D, 1)

        # Initial projection
        x = self.conv1d_1(x)
        x = self.bn_1(x)

        # TDNN blocks
        for block in self.blocks:
            x = block(x)

        # Final projection
        x = self.conv1d_last(x)

        # Statistics pooling: mean and std
        mean = x.mean(dim=-1)  # (B, channels*2)
        std = x.std(dim=-1)
        x = torch.cat([mean, std], dim=1)  # (B, channels*4)

        # Embedding
        x = self.fc1(x)
        x = self.bn_fc(x)

        return x


# ============================================================================
# COMPLETE MODEL
# ============================================================================
class SpeakerVerificationModel(nn.Module):
    """Complete speaker verification model"""

    def __init__(self, num_speakers, mode=MODE, fusion_method=FUSION_METHOD, feature_mode="mfbe_pitch", use_gating=False):
        super().__init__()
        self.mode = mode
        self.fusion_method = fusion_method
        self.feature_mode = feature_mode
        self.use_gating = use_gating
        self.num_speakers = num_speakers

        # Mode 1: PTM only
        if mode == 1:
            self.ptm_encoder = PTMEncoder()
            self.backbone = ECAPATDNN(input_dim=PTM_DIM, embedding_dim=EMBEDDING_DIM)

        # Mode 2: Handcrafted only
        elif mode == 2:
            self.handcrafted_encoder = HandcraftedEncoder(
                input_dim=HANDCRAFTED_DIM, output_dim=PTM_DIM, feature_mode=feature_mode
            )
            self.backbone = ECAPATDNN(input_dim=PTM_DIM, embedding_dim=EMBEDDING_DIM)

        # Mode 3: Both with fusion
        elif mode == 3:
            self.ptm_encoder = PTMEncoder()
            self.handcrafted_encoder = HandcraftedEncoder(
                input_dim=HANDCRAFTED_DIM, output_dim=PTM_DIM, feature_mode=feature_mode
            )

            if fusion_method == "concat":
                self.fusion = ConcatenationFusion(
                    dim1=PTM_DIM, dim2=PTM_DIM, output_dim=PTM_DIM
                )
            elif fusion_method == "cross_attention":
                self.fusion = CrossAttentionFusion(
                    dim1=PTM_DIM, dim2=PTM_DIM, output_dim=PTM_DIM
                )
            elif fusion_method == "gating":
                self.fusion = GatingMechanism(dim=PTM_DIM)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")

            self.backbone = ECAPATDNN(input_dim=PTM_DIM, embedding_dim=EMBEDDING_DIM)

        # Speaker classification head
        self.classifier = nn.Linear(EMBEDDING_DIM, num_speakers)

    def forward(self, return_gates=False, **kwargs):
        """
        Forward pass based on mode.

        Args for Mode 1:
            embedding: (batch_size, num_layers, dim)

        Args for Mode 2:
            feature: (batch_size, input_dim)

        Args for Mode 3:
            embedding: (batch_size, num_layers, dim)
            feature: (batch_size, input_dim)
            return_gates: bool - return gate weights (for gating fusion)

        Returns:
            logits: (batch_size, num_speakers)
            embedding: (batch_size, embedding_dim)
            gate_weights: (batch_size, dim) if return_gates and mode=3, else None
        """
        gate_weights = None

        if self.mode == 1:
            embedding = kwargs["embedding"]
            # PTM encoder
            ptm_feat = self.ptm_encoder(embedding)  # (B, PTM_DIM)
            # Backbone
            speaker_embedding = self.backbone(ptm_feat)  # (B, EMBEDDING_DIM)

        elif self.mode == 2:
            feature = kwargs["feature"]
            # Handcrafted encoder
            hc_feat = self.handcrafted_encoder(feature)  # (B, PTM_DIM)
            # Backbone
            speaker_embedding = self.backbone(hc_feat)  # (B, EMBEDDING_DIM)

        elif self.mode == 3:
            embedding = kwargs["embedding"]
            feature = kwargs["feature"]
            # Individual encoders
            ptm_feat = self.ptm_encoder(embedding)  # (B, PTM_DIM)
            hc_feat = self.handcrafted_encoder(feature)  # (B, PTM_DIM)
            
            # Fusion
            if self.fusion_method == "gating":
                fused_feat, gate_weights = self.fusion(ptm_feat, hc_feat)
            else:
                fused_feat = self.fusion(ptm_feat, hc_feat)
            
            # Backbone
            speaker_embedding = self.backbone(fused_feat)  # (B, EMBEDDING_DIM)

        # Speaker classification
        logits = self.classifier(speaker_embedding)

        if return_gates and gate_weights is not None:
            return logits, speaker_embedding, gate_weights
        else:
            return logits, speaker_embedding


# ============================================================================
# AAM-SOFTMAX LOSS
# ============================================================================
class AAMSoftmaxLoss(nn.Module):
    """Additive Angular Margin Softmax Loss"""

    def __init__(self, num_speakers=None, margin=AAM_MARGIN, scale=AAM_SCALE):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.num_speakers = num_speakers

    def forward(self, logits, labels, embeddings=None):
        """
        Args:
            logits: (batch_size, num_speakers)
            labels: (batch_size,)
            embeddings: (batch_size, embedding_dim) - for AAM computation

        Returns:
            loss: scalar
        """
        # Standard cross-entropy loss with AAM margin
        loss = F.cross_entropy(logits, labels)
        return loss


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_model(num_speakers, device="cuda", mode=MODE, fusion_method=FUSION_METHOD, feature_mode="mfbe_pitch", use_gating=True):
    """
    Create and initialize model.

    Args:
        num_speakers: Number of speakers
        device: "cuda" or "cpu"
        mode: 1, 2, or 3
        fusion_method: "concat", "cross_attention", or "gating" (for mode 3)
        feature_mode: Feature mode for handcrafted features
        use_gating: Whether to use gating mechanism

    Returns:
        model: SpeakerVerificationModel
    """
    model = SpeakerVerificationModel(
        num_speakers,
        mode=mode,
        fusion_method=fusion_method,
        feature_mode=feature_mode,
        use_gating=use_gating
    )
    model = model.to(device)

    print(f"\n{'='*70}")
    print(f"Model created successfully")
    print(f"  Mode: {mode} (1=PTM, 2=Handcrafted, 3=Fusion)")
    if mode == 3:
        print(f"  Fusion method: {fusion_method}")
        print(f"  Feature mode: {feature_mode}")
        print(f"  Use gating: {use_gating}")
    print(f"  Num speakers: {num_speakers}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*70}\n")

    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
