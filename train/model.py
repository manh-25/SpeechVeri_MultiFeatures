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
    DIM_MAP,
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
        self.norm = nn.LayerNorm(dim)

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
        output = self.norm(output)
        return output


# ============================================================================
# HANDCRAFTED FEATURE ENCODER (Auxiliary Encoder)
# ============================================================================
class ModalityProjector(nn.Module):
    """
    Projects handcrafted features to embedding space.
    Đã nâng cấp Kernel Size để bắt ngữ cảnh thời gian (Temporal Context).
    """
    def __init__(self, input_dim=HANDCRAFTED_DIM, output_dim=PTM_DIM):
        super().__init__()
        self.net = nn.Sequential(
            # Nhìn rộng ra 5 frames
            nn.Conv1d(input_dim, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # Nhìn rộng ra 3 frames
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, output_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)

class HandcraftedEncoder(nn.Module):
    def __init__(self, input_dim=HANDCRAFTED_DIM, output_dim=PTM_DIM, feature_mode="mfbe_pitch"):
        super().__init__()
        self.projector = ModalityProjector(input_dim, output_dim)

    def forward(self, x):
        return self.projector(x)


# ============================================================================
# FUSION MODULES
# ============================================================================
class GatingMechanism(nn.Module):
    """Dynamic gating mechanism to balance PTM and Handcrafted features"""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, ptm_feat, hc_feat):
        # ptm_feat, hc_feat: (B, D, T)
        combined = torch.cat([ptm_feat, hc_feat], dim=1)
        gate_weights = self.gate(combined) # (B, D, T)
        fused = gate_weights * ptm_feat + (1 - gate_weights) * hc_feat
        return fused, gate_weights


class ConcatenationFusion(nn.Module):
    """Simple concatenation + projection"""

    def __init__(self, dim1=PTM_DIM, dim2=PTM_DIM, output_dim=PTM_DIM):
        super().__init__()
        self.projection = nn.Conv1d(dim1 + dim2, output_dim, kernel_size=1)

    def forward(self, feat1, feat2):
        combined = torch.cat([feat1, feat2], dim=1)
        return self.projection(combined)


class CrossAttentionFusion(nn.Module):
    """Cross-modal attention fusion (Handcrafted temporal query -> PTM static context)"""
    def __init__(self, dim1=PTM_DIM, dim2=PTM_DIM, output_dim=PTM_DIM, num_heads=8):
        super().__init__()
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)

        # dim1 là PTM (static), dim2 là HC (temporal)
        self.q_proj = nn.Conv1d(dim2, output_dim, kernel_size=1) # HC làm Query
        self.k_proj = nn.Conv1d(dim1, output_dim, kernel_size=1) # PTM làm Key
        self.v_proj = nn.Conv1d(dim1, output_dim, kernel_size=1) # PTM làm Value
        self.out_proj = nn.Conv1d(output_dim, output_dim, kernel_size=1)

    def forward(self, ptm_static, hc_temporal):
        # Đảm bảo PTM có shape (B, D, 1)
        if ptm_static.dim() == 2:
            ptm_static = ptm_static.unsqueeze(-1)
            
        # Q lấy từ HC (B, T, D) | K, V lấy từ PTM (B, 1, D)
        Q = self.q_proj(hc_temporal).transpose(1, 2) 
        K = self.k_proj(ptm_static).transpose(1, 2)
        V = self.v_proj(ptm_static).transpose(1, 2)

        # Output sẽ tự động có shape (B, T, D)
        attn_output, _ = self.mha(query=Q, key=K, value=V)

        # Transpose về lại (B, D, T) cho Convolution/ECAPA
        output = self.out_proj(attn_output.transpose(1, 2))
        
        return output

class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation (FiLM) Fusion"""
    def __init__(self, dim1=PTM_DIM, dim2=PTM_DIM, output_dim=PTM_DIM):
        super().__init__()
        # dim1 là PTM (tĩnh), dim2 là Handcrafted (động)
        # Sinh ra Scale (gamma) và Shift (beta) từ PTM
        self.film_gen = nn.Linear(dim1, dim2 * 2)
        self.out_proj = nn.Conv1d(dim2, output_dim, kernel_size=1)

    def forward(self, ptm_static, hc_temporal):
        # ptm_static: (B, D)
        # hc_temporal: (B, D, T)
        
        # Nếu PTM lỡ bị chèn thêm chiều T (B, D, 1), thì bóp nó lại
        if ptm_static.dim() == 3:
            ptm_static = ptm_static.squeeze(2)
            
        film_params = self.film_gen(ptm_static) # (B, 2*D)
        gamma, beta = torch.chunk(film_params, 2, dim=1) # (B, D), (B, D)
        
        # Reshape để broadcast dọc theo chiều thời gian T
        gamma = gamma.unsqueeze(2) # (B, D, 1)
        beta = beta.unsqueeze(2)   # (B, D, 1)
        
        # Modulation: Đặc trưng ngữ âm (HC) bị uốn nắn bởi ngữ cảnh người nói (PTM)
        fused = (hc_temporal * gamma) + beta
        return self.out_proj(fused)

# ============================================================================
# SQUEEZE-AND-EXCITATION & ASP POOLING
# ============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block 1D"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class AttentiveStatisticsPooling(nn.Module):
    """PTM-Guided Attentive Statistics Pooling"""
    def __init__(self, channels, ptm_dim=PTM_DIM, use_ptm_guide=False):
        super().__init__()
        self.use_ptm_guide = use_ptm_guide
        attn_in_channels = channels + ptm_dim if use_ptm_guide else channels
        
        self.attention = nn.Sequential(
            nn.Conv1d(attn_in_channels, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, channels, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x, ptm_context=None):
        if self.use_ptm_guide and ptm_context is not None:
            if ptm_context.dim() == 2:
                ptm_context = ptm_context.unsqueeze(2)
            ptm_exp = ptm_context.expand(-1, -1, x.size(2))
            attn_input = torch.cat([x, ptm_exp], dim=1)
        else:
            attn_input = x

        w = self.attention(attn_input) 
        
        x_f32 = x.float()
        w_f32 = w.float()
        
        mu = torch.sum(x_f32 * w_f32, dim=2)
        sg = torch.sqrt((torch.sum((x_f32**2) * w_f32, dim=2) - mu**2).clamp(min=1e-5))
        
        return torch.cat((mu, sg), 1).type_as(x)

# ============================================================================
# ECAPA-TDNN BACKBONE (NÂNG CẤP SOTA)
# ============================================================================
class BottleneckBlock(nn.Module):
    """Bottleneck block tích hợp Squeeze-and-Excitation (SE)"""
    def __init__(
        self,
        channels=ECAPA_CHANNELS,
        kernel_size=ECAPA_KERNEL_SIZE,
        dilation=1,
    ):
        super().__init__()
        self.conv1x1_1 = nn.Conv1d(channels, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.conv1d = nn.Conv1d(
            128, 128, kernel_size=kernel_size, 
            padding=(kernel_size * dilation - dilation) // 2, # Padding chuẩn để không lệch size
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv1x1_2 = nn.Conv1d(128, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(channels)
        
        # Thêm não SE
        self.se = SEBlock(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1x1_1(x)))
        x = self.relu(self.bn2(self.conv1d(x)))
        x = self.bn3(self.conv1x1_2(x))
        x = self.se(x) # Lọc nhiễu channel trước khi cộng
        x = x + residual
        return x


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN (Tích hợp SE, Dilation động, MFA và PTM-Guided ASP)"""
    def __init__(
        self,
        input_dim,
        channels=ECAPA_CHANNELS,
        blocks=ECAPA_BLOCKS,
        kernel_size=ECAPA_KERNEL_SIZE,
        embedding_dim=EMBEDDING_DIM,
        use_ptm_guide=False
    ):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.bn_1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        self.blocks = nn.ModuleList([
            BottleneckBlock(channels, kernel_size, dilation=d)
            for d in range(2, 2 + blocks)
        ])

        self.mfa = nn.Conv1d(channels * (blocks + 1), channels * 3, kernel_size=1)

        self.pooling = AttentiveStatisticsPooling(channels * 3, use_ptm_guide=use_ptm_guide)
        self.bn_pool = nn.BatchNorm1d(channels * 3 * 2)

        self.fc1 = nn.Linear(channels * 3 * 2, embedding_dim)
        self.bn_fc = nn.BatchNorm1d(embedding_dim)

    def forward(self, x, ptm_context=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.size(-1) == 1:
            x = x.expand(-1, -1, 10)

        x = self.relu(self.bn_1(self.conv1d_1(x)))

        layer_outputs = [x]
        for block in self.blocks:
            x = block(x)
            layer_outputs.append(x)

        x = torch.cat(layer_outputs, dim=1)
        x = self.mfa(x)

        x = self.pooling(x, ptm_context=ptm_context)
        x = self.bn_pool(x)

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

        actual_input_dim = DIM_MAP.get(feature_mode, 81)

        # Mode 1: PTM only
        if mode == 1:
            self.ptm_encoder = PTMEncoder()
            self.backbone = ECAPATDNN(input_dim=PTM_DIM, embedding_dim=EMBEDDING_DIM, use_ptm_guide=False)

        # Mode 2: Handcrafted only
        elif mode == 2:
            self.handcrafted_encoder = HandcraftedEncoder(
                input_dim=actual_input_dim, output_dim=PTM_DIM, feature_mode=feature_mode
            )
            self.backbone = ECAPATDNN(input_dim=PTM_DIM, embedding_dim=EMBEDDING_DIM, use_ptm_guide=False)

        # Mode 3: Both with fusion
        elif mode == 3:
            self.ptm_encoder = PTMEncoder()
            self.handcrafted_encoder = HandcraftedEncoder(
                input_dim=actual_input_dim, output_dim=PTM_DIM, feature_mode=feature_mode
            )

            if fusion_method == "concat":
                self.fusion = ConcatenationFusion(dim1=PTM_DIM, dim2=PTM_DIM, output_dim=PTM_DIM)
            elif fusion_method == "cross_attention":
                self.fusion = CrossAttentionFusion(dim1=PTM_DIM, dim2=PTM_DIM, output_dim=PTM_DIM)
            elif fusion_method == "gating":
                self.fusion = GatingMechanism(dim=PTM_DIM)
            elif fusion_method == "film": # <--- FiLM FUSION ĐÂY
                self.fusion = FiLMFusion(dim1=PTM_DIM, dim2=PTM_DIM, output_dim=PTM_DIM)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")

            # Khai báo ECAPA có bật tính năng Guide
            self.backbone = ECAPATDNN(input_dim=PTM_DIM, embedding_dim=EMBEDDING_DIM, use_ptm_guide=True)
       

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
            feature = kwargs["feature"] # (B, C_hc, T)
            # Handcrafted encoder
            hc_feat = self.handcrafted_encoder(feature)  # (B, 768, T)
            # Backbone
            speaker_embedding = self.backbone(hc_feat)  # (B, EMBEDDING_DIM)

        elif self.mode == 3:
            embedding = kwargs["embedding"] 
            feature = kwargs["feature"]     
            
            ptm_feat = self.ptm_encoder(embedding)       
            hc_feat = self.handcrafted_encoder(feature)  
            
            # Fusion
            if self.fusion_method == "cross_attention":
                fused_feat = self.fusion(ptm_feat, hc_feat)
            elif self.fusion_method == "film":
                fused_feat = self.fusion(ptm_feat, hc_feat)
            else:
                T = feature.size(-1)
                ptm_feat_expanded = ptm_feat.unsqueeze(-1).expand(-1, -1, T)
                
                if self.fusion_method == "gating":
                    fused_feat, gate_weights = self.fusion(ptm_feat_expanded, hc_feat)
                else:
                    fused_feat = self.fusion(ptm_feat_expanded, hc_feat)
            
            # Backbone nhận thêm ptm_feat làm guide cho lớp Pooling
            speaker_embedding = self.backbone(fused_feat, ptm_context=ptm_feat)

        if return_gates and gate_weights is not None:
            return None, speaker_embedding, gate_weights
        else:
            return None, speaker_embedding


# ============================================================================
# AAM-SOFTMAX LOSS
# ============================================================================
class AAMSoftmaxLoss(nn.Module):
    """Additive Angular Margin Softmax Loss"""

    def __init__(self, num_speakers, embedding_dim=EMBEDDING_DIM, margin=AAM_MARGIN, scale=AAM_SCALE):
        super(AAMSoftmaxLoss, self).__init__()
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale

        # Trọng số của các speaker (prototypes)
        self.weight = nn.Parameter(torch.FloatTensor(num_speakers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Các hằng số tính toán sẵn để tăng tốc
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, logits, labels, embeddings=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-7, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        loss = F.cross_entropy(output, labels)
        return loss, output


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
