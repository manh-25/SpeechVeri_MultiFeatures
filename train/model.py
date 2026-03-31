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


class LayerAttentionPooling(nn.Module):
    """Attentive pooling over the PTM layer dimension.

    Input:  x (B, L, D)
    Output: pooled (B, D)
    """

    def __init__(self, dim: int = PTM_DIM):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        attn_logits = self.score(x).squeeze(-1)  # (B, L)
        attn = F.softmax(attn_logits, dim=1).unsqueeze(-1)  # (B, L, 1)
        return (x * attn).sum(dim=1)


class PTMEmbeddingHead(nn.Module):
    """PTM-only head for static utterance embedding when no time axis is available.

    Uses attention pooling across PTM layers (L=13) then projects to EMBEDDING_DIM.
    """

    def __init__(self, ptm_dim: int = PTM_DIM, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.pool = LayerAttentionPooling(dim=ptm_dim)
        self.in_norm = nn.LayerNorm(ptm_dim)
        self.proj = nn.Sequential(
            nn.Linear(ptm_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        pooled = self.pool(x)
        pooled = self.in_norm(pooled)
        return self.proj(pooled)


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


class EmbeddingGatingFusion(nn.Module):
    """Embedding-level gating fusion.

    Produces a per-dimension gate g in (0,1) and fuses:
        y = g * ptm + (1-g) * hc
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(self, ptm_emb: torch.Tensor, hc_emb: torch.Tensor, return_gate: bool = False):
        gate = self.gate_fc(torch.cat([ptm_emb, hc_emb], dim=1))
        fused = gate * ptm_emb + (1.0 - gate) * hc_emb
        if return_gate:
            return fused, gate
        return fused


class EmbeddingConcatFusion(nn.Module):
    """Embedding-level concatenation fusion + projection."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, ptm_emb: torch.Tensor, hc_emb: torch.Tensor):
        return self.proj(torch.cat([ptm_emb, hc_emb], dim=1))


class EmbeddingFiLMFusion(nn.Module):
    """Embedding-level FiLM: condition HC embedding using PTM, then combine."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(embedding_dim, embedding_dim * 2)
        self.out_ln = nn.LayerNorm(embedding_dim)

    def forward(self, ptm_emb: torch.Tensor, hc_emb: torch.Tensor):
        gamma, beta = self.to_gamma_beta(ptm_emb).chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        hc_mod = hc_emb * (1.0 + gamma) + beta
        return self.out_ln(hc_mod + ptm_emb)

# ============================================================================
# SQUEEZE-AND-EXCITATION & ASP POOLING
# ============================================================================
class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x
    
class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, channels, ptm_dim=PTM_DIM, use_ptm_guide=False):
        super().__init__()
        self.use_ptm_guide = use_ptm_guide
        
        # ECAPA chuẩn tính attention trên global_x gồm 3 phần: mfa + mean + std
        global_channels = channels * 3 
        
        # Nếu có PTM Guide, cộng thêm số chiều của PTM
        attn_in_channels = global_channels + ptm_dim if use_ptm_guide else global_channels
        
        self.attention = nn.Sequential(
            nn.Conv1d(attn_in_channels, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, channels, kernel_size=1), # Output = channels để nhân với MFA
            nn.Softmax(dim=2)
        )

    def forward(self, x, ptm_context=None):
        # x chính là output của lớp MFA: (B, channels, T)
        t = x.size()[-1]
        
        # 1. Tạo Global Context chuẩn của ECAPA
        global_x = torch.cat((
            x, 
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
        ), dim=1) # Shape: (B, channels * 3, T)
        
        # 2. Ghép thêm PTM Context làm Guide (Ý tưởng gốc của bạn)
        if self.use_ptm_guide and ptm_context is not None:
            if ptm_context.dim() == 2:
                ptm_context = ptm_context.unsqueeze(2) # Ép về (B, D, 1)
            ptm_exp = ptm_context.expand(-1, -1, t)    # Kéo giãn ra (B, D, T)
            attn_input = torch.cat([global_x, ptm_exp], dim=1)
        else:
            attn_input = global_x
            
        # 3. Tính trọng số Attention
        w = self.attention(attn_input) 
        
        # 4. Tính Weighted Mean và Std
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))
        
        # Trả về vector ghép (Mean, Std)
        return torch.cat((mu, sg), 1)

# ============================================================================
# ECAPA-TDNN BACKBONE (NÂNG CẤP SOTA)
# ============================================================================
class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i] # Mấu chốt Res2Net: Cộng dồn đặc trưng
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += residual
        return out


class ECAPATDNN(nn.Module):
    def __init__(self, input_dim, channels=ECAPA_CHANNELS, embedding_dim=EMBEDDING_DIM, use_ptm_guide=False):
        super().__init__()
        
        self.conv1d_1 = nn.Conv1d(input_dim, channels, kernel_size=5, stride=1, padding=2)
        self.bn_1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        # Khai báo cứng 3 block để dễ dàng thực hiện skip-connection lũy tiến
        self.layer1 = Bottle2neck(channels, channels, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(channels, channels, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(channels, channels, kernel_size=3, dilation=4, scale=8)
        
        # MFA Layer
        self.layer4 = nn.Conv1d(3 * channels, 1536, kernel_size=1)
        
        # ASP Block
        self.pooling = AttentiveStatisticsPooling(channels=1536, use_ptm_guide=use_ptm_guide)
        
        self.bn_pool = nn.BatchNorm1d(3072) # 1536 * 2 (mu + sg)
        self.fc1 = nn.Linear(3072, embedding_dim)
        self.bn_fc = nn.BatchNorm1d(embedding_dim)

    def forward(self, x, ptm_context=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        if x.size(-1) == 1:
            x = x.expand(-1, -1, 10)
            
        x = self.relu(self.bn_1(self.conv1d_1(x)))

        # Progressive Skip-Connections (CỰC KỲ QUAN TRỌNG)
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        mfa_out = self.relu(self.layer4(torch.cat((x1, x2, x3), dim=1)))

        # Chuyền mfa_out và ptm_context vào Pooling
        pooled = self.pooling(mfa_out, ptm_context=ptm_context)
        
        x = self.bn_pool(pooled)
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
            # Option 2: attention pooling over 13 layers -> projection to 512
            self.ptm_encoder = PTMEmbeddingHead(ptm_dim=PTM_DIM, embedding_dim=EMBEDDING_DIM)
            self.backbone = nn.Identity()

        # Mode 2: Handcrafted only
        elif mode == 2:
            self.handcrafted_encoder = HandcraftedEncoder(
                input_dim=actual_input_dim, output_dim=PTM_DIM, feature_mode=feature_mode
            )
            self.backbone = ECAPATDNN(input_dim=PTM_DIM, embedding_dim=EMBEDDING_DIM, use_ptm_guide=False)

        # Mode 3: Both with fusion
        elif mode == 3:
            # PTM branch: same as Mode 1 (no ECAPA since PTM has no true time axis)
            self.ptm_encoder = PTMEmbeddingHead(ptm_dim=PTM_DIM, embedding_dim=EMBEDDING_DIM)
            self.handcrafted_encoder = HandcraftedEncoder(
                input_dim=actual_input_dim, output_dim=PTM_DIM, feature_mode=feature_mode
            )

            # HC branch: keep ECAPA to exploit real temporal dynamics
            self.hc_backbone = ECAPATDNN(input_dim=PTM_DIM, embedding_dim=EMBEDDING_DIM, use_ptm_guide=False)

            # Light per-branch normalization before fusion
            self.ptm_emb_ln = nn.LayerNorm(EMBEDDING_DIM)
            self.hc_emb_ln = nn.LayerNorm(EMBEDDING_DIM)
            self.fused_emb_ln = nn.LayerNorm(EMBEDDING_DIM)

            # Embedding-level fusion head
            if fusion_method == "gating":
                self.fusion = EmbeddingGatingFusion(EMBEDDING_DIM)
            elif fusion_method == "film":
                self.fusion = EmbeddingFiLMFusion(EMBEDDING_DIM)
            elif fusion_method == "concat":
                self.fusion = EmbeddingConcatFusion(EMBEDDING_DIM)
            elif fusion_method == "cross_attention":
                raise ValueError(
                    "cross_attention has been removed for Mode 3. "
                    "Use one of: gating, film, concat."
                )
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
       

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
            speaker_embedding = self.ptm_encoder(embedding)  # (B, EMBEDDING_DIM)

        elif self.mode == 2:
            feature = kwargs["feature"] # (B, C_hc, T)
            # Handcrafted encoder
            hc_feat = self.handcrafted_encoder(feature)  # (B, 768, T)
            # Backbone
            speaker_embedding = self.backbone(hc_feat)  # (B, EMBEDDING_DIM)

        elif self.mode == 3:
            embedding = kwargs["embedding"] 
            feature = kwargs["feature"]     
            
            # Encode PTM to embedding directly
            ptm_emb = self.ptm_encoder(embedding)        # (B, EMBEDDING_DIM)

            # HC -> ECAPA
            hc_feat = self.handcrafted_encoder(feature)  # (B, PTM_DIM, T)
            hc_emb = self.hc_backbone(hc_feat)           # (B, EMBEDDING_DIM)

            ptm_emb = self.ptm_emb_ln(ptm_emb)
            hc_emb = self.hc_emb_ln(hc_emb)

            # Fuse at embedding level
            if self.fusion_method == "gating":
                fused_emb, gate_weights = self.fusion(ptm_emb, hc_emb, return_gate=True)
            else:
                fused_emb = self.fusion(ptm_emb, hc_emb)

            fused_emb = self.fused_emb_ln(fused_emb)
            speaker_embedding = fused_emb

        speaker_embedding = F.normalize(speaker_embedding, p=2, dim=1)

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
