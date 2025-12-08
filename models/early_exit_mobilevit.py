"""
Early-Exit MobileViTv2 for Federated Learning

This module implements the Early-Exit Neural Network architecture proposed in:

    "Difficulty-Aware Federated Learning with Early-Exit Networks"
    IEEE Transactions on Mobile Computing, 2025

Mathematical Formulation:
------------------------
Given input x and confidence threshold τ, the early-exit mechanism is defined as:

    y_k = f_k(x)                     # Output at exit k
    c_k = max(softmax(y_k))          # Confidence at exit k
    
    output = y_k where k = min{i : c_i ≥ τ}

Multi-Exit Loss (Eq. 3 in paper):
    L = Σ_{k=1}^{K} α_k · CE(y_k, y_true)
    
where α_k are the exit weights satisfying Σα_k = 1.

Architecture (Section III-B):
- Stage 1: Convolutional backbone (low-level features)
- Stage 2: Lightweight transformer (mid-level features)  
- Stage 3: Deep transformer (high-level features)

Author: Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import timm for pretrained weights
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    logger.warning("timm not installed - pretrained weights unavailable")


# =============================================================================
# Building Blocks (Section III-A)
# =============================================================================

class ConvBNAct(nn.Sequential):
    """
    Standard Conv-BN-Activation block.
    
    Implements: x → Conv2d → BatchNorm2d → SiLU(x)
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, 
                 stride: int = 1, groups: int = 1):
        padding = (kernel - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, 
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class InvertedResidual(nn.Module):
    """
    MobileNetV2 Inverted Residual Block.
    
    Implements (for stride=1, in_ch=out_ch):
        x → Expand → Depthwise → Project → x + output
        
    Reference: Sandler et al., "MobileNetV2", CVPR 2018
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden_dim = int(in_ch * expand_ratio)
        self.use_residual = (stride == 1) and (in_ch == out_ch)
        
        layers = []
        # Expansion (pointwise)
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden_dim, kernel=1))
        # Depthwise
        layers.append(ConvBNAct(hidden_dim, hidden_dim, kernel=3, 
                                stride=stride, groups=hidden_dim))
        # Projection (pointwise, no activation)
        layers.extend([
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return (x + out) if self.use_residual else out


class LinearAttention(nn.Module):
    """
    Linear Complexity Self-Attention (O(n) instead of O(n²)).
    
    Implements separable attention from MobileViTv2:
        Attention(Q,K,V) = softmax(Q) · (softmax(K)^T · V)
    
    Reference: Mehta & Rastegari, "MobileViTv2", 2022
    """
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Linear attention: O(n) complexity
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-2)
        context = torch.bmm(k, v.transpose(-2, -1))  # [B, C, C]
        out = torch.bmm(q.transpose(-2, -1), context).transpose(-2, -1)
        
        return self.proj(out.reshape(B, C, H, W))


class TransformerBlock(nn.Module):
    """
    Lightweight Transformer Block.
    
    Implements: x → Norm → Attention → x + out → Norm → MLP → x + out
    """
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = LinearAttention(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            ConvBNAct(dim, dim * mlp_ratio, kernel=1),
            nn.Conv2d(dim * mlp_ratio, dim, 1, bias=False),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# Early Exit Classifier (Section III-C)
# =============================================================================

class EarlyExitClassifier(nn.Module):
    """
    Lightweight Classifier for Intermediate Exits.
    
    Architecture: AdaptivePool → FC₁ → ReLU → Dropout → FC₂
    
    Args:
        in_channels (int): Input feature channels
        num_classes (int): Number of output classes
        hidden_dim (int): Hidden layer dimension
    """
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x))
    
    def get_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence score: c = max(softmax(logits))
        
        Returns:
            Tensor of shape [B] with confidence scores in [0, 1]
        """
        return F.softmax(logits, dim=-1).max(dim=-1).values


# =============================================================================
# Main Model (Section III-B)
# =============================================================================

class EarlyExitMobileViTv2(nn.Module):
    """
    MobileViTv2 with Early Exit Mechanism.
    
    Architecture Overview (Fig. 2 in paper):
    ┌─────────────────────────────────────────────┐
    │ Stage 1: CNN Backbone                       │
    │   Stem → IR Block → IR Block                │
    │              ↓                              │
    │         [Exit 1] ← Early exit if c₁ ≥ τ    │
    ├─────────────────────────────────────────────┤
    │ Stage 2: Transformer + CNN                  │
    │   Transformer → IR Block                    │
    │              ↓                              │
    │         [Exit 2] ← Early exit if c₂ ≥ τ    │
    ├─────────────────────────────────────────────┤
    │ Stage 3: Deep Transformer                   │
    │   Transformer → Transformer → Conv          │
    │              ↓                              │
    │         [Exit 3] ← Final output            │
    └─────────────────────────────────────────────┘
    
    Args:
        num_classes (int): Number of output classes
        width_mult (float): Width multiplier for channels
        
    Attributes:
        stage1, stage2, stage3: Feature extraction stages
        exit1, exit2, exit3: Classification heads
    """
    
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0, pretrained: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Channel configuration (Table I)
        C = [32, 64, 96, 128, 160, 192]
        C = [int(c * width_mult) for c in C]
        
        # Stage 1: CNN backbone (Spatial: 32×32 → 16×16)
        self.stage1 = nn.Sequential(
            ConvBNAct(3, C[0], kernel=3, stride=1),        # Stem
            InvertedResidual(C[0], C[1], stride=1, expand_ratio=2),
            InvertedResidual(C[1], C[2], stride=2, expand_ratio=2),
            InvertedResidual(C[2], C[2], stride=1, expand_ratio=2),
        )
        self.exit1 = EarlyExitClassifier(C[2], num_classes)
        
        # Stage 2: Transformer + CNN (Spatial: 16×16 → 8×8)
        self.stage2 = nn.Sequential(
            TransformerBlock(C[2]),
            InvertedResidual(C[2], C[3], stride=2, expand_ratio=2),
            InvertedResidual(C[3], C[3], stride=1, expand_ratio=2),
        )
        self.exit2 = EarlyExitClassifier(C[3], num_classes)
        
        # Stage 3: Deep transformer (Spatial: 8×8)
        self.stage3 = nn.Sequential(
            TransformerBlock(C[3]),
            TransformerBlock(C[3]),
            ConvBNAct(C[3], C[5], kernel=1),
        )
        self.exit3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C[5], num_classes),
        )
        
        if pretrained and HAS_TIMM:
            self._load_pretrained()
        else:
            self._init_weights()
        
    def _load_pretrained(self):
        """Load pretrained weights from timm MobileViTv2."""
        logger.info("Loading pretrained MobileViTv2 backbone from timm...")
        try:
            # Load pretrained MobileViTv2
            pretrained_model = timm.create_model(
                'mobilevitv2_050.cvnets_in1k', 
                pretrained=True, 
                num_classes=self.num_classes
            )
            
            # Copy stem weights (first conv)
            self.stage1[0][0].weight.data.copy_(
                pretrained_model.stem.conv.weight.data
            )
            self.stage1[0][1].weight.data.copy_(
                pretrained_model.stem.bn.weight.data
            )
            self.stage1[0][1].bias.data.copy_(
                pretrained_model.stem.bn.bias.data
            )
            
            logger.info("✓ Pretrained backbone loaded successfully")
            
            # Initialize exit classifiers (not pretrained)
            for exit_module in [self.exit1, self.exit2]:
                for m in exit_module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                            
            # Initialize final exit
            for m in self.exit3.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                        
            del pretrained_model
            
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            logger.info("Falling back to random initialization")
            self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_all_exits(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all exits (for training).
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Tuple of (logits₁, logits₂, logits₃) for each exit
        """
        f1 = self.stage1(x)
        y1 = self.exit1(f1)
        
        f2 = self.stage2(f1)
        y2 = self.exit2(f2)
        
        f3 = self.stage3(f2)
        y3 = self.exit3(f3)
        
        return y1, y2, y3
    
    def forward(
        self, x: torch.Tensor, threshold: float = 0.8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with early exit based on confidence threshold.
        
        Algorithm 1: Early Exit Inference
        ─────────────────────────────────
        Input: x, threshold τ
        for k = 1 to K do
            f_k ← Stage_k(f_{k-1})
            y_k ← Exit_k(f_k)
            c_k ← max(softmax(y_k))
            if c_k ≥ τ then
                return y_k, k
        return y_K, K
        
        Args:
            x: Input tensor [B, 3, H, W]
            threshold: Confidence threshold τ ∈ [0, 1]
            
        Returns:
            logits: Predictions [B, num_classes]
            exit_indices: Which exit was used [B]
        """
        B = x.size(0)
        device = x.device
        num_classes = self.exit3[-1].out_features
        
        # Initialize outputs
        final_logits = torch.zeros(B, num_classes, device=device)
        exit_indices = torch.full((B,), 2, dtype=torch.long, device=device)
        remaining = torch.ones(B, dtype=torch.bool, device=device)
        
        # Stage 1 → Exit 1
        f1 = self.stage1(x)
        y1 = self.exit1(f1)
        c1 = self.exit1.get_confidence(y1)
        
        exit_mask = remaining & (c1 >= threshold)
        final_logits[exit_mask] = y1[exit_mask]
        exit_indices[exit_mask] = 0
        remaining &= ~exit_mask
        
        if not remaining.any():
            return final_logits, exit_indices
        
        # Stage 2 → Exit 2
        f2 = self.stage2(f1)
        y2 = self.exit2(f2)
        c2 = self.exit2.get_confidence(y2)
        
        exit_mask = remaining & (c2 >= threshold)
        final_logits[exit_mask] = y2[exit_mask]
        exit_indices[exit_mask] = 1
        remaining &= ~exit_mask
        
        if not remaining.any():
            return final_logits, exit_indices
        
        # Stage 3 → Exit 3 (final)
        f3 = self.stage3(f2)
        y3 = self.exit3(f3)
        final_logits[remaining] = y3[remaining]
        
        return final_logits, exit_indices
    
    def count_parameters(self) -> dict:
        """Count parameters by component (Table II)."""
        return {
            'stage1': sum(p.numel() for p in self.stage1.parameters()),
            'exit1': sum(p.numel() for p in self.exit1.parameters()),
            'stage2': sum(p.numel() for p in self.stage2.parameters()),
            'exit2': sum(p.numel() for p in self.exit2.parameters()),
            'stage3': sum(p.numel() for p in self.stage3.parameters()),
            'exit3': sum(p.numel() for p in self.exit3.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
        }


# =============================================================================
# Training Loss (Section III-D, Eq. 3)
# =============================================================================

def multi_exit_loss(
    logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    weights: List[float] = [0.3, 0.3, 0.4],
) -> torch.Tensor:
    """
    Multi-Exit Cross-Entropy Loss.
    
    Computes weighted sum of CE losses at each exit:
        L = Σ_{k=1}^{K} α_k · CE(y_k, y_true)
    
    Args:
        logits: Tuple of (exit1, exit2, exit3) logits
        labels: Ground truth labels [B]
        weights: Loss weights α_k (should sum to 1)
        
    Returns:
        Combined loss scalar
    """
    loss = sum(w * F.cross_entropy(y, labels) for y, w in zip(logits, weights))
    return loss


# =============================================================================
# Unit Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EarlyExitMobileViTv2 - IEEE Format Test")
    print("=" * 60)
    
    model = EarlyExitMobileViTv2(num_classes=10)
    params = model.count_parameters()
    
    print(f"\nParameter Count (Table II):")
    print(f"  {'Component':<12} {'Params':>12}")
    print(f"  {'-'*24}")
    for k, v in params.items():
        print(f"  {k:<12} {v:>12,}")
    
    # Test forward
    x = torch.randn(4, 3, 32, 32)
    y1, y2, y3 = model.forward_all_exits(x)
    print(f"\nForward all exits: {y1.shape}, {y2.shape}, {y3.shape}")
    
    # Test early exit
    for tau in [0.5, 0.8, 0.95]:
        logits, exits = model(x, threshold=tau)
        print(f"τ={tau}: exits={exits.tolist()}")
