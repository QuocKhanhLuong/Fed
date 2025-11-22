import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from typing import Callable

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
            activation_layer(),
        )

class InvertedResidual(nn.Module):
    """Standard MobileNetV2 Block"""
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNActivation(inp, hidden_dim, kernel_size=1))
        
        layers.extend([
            ConvBNActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class SeparableSelfAttention(nn.Module):
    """
    Linear Complexity Self-Attention for MobileViTv2.
    Complexity: O(k) instead of O(k^2)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.project_i = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False) # Input
        self.project_k = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False) # Key
        self.project_v = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False) # Value
        self.project_out = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        
        # 1. Projections
        I = self.project_i(x)
        K = self.project_k(x)
        V = self.project_v(x)
        
        # 2. Compute Context Score (Local Context)
        # Apply softmax over spatial dimensions to get attention distribution
        B, C, H, W = I.shape
        I_flat = I.view(B, C, -1) # [B, C, N]
        context_scores = F.softmax(I_flat, dim=2).view(B, C, H, W) # [B, C, H, W]
        
        # 3. Compute Context Vector (Global Context)
        # Weighted sum of Keys based on Context Scores
        # Note: This is element-wise multiplication followed by sum, keeping linear complexity
        context_vector = (K * context_scores).sum(dim=[2, 3], keepdim=True) # [B, C, 1, 1]
        
        # 4. Update Values with Global Context
        # Broadcast context_vector to all spatial locations
        out = V * context_vector # [B, C, H, W]
        
        # 5. Final Projection
        out = self.project_out(out)
        
        return out

class GatedMobileViTBlock(nn.Module):
    """
    Network-Adaptive MobileViT Block.
    Fuses Separable Attention (Global) with a lightweight Expert (Local).
    """
    def __init__(self, in_channels: int, transformer_dim: int, ffn_dim: int):
        super().__init__()
        
        # Local Representation (CNN)
        self.local_rep = nn.Sequential(
            ConvBNActivation(in_channels, in_channels, kernel_size=3, groups=in_channels),
            nn.Conv2d(in_channels, transformer_dim, 1, bias=False)
        )
        
        # Global Branch: Separable Attention
        self.global_att = SeparableSelfAttention(transformer_dim)
        self.ffn = nn.Sequential(
            ConvBNActivation(transformer_dim, ffn_dim, kernel_size=1),
            nn.Conv2d(ffn_dim, transformer_dim, 1, bias=False)
        )
        
        # Local Expert Branch (Depthwise Conv) - Activates when network is poor
        self.expert = nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim, 3, padding=1, groups=transformer_dim, bias=False),
            nn.BatchNorm2d(transformer_dim),
            nn.SiLU()
        )
        
        # Fusion Projection
        self.proj_back = nn.Sequential(
            ConvBNActivation(transformer_dim, in_channels, kernel_size=1),
            ConvBNActivation(in_channels, in_channels, kernel_size=3, groups=in_channels)
        )
        
        # Gating Mechanism
        # Takes (1 - quality_score) and scales the expert output
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.5)
        self.quality_score = 1.0

    def set_quality_score(self, score: float):
        self.quality_score = score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        
        # Local Rep
        feat = self.local_rep(x)
        
        # Branch 1: Global Attention
        att_out = self.global_att(feat)
        att_out = self.ffn(att_out)
        
        # Branch 2: Local Expert
        # Logic: If quality is low (score -> 0), expert contribution increases
        expert_out = self.expert(feat)
        
        # Gating Logic
        # If score=1.0 (Good) -> gate=0 -> Only Attention
        # If score=0.0 (Bad) -> gate=scale -> Attention + Expert
        gate_val = self.gate_scale * (1.0 - self.quality_score)
        
        combined = att_out + gate_val * expert_out
        
        # Fusion
        out = self.proj_back(combined)
        
        return res + out

class MobileViTv2(nn.Module):
    """
    Custom MobileViTv2 for CIFAR-10 / MedMNIST (32x32 input).
    """
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        
        # Architecture Configuration for 32x32 images
        # Channels: [Stem, Layer1, Layer2, Layer3, Layer4, Layer5]
        channels = [32, 64, 96, 128, 160, 192]
        channels = [int(c * width_mult) for c in channels]
        
        self.stem = ConvBNActivation(3, channels[0], stride=1)
        
        # Layer 1: MV2 (32x32)
        self.layer1 = nn.Sequential(
            InvertedResidual(channels[0], channels[1], 1, 2)
        )
        
        # Layer 2: MV2 (Downsample -> 16x16)
        self.layer2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], 2, 2),
            InvertedResidual(channels[2], channels[2], 1, 2)
        )
        
        # Layer 3: Gated Transformer (16x16)
        self.layer3 = GatedMobileViTBlock(channels[2], channels[2]*2, channels[2]*4)
        
        # Layer 4: MV2 (Downsample -> 8x8)
        self.layer4 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], 2, 2),
            InvertedResidual(channels[3], channels[3], 1, 2)
        )
        
        # Layer 5: Gated Transformer (8x8)
        self.layer5 = nn.Sequential(
            GatedMobileViTBlock(channels[3], channels[3]*2, channels[3]*4),
            GatedMobileViTBlock(channels[3], channels[3]*2, channels[3]*4)
        )
        
        # Classifier
        self.conv_1x1_exp = ConvBNActivation(channels[3], channels[5], kernel_size=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[5], num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def set_quality_score(self, score: float):
        for m in self.modules():
            if isinstance(m, GatedMobileViTBlock):
                m.set_quality_score(score)

    def forward(self, x: torch.Tensor, quality_score: Optional[float] = None) -> torch.Tensor:
        if quality_score is not None:
            self.set_quality_score(quality_score)
            
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.conv_1x1_exp(x)
        x = self.classifier(x)
        return x
