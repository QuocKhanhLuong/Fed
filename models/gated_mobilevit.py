import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GatedBlockWrapper(nn.Module):
    """
    Wrapper for MobileViT blocks that adds a network-adaptive expert branch.
    
    The expert branch consists of a lightweight CNN (Depthwise -> Pointwise)
    that is gated by a quality score.
    """
    def __init__(self, original_block: nn.Module):
        super().__init__()
        self.original_block = original_block
        
        # Determine input channels from the original block
        # We attempt to find the first Conv2d layer to infer channels
        self.in_channels = self._infer_channels(original_block)
        
        # Expert Branch: Lightweight CNN
        # DepthwiseConv3x3 -> BatchNorm -> SiLU -> PointwiseConv
        self.expert = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, 
                      padding=1, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False)
        )
        
        # Gating Mechanism
        # Maps quality_score (scalar) -> lambda (scalar in [0, 1])
        self.gate = nn.Linear(1, 1)
        
        # Initialize gate to output 0.0 (sigmoid(0) = 0.5) or bias it towards 0?
        # We'll initialize weights to 0 so it starts neutral or we can bias it.
        # Let's stick to default initialization for now, or maybe zero init for less disruption.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        
        self.quality_score = 1.0
        
    def _infer_channels(self, block: nn.Module) -> int:
        # Try specific attributes for MobileVitV2Block
        if hasattr(block, 'conv_kxk') and hasattr(block.conv_kxk, 'conv'):
            return block.conv_kxk.conv.in_channels
            
        # Fallback: find first Conv2d
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                return m.in_channels
        
        raise ValueError(f"Could not infer input channels for block: {block}")

    def set_quality_score(self, score: float):
        self.quality_score = score
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original block forward pass
        out_orig = self.original_block(x)
        
        # Expert branch forward pass
        out_expert = self.expert(x)
        
        # Calculate gating weight lambda
        # Input to linear layer must be tensor
        score_tensor = torch.tensor([self.quality_score], dtype=x.dtype, device=x.device)
        gate_logits = self.gate(score_tensor)
        lambda_val = torch.sigmoid(gate_logits)
        
        # Reshape for broadcasting: [1, 1, 1, 1]
        lambda_val = lambda_val.view(1, 1, 1, 1)
        
        # Combine
        # Check for shape mismatch (e.g. if original block changed channels/resolution)
        if out_orig.shape != out_expert.shape:
            # If shapes don't match, we might need to project out_expert
            # For MobileViT V2 blocks, they typically preserve shape.
            # If mismatch occurs, we log a warning or try to handle it?
            # For now, we assume they match. If not, this will raise a runtime error.
            pass
            
        return out_orig + lambda_val * out_expert
