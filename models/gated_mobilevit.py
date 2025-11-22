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
        
        # Determine input/output channels from the original block
        # We need to match the OUTPUT channels of the block for proper addition
        self.in_channels, self.out_channels = self._infer_channels(original_block)
        
        # Expert Branch: Lightweight CNN
        # Takes input with in_channels, outputs out_channels to match original block
        # DepthwiseConv3x3 -> BatchNorm -> SiLU -> PointwiseConv
        self.expert = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, 
                      padding=1, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        )
        
        # Gating Mechanism
        # Maps quality_score (scalar) -> lambda (scalar in [0, 1])
        self.gate = nn.Linear(1, 1)
        
        # Initialize gate to near zero so expert branch has minimal impact initially
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)  # sigmoid(-2) ≈ 0.12
        
        self.quality_score = 1.0
        
    def _infer_channels(self, block: nn.Module) -> tuple[int, int]:
        """Infer input and output channels from the block."""
        # Try specific attributes for MobileVitV2Block
        if hasattr(block, 'conv_kxk') and hasattr(block.conv_kxk, 'conv'):
            in_ch = block.conv_kxk.conv.in_channels
            # Find output channels from last conv layer
            out_ch = in_ch  # default to same
            for m in reversed(list(block.modules())):
                if isinstance(m, nn.Conv2d):
                    out_ch = m.out_channels
                    break
            return in_ch, out_ch
            
        # Fallback: find first and last Conv2d
        convs = [m for m in block.modules() if isinstance(m, nn.Conv2d)]
        if len(convs) > 0:
            in_ch = convs[0].in_channels
            out_ch = convs[-1].out_channels
            return in_ch, out_ch
        
        raise ValueError(f"Could not infer channels for block: {block}")

    def set_quality_score(self, score: float):
        self.quality_score = score
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original block forward pass
        out_orig = self.original_block(x)
        
        # Expert branch forward pass  
        out_expert = self.expert(x)
        
        # Handle spatial dimension mismatch (e.g., if block does downsampling)
        if out_orig.shape[2:] != out_expert.shape[2:]:
            # Resize expert output to match original output spatial dimensions
            out_expert = F.adaptive_avg_pool2d(out_expert, out_orig.shape[2:])
        
        # Handle channel mismatch (shouldn't happen with proper initialization, but safety check)
        if out_orig.shape[1] != out_expert.shape[1]:
            # This is a critical error - expert should output same channels
            raise RuntimeError(
                f"Channel mismatch: orig={out_orig.shape[1]}, expert={out_expert.shape[1]}. "
                f"Check _infer_channels implementation."
            )
        
        # Calculate gating weight lambda
        # Input to linear layer must be tensor
        score_tensor = torch.tensor([self.quality_score], dtype=x.dtype, device=x.device)
        gate_logits = self.gate(score_tensor)
        lambda_val = torch.sigmoid(gate_logits)
        
        # Reshape for broadcasting: [1, 1, 1, 1]
        lambda_val = lambda_val.view(1, 1, 1, 1)
            
        return out_orig + lambda_val * out_expert
