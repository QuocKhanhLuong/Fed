import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GatedBlockWrapper(nn.Module):
    def __init__(self, original_block: nn.Module):
        super().__init__()
        self.original_block = original_block
        self.in_channels, self.out_channels = self._infer_channels(original_block)
        
        self.expert = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, 
                      padding=1, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False)
        )
        
        self.gate = nn.Linear(1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        
        self.quality_score = 1.0
        
    def _infer_channels(self, block: nn.Module) -> tuple[int, int]:
        if hasattr(block, 'conv_kxk') and hasattr(block.conv_kxk, 'conv'):
            in_ch = block.conv_kxk.conv.in_channels
            out_ch = in_ch
            for m in reversed(list(block.modules())):
                if isinstance(m, nn.Conv2d):
                    out_ch = m.out_channels
                    break
            return in_ch, out_ch
            
        convs = [m for m in block.modules() if isinstance(m, nn.Conv2d)]
        if len(convs) > 0:
            return convs[0].in_channels, convs[-1].out_channels
        
        raise ValueError(f"Could not infer channels for block: {block}")

    def set_quality_score(self, score: float):
        self.quality_score = score
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_orig = self.original_block(x)
        out_expert = self.expert(x)
        
        if out_orig.shape[2:] != out_expert.shape[2:]:
            out_expert = F.adaptive_avg_pool2d(out_expert, out_orig.shape[2:])
        
        if out_orig.shape[1] != out_expert.shape[1]:
            raise RuntimeError(f"Channel mismatch: {out_orig.shape[1]} vs {out_expert.shape[1]}")
        
        score_tensor = torch.tensor([self.quality_score], dtype=x.dtype, device=x.device)
        lambda_val = torch.sigmoid(self.gate(score_tensor)).view(1, 1, 1, 1)
            
        return out_orig + lambda_val * out_expert
