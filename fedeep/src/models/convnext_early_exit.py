"""
ConvNeXt-Tiny with 4 Early Exit Heads for FedEEP

Architecture:
  ConvNeXt-Tiny backbone (timm, pretrained on ImageNet-12k)
  4 stages: 96 -> 192 -> 384 -> 768 channels
  4 exit heads: one per stage (GAP -> LayerNorm -> Linear)

Forward modes:
  forward_all_exits(x)     -> (y1, y2, y3, y4) for training
  forward(x, threshold)    -> (logits, exit_idx) for inference

CMS regularization is handled externally in the trainer as a loss term.
"""

import logging
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("fedeep.models")

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    logger.error("timm is required: pip install timm>=0.9.0")


class ExitHead(nn.Module):
    """
    Lightweight classifier head for intermediate exits.

    Architecture: Global Average Pool -> LayerNorm -> Linear(num_classes)

    Uses LayerNorm (not BatchNorm) because ConvNeXt uses LN throughout
    and small FL batch sizes make BN statistics unstable.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.norm = nn.LayerNorm(in_channels)
        self.linear = nn.Linear(in_channels, num_classes)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> logits: (B, num_classes)"""
        x = self.flatten(self.gap(x))
        x = self.norm(x)
        return self.linear(x)

    def confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """Confidence = max of softmax. Shape: (B,)"""
        return F.softmax(logits, dim=-1).max(dim=-1).values


class ConvNeXtEarlyExit(nn.Module):
    """
    ConvNeXt-Tiny pretrained backbone with 4 early exit heads.

    Stage channel dims: [96, 192, 384, 768]

    Exit positions:
      Exit 1 -> after Stage 1  (96ch)
      Exit 2 -> after Stage 2  (192ch)
      Exit 3 -> after Stage 3  (384ch)
      Exit 4 -> after Stage 4  (768ch, final exit)

    Args:
        num_classes: Number of output classes
        model_name:  timm model identifier
        pretrained:  Load ImageNet pretrained weights
    """

    STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(
        self,
        num_classes: int = 100,
        model_name: str = "convnext_tiny.in12k",
        pretrained: bool = True,
    ):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm is required: pip install timm>=0.9.0")

        self.num_classes = num_classes

        logger.info(f"Loading {model_name} (pretrained={pretrained})...")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        feat_info = self.backbone.feature_info.channels()
        logger.info(f"Backbone stage channels: {feat_info}")
        assert feat_info == self.STAGE_CHANNELS, (
            f"Expected channels {self.STAGE_CHANNELS}, got {feat_info}. "
            f"Check model_name='{model_name}'."
        )

        self._fix_stem_stride()

        C = self.STAGE_CHANNELS
        self.exit1 = ExitHead(C[0], num_classes)
        self.exit2 = ExitHead(C[1], num_classes)
        self.exit3 = ExitHead(C[2], num_classes)
        self.exit4 = ExitHead(C[3], num_classes)

        total = sum(p.numel() for p in self.parameters())
        backbone = sum(p.numel() for p in self.backbone.parameters())
        logger.info(
            f"ConvNeXtEarlyExit: {total:,} params total "
            f"({backbone:,} backbone + {total-backbone:,} exit heads)"
        )

    def _fix_stem_stride(self):
        """
        Reduce ConvNeXt stem stride from 4 to 1 for CIFAR-sized inputs (32x32).

        ConvNeXt stem = Conv2d(3, 96, k=4, s=4) + LayerNorm.
        Stride 4->1 so 32x32 stays 32x32 after stem.
        """
        fixed = False
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.stride == (4, 4):
                original_stride = module.stride
                module.stride = (1, 1)
                module.padding = (
                    module.kernel_size[0] // 2,
                    module.kernel_size[1] // 2,
                )
                logger.info(
                    f"CIFAR fix: {name} stride {original_stride} -> (1,1), "
                    f"padding adjusted"
                )
                fixed = True
                break
        if not fixed:
            logger.warning(
                "Could not find stem Conv2d with stride=4. "
                "Spatial dims may be too small for CIFAR."
            )

    # ── Forward modes ─────────────────────────────────────────────────────────

    def forward_all_exits(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward: all 4 stages, return all 4 exit logits.

        Returns:
            (y1, y2, y3, y4): Logits from each exit, each (B, num_classes)
        """
        features = self.backbone(x)
        f1, f2, f3, f4 = features
        return self.exit1(f1), self.exit2(f2), self.exit3(f3), self.exit4(f4)

    def forward(
        self,
        x: torch.Tensor,
        threshold: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference forward with confidence-based early exit.

        Exits at first stage where max(softmax(logits)) >= threshold.
        Exit 4 always used as fallback.

        Returns:
            logits:       Final predictions (B, num_classes)
            exit_indices: Which exit per sample (B,), values 0..3
        """
        B = x.size(0)
        device = x.device

        final_logits = torch.zeros(B, self.num_classes, device=device)
        exit_indices = torch.full((B,), 3, dtype=torch.long, device=device)
        remaining = torch.ones(B, dtype=torch.bool, device=device)

        features = self.backbone(x)

        for k, (feat, exit_head) in enumerate(
            zip(features, [self.exit1, self.exit2, self.exit3, self.exit4])
        ):
            logits_k = exit_head(feat)
            conf_k = exit_head.confidence(logits_k)

            if k < 3:
                exit_mask = remaining & (conf_k >= threshold)
                final_logits[exit_mask] = logits_k[exit_mask]
                exit_indices[exit_mask] = k
                remaining &= ~exit_mask
                if not remaining.any():
                    break
            else:
                final_logits[remaining] = logits_k[remaining]
                exit_indices[remaining] = 3

        return final_logits, exit_indices

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        """Parameter count by component."""
        counts = {
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
        }
        for k, head in enumerate(
            [self.exit1, self.exit2, self.exit3, self.exit4], 1
        ):
            counts[f"exit{k}"] = sum(p.numel() for p in head.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def get_exit_param_groups(self) -> List[nn.Parameter]:
        """All exit head parameters (Fast Weights)."""
        params = []
        for head in [self.exit1, self.exit2, self.exit3, self.exit4]:
            params.extend(head.parameters())
        return params

    def get_backbone_param_groups(self) -> List[nn.Parameter]:
        """All backbone parameters (Slow Weights)."""
        return list(self.backbone.parameters())
