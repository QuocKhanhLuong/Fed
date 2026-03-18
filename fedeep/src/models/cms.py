"""
Continuum Memory System (CMS) — regularizer for backbone (Slow) parameters.

Maintains M EMA memory buffers of backbone weights, each with a different
decay rate. Encourages the backbone to stay close to historical snapshots,
preventing catastrophic forgetting when receiving a new global model.

Formula:
    L_CMS = (mu / M) * sum_m  w_m * ||theta - theta^(m)||^2 / sqrt(N)

where theta^(m) = EMA buffer m, w_m = level weight (heavier on older levels).

Zero communication overhead: buffers are stored only on the client,
never sent to the server.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("fedeep.cms")


class ContinuumMemorySystem:
    """
    CMS regularizer for backbone (Slow) parameters.

    Args:
        decay_rates: EMA decay per buffer. Default: [0.90, 0.99, 0.999, 0.9999]
        loss_weight: Overall scale mu for the CMS loss term
    """

    def __init__(
        self,
        decay_rates: List[float] = None,
        loss_weight: float = 0.1,
    ):
        self.decay_rates = decay_rates or [0.90, 0.99, 0.999, 0.9999]
        self.loss_weight = loss_weight
        self.num_buffers = len(self.decay_rates)
        self.buffers: List[Optional[torch.Tensor]] = [None] * self.num_buffers
        self._initialized = False

    def _get_flat_backbone(self, slow_params: List[nn.Parameter]) -> torch.Tensor:
        """Flatten all slow parameters into a single 1D tensor."""
        return torch.cat([p.detach().view(-1) for p in slow_params])

    def _ensure_initialized(self, slow_params: List[nn.Parameter]):
        if self._initialized:
            return
        init_weights = self._get_flat_backbone(slow_params)
        for i in range(self.num_buffers):
            self.buffers[i] = init_weights.clone()
        self._initialized = True
        logger.info(
            f"CMS initialized: {init_weights.numel():,} backbone params x "
            f"{self.num_buffers} buffers, decay={self.decay_rates}"
        )

    @torch.no_grad()
    def update(self, slow_params: List[nn.Parameter]):
        """
        Update all EMA buffers with current backbone weights.
        Call after each slow optimizer step.
        """
        self._ensure_initialized(slow_params)
        current = self._get_flat_backbone(slow_params)
        for i, decay in enumerate(self.decay_rates):
            self.buffers[i] = decay * self.buffers[i] + (1.0 - decay) * current

    def compute_loss(self, slow_params: List[nn.Parameter]) -> torch.Tensor:
        """
        CMS regularization loss: weighted L2 distance to memory buffers.

        Uses normalized L2 (||theta - theta^m||^2 / sqrt(N)) instead of
        mean-reduced MSE to avoid vanishing loss when N is very large (~28M).

        Returns scalar loss term ready to add to total_loss.
        """
        if not self._initialized:
            self._ensure_initialized(slow_params)
            device = slow_params[0].device
            return torch.tensor(0.0, device=device)

        device = slow_params[0].device
        current = torch.cat([p.view(-1) for p in slow_params])  # differentiable
        num_params = current.numel()
        norm_factor = num_params ** 0.5

        loss = torch.tensor(0.0, device=device)
        for i, buf in enumerate(self.buffers):
            if buf is None:
                continue
            buf_dev = buf.to(device)
            level_weight = (i + 1) / self.num_buffers
            diff_sq = (current - buf_dev).pow(2).sum() / norm_factor
            loss = loss + level_weight * diff_sq

        return self.loss_weight * loss / self.num_buffers

    def get_stats(self) -> Dict[str, float]:
        if not self._initialized:
            return {}
        return {
            f"cms_buf{i}_norm": buf.norm().item()
            for i, buf in enumerate(self.buffers)
            if buf is not None
        }
