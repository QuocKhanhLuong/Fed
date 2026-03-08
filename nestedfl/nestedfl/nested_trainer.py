"""
FedEEP Nested Trainer

Local training logic for Federated Early-Exit with Progressive phases.

Components (activated by phase):
  Phase 0  — Backbone warmup: only Exit4 CE, single optimizer, no fast/slow
  Phase 1  — Multi-exit CE: all 4 exits supervised, still single optimizer
  Phase 2  — Fast/Slow + CMS: fast heads + slow backbone, CMS regularizer
  Phase 3  — + Self-Distillation KD chain (E1←E2←E3←E4)
  Phase 4  — Full system with all components active

CMS (ContinuumMemorySystem):
  - External to model — NOT part of state_dict sent to server
  - 4 EMA memory buffers of backbone weights at different decay rates
  - compute_loss() = weighted MSE between current backbone and memory buffers
  - update() called after each slow optimizer step

Fast/Slow split:
  - Fast params: 4 exit heads (update every step, high lr)
  - Slow params: ConvNeXt backbone (update every K steps, low lr)
"""

import logging
import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Continuum Memory System (CMS)
# ─────────────────────────────────────────────────────────────────────────────

class ContinuumMemorySystem:
    """
    CMS regularizer for backbone (Slow) parameters.

    Maintains M EMA memory buffers of backbone weights, each with a different
    decay rate. Encourages the backbone to stay close to historical snapshots,
    preventing catastrophic forgetting when receiving a new global model.

    Formula:
        L_CMS = (μ / M) * Σ_m w_m * ||θ - θ^(m)||^2

    where θ^(m) = EMA buffer m, w_m = level weight (heavier on older levels).

    Zero communication overhead: buffers are stored only on the client,
    never sent to the server.

    Args:
        decay_rates: EMA decay per buffer. Default: [0.90, 0.99, 0.999, 0.9999]
        loss_weight: Overall scale μ for the CMS loss term
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
            f"CMS initialized: {init_weights.numel():,} backbone params × "
            f"{self.num_buffers} buffers, decay={self.decay_rates}"
        )

    @torch.no_grad()
    def update(self, slow_params: List[nn.Parameter]):
        """
        Update all EMA buffers with current backbone weights.
        Call this after each slow optimizer step.
        """
        self._ensure_initialized(slow_params)
        current = self._get_flat_backbone(slow_params)
        for i, decay in enumerate(self.decay_rates):
            self.buffers[i] = decay * self.buffers[i] + (1.0 - decay) * current

    def compute_loss(self, slow_params: List[nn.Parameter]) -> torch.Tensor:
        """
        CMS regularization loss: weighted L2 distance to memory buffers.

        Uses normalized L2 (||θ - θ^m||^2 / √N) instead of mean-reduced MSE
        to avoid vanishing loss when N (param count) is very large (~28M).

        Returns scalar loss term ready to add to total_loss.
        """
        if not self._initialized:
            # First call: init and return 0 (no history yet)
            self._ensure_initialized(slow_params)
            device = slow_params[0].device
            return torch.tensor(0.0, device=device)

        device = slow_params[0].device
        current = torch.cat([p.view(-1) for p in slow_params])  # Differentiable
        num_params = current.numel()
        # Normalize by sqrt(N) — between sum (too large) and mean (too small)
        norm_factor = num_params ** 0.5

        loss = torch.tensor(0.0, device=device)
        for i, buf in enumerate(self.buffers):
            if buf is None:
                continue
            buf_dev = buf.to(device)
            # Heavier weight for older (more stable) memories
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


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class NestedEarlyExitTrainer:
    """
    FedEEP local trainer for ConvNeXtEarlyExit models.

    Key design decisions:
    - CMS lives here (not in model) → never aggregated by server
    - Phase controls which components are active (set_phase() each round)
    - Fast params (exit heads) updated every step
    - Slow params (backbone) updated every K steps with CMS regularization
    - KD chain: Exit1←Exit2←Exit3←Exit4 (deep teaches shallow)

    Usage in Flower client:
        trainer = NestedEarlyExitTrainer(model, device)
        trainer.set_phase(phase)              # set by server each round
        metrics = trainer.train(dataloader, epochs, lr)
    """

    # Loss weights α_k for each exit (sum ≈ 1)
    # Exit4 (deepest, most accurate) gets highest weight
    DEFAULT_EXIT_WEIGHTS = [0.1, 0.2, 0.3, 0.4]

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        exit_weights: List[float] = None,
        fast_lr_multiplier: float = 3.0,
        slow_update_freq: int = 5,
        # KD settings
        kd_weight: float = 0.3,
        kd_temperature: float = 4.0,
        # CMS settings
        cms_decay_rates: List[float] = None,
        cms_weight: float = 0.1,
        # AMP
        use_mixed_precision: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.exit_weights = exit_weights or self.DEFAULT_EXIT_WEIGHTS
        assert len(self.exit_weights) == 4, "Need exactly 4 exit weights"

        self.fast_lr_multiplier = fast_lr_multiplier
        self.slow_update_freq = slow_update_freq
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature

        # CMS — external to model
        self.cms = ContinuumMemorySystem(
            decay_rates=cms_decay_rates or [0.90, 0.99, 0.999, 0.9999],
            loss_weight=cms_weight,
        )

        # AMP
        self.use_amp = use_mixed_precision and (device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Parameter groups
        self.fast_params = list(model.get_exit_param_groups())
        self.slow_params = list(model.get_backbone_param_groups())

        # Phase state (set each round by the server)
        self._phase: int = 0
        self.use_multi_exit: bool = False
        self.use_fast_slow: bool = False
        self.use_cms: bool = False
        self.use_kd: bool = False

        n_fast = sum(p.numel() for p in self.fast_params)
        n_slow = sum(p.numel() for p in self.slow_params)
        logger.info(
            f"Trainer ready | fast={n_fast:,} slow={n_slow:,} | "
            f"amp={self.use_amp}"
        )

    # ── Phase control ──────────────────────────────────────────────────────────

    def set_phase(self, phase: int) -> None:
        """
        Toggle components based on current training phase.

        Phase 0: backbone warmup — only Exit4, single optimizer
        Phase 1: + multi-exit CE for all 4 exits
        Phase 2: + fast/slow split, + CMS regularization
        Phase 3: + KD chain (E1←E2←E3←E4)
        Phase 4: full system (same as 3, EDPA is server-side)
        """
        self._phase = phase
        self.use_multi_exit = phase >= 1
        self.use_fast_slow  = phase >= 2
        self.use_cms        = phase >= 2
        self.use_kd         = phase >= 3
        logger.info(
            f"Phase set to {phase}: "
            f"multi_exit={self.use_multi_exit}, "
            f"fast_slow={self.use_fast_slow}, "
            f"cms={self.use_cms}, "
            f"kd={self.use_kd}"
        )

    # ── Loss utilities ─────────────────────────────────────────────────────────

    def _multi_exit_ce(
        self,
        logits: Tuple[torch.Tensor, ...],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted cross-entropy across all active exits.

        Phase 0: only Exit4 (final exit)
        Phase 1+: all 4 exits with self.exit_weights
        """
        if not self.use_multi_exit:
            # Only final exit
            return F.cross_entropy(logits[-1], labels)

        loss = torch.tensor(0.0, device=self.device)
        for y_k, alpha_k in zip(logits, self.exit_weights):
            loss = loss + alpha_k * F.cross_entropy(y_k, labels)
        return loss

    def _kd_loss(
        self,
        logits: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        Self-distillation: chain E1←E2←E3←E4.

        Each shallower exit learns soft targets from the next deeper exit.
        Teacher gradients are stopped (detach).

        L_KD = Σ_{k=1..3} KL( student_k(T) || teacher_{k+1}(T) ) × T^2
        """
        T = self.kd_temperature
        total_kd = torch.tensor(0.0, device=self.device)
        # logits = (y1, y2, y3, y4); pairs: (y1,y2), (y2,y3), (y3,y4)
        for student, teacher in zip(logits[:-1], logits[1:]):
            teacher_soft = F.softmax(teacher.detach() / T, dim=-1)
            student_log  = F.log_softmax(student / T, dim=-1)
            kl = F.kl_div(student_log, teacher_soft, reduction="batchmean")
            total_kd = total_kd + kl * (T ** 2)
        return total_kd

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        learning_rate: float,
    ) -> Dict[str, float]:
        """
        Run E local epochs of FedEEP training.

        Returns dict of training metrics.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.train()

        # ── Optimizers ────────────────────────────────────────────────────────
        if self.use_fast_slow:
            optimizer_fast = torch.optim.AdamW(
                self.fast_params,
                lr=learning_rate * self.fast_lr_multiplier,
                weight_decay=0.01,
            )
            optimizer_slow = torch.optim.AdamW(
                self.slow_params,
                lr=learning_rate,
                weight_decay=0.01,
            )
        else:
            # Single optimizer for all params (phases 0-1)
            optimizer_all = torch.optim.AdamW(
                list(self.fast_params) + list(self.slow_params),
                lr=learning_rate,
                weight_decay=0.01,
            )

        # ── Training loop ─────────────────────────────────────────────────────
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        step_counter = 0

        for epoch in range(epochs):
            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.use_fast_slow:
                    self._step_fast_slow(
                        images, labels, step_counter,
                        optimizer_fast, optimizer_slow,
                        total_loss_ref=None,  # updated below
                    )
                    # Recompute loss for metrics (no grad needed)
                    with torch.no_grad():
                        with torch.amp.autocast('cuda', enabled=self.use_amp):
                            logits = self.model.forward_all_exits(images)
                        ce = self._multi_exit_ce(logits, labels)
                    batch_loss = ce.item()
                    preds = logits[-1].argmax(1)
                else:
                    batch_loss, preds = self._step_single(
                        images, labels, optimizer_all
                    )

                total_loss    += batch_loss * labels.size(0)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                step_counter  += 1

        metrics = {
            "loss":        total_loss / max(total_samples, 1),
            "accuracy":    total_correct / max(total_samples, 1),
            "num_samples": total_samples,
            "phase":       self._phase,
        }
        if self.use_cms:
            metrics.update(self.cms.get_stats())

        logger.info(
            f"Train phase={self._phase}: "
            f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}"
        )
        return metrics

    def _step_single(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, torch.Tensor]:
        """Phase 0-1: single optimizer step."""
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            logits = self.model.forward_all_exits(images)
            loss = self._multi_exit_ce(logits, labels)
            if self.use_kd:
                loss = loss + self.kd_weight * self._kd_loss(logits)

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(self.fast_params) + list(self.slow_params), 1.0
            )
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.fast_params) + list(self.slow_params), 1.0
            )
            optimizer.step()

        return loss.item(), logits[-1].detach().argmax(1)

    def _step_fast_slow(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        step_counter: int,
        optimizer_fast: torch.optim.Optimizer,
        optimizer_slow: torch.optim.Optimizer,
        total_loss_ref,  # unused, kept for signature clarity
    ):
        """Phase 2+: fast update every step, slow update every K steps."""
        # ── Fast update (inner loop) ──────────────────────────────────────────
        optimizer_fast.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            logits = self.model.forward_all_exits(images)
            loss_fast = self._multi_exit_ce(logits, labels)
            if self.use_kd:
                loss_fast = loss_fast + self.kd_weight * self._kd_loss(logits)

        if self.scaler:
            self.scaler.scale(loss_fast).backward(retain_graph=True)
            self.scaler.unscale_(optimizer_fast)
            nn.utils.clip_grad_norm_(self.fast_params, 1.0)
            self.scaler.step(optimizer_fast)
            self.scaler.update()
        else:
            loss_fast.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.fast_params, 1.0)
            optimizer_fast.step()

        # ── Slow update (outer loop, every K steps) ───────────────────────────
        adaptive_k = max(2, int(
            self.slow_update_freq * (1 - 0.5 * step_counter /
            max(step_counter + 1, 1))
        ))

        if step_counter % self.slow_update_freq == 0:
            optimizer_slow.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits_slow = self.model.forward_all_exits(images)
                loss_slow = self._multi_exit_ce(logits_slow, labels)
                if self.use_kd:
                    loss_slow = loss_slow + self.kd_weight * self._kd_loss(logits_slow)
                # ── CMS regularization ────────────────────────────────────────
                if self.use_cms:
                    loss_slow = loss_slow + self.cms.compute_loss(self.slow_params)

            if self.scaler:
                self.scaler.scale(loss_slow).backward()
                self.scaler.unscale_(optimizer_slow)
                nn.utils.clip_grad_norm_(self.slow_params, 1.0)
                self.scaler.step(optimizer_slow)
                self.scaler.update()
            else:
                loss_slow.backward()
                nn.utils.clip_grad_norm_(self.slow_params, 1.0)
                optimizer_slow.step()

            # ── CMS buffer update ─────────────────────────────────────────────
            if self.use_cms:
                self.cms.update(self.slow_params)

    # ── Evaluation ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader,
        threshold: float = 0.8,
    ) -> Dict[str, float]:
        """Evaluate with early-exit inference."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        exit_counts = [0, 0, 0, 0]

        for images, labels in test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits, exits = self.model(images, threshold=threshold)
            loss = F.cross_entropy(logits, labels)
            total_loss    += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
            for e in exits.cpu().tolist():
                exit_counts[e] += 1

        return {
            "loss":             total_loss / max(total_samples, 1),
            "accuracy":         total_correct / max(total_samples, 1),
            "num_samples":      total_samples,
            "exit_distribution": exit_counts,
            "avg_exit":         sum(i * c for i, c in enumerate(exit_counts)) / max(total_samples, 1),
        }

    # ── FL parameter exchange ──────────────────────────────────────────────────

    def get_model_state_dict(self) -> dict:
        """
        Extract state_dict for FL aggregation.
        CMS buffers are NOT included — they stay local.
        """
        return {
            k: v.detach().cpu()
            for k, v in self.model.state_dict().items()
        }

    def load_model_state_dict(self, state_dict: dict) -> None:
        """Load global model weights from server."""
        self.model.load_state_dict(state_dict, strict=False)
