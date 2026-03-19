"""
FedEEP Local Trainer

Local training logic for Federated Early-Exit with Progressive phases.

Components (activated by phase):
  Phase 0  -- Backbone warmup: only Exit4 CE, single optimizer, no fast/slow
  Phase 1  -- Multi-exit CE: all 4 exits supervised, still single optimizer
  Phase 2  -- Fast/Slow + CMS: fast heads + slow backbone, CMS regularizer
  Phase 3  -- + Self-Distillation KD chain (E1<-E2<-E3<-E4)
  Phase 4  -- Full system with all components active

Fast/Slow split:
  - Fast params: 4 exit heads (update every step, high lr)
  - Slow params: ConvNeXt backbone (update every K steps, low lr)
"""

import gc
import logging
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cms import ContinuumMemorySystem

logger = logging.getLogger("fedeep.trainer")

DEFAULT_EXIT_WEIGHTS = [0.1, 0.2, 0.3, 0.4]


class LocalTrainer:
    """
    FedEEP local trainer for ConvNeXtEarlyExit models.

    Args:
        model:              ConvNeXtEarlyExit model instance.
        device:             torch.device to use.
        exit_weights:       alpha_k weights for multi-exit CE loss.
        fast_lr_multiplier: Fast LR = base_lr * this multiplier.
        slow_update_freq:   K -- backbone updates every K steps.
        kd_weight:          Weight for KD chain loss.
        kd_temperature:     Temperature T for soft targets.
        cms_decay_rates:    EMA decay rates for CMS buffers.
        cms_weight:         mu -- overall CMS loss scale.
        proximal_mu:        FedProx proximal term weight (0 = disabled).
        weight_decay:       Optimizer weight decay.
        optimizer_type:     "adamw" or "sgd".
        lr_scheduler:       "cosine", "step", or "none".
        use_mixed_precision: Enable AMP on CUDA.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        exit_weights: List[float] = None,
        fast_lr_multiplier: float = 3.0,
        slow_update_freq: int = 5,
        kd_weight: float = 0.3,
        kd_temperature: float = 4.0,
        cms_decay_rates: List[float] = None,
        cms_weight: float = 0.1,
        proximal_mu: float = 0.0,
        weight_decay: float = 0.01,
        optimizer_type: str = "adamw",
        lr_scheduler: str = "cosine",
        use_mixed_precision: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.exit_weights = exit_weights or DEFAULT_EXIT_WEIGHTS
        assert len(self.exit_weights) == 4, "Need exactly 4 exit weights"

        self.fast_lr_multiplier = fast_lr_multiplier
        self.slow_update_freq = slow_update_freq
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature
        self.proximal_mu = proximal_mu
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.lr_scheduler_type = lr_scheduler

        self.cms = ContinuumMemorySystem(
            decay_rates=cms_decay_rates or [0.90, 0.99, 0.999, 0.9999],
            loss_weight=cms_weight,
        )

        self.use_amp = use_mixed_precision and (device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.fast_params = list(model.get_exit_param_groups())
        self.slow_params = list(model.get_backbone_param_groups())

        self._phase: int = 0
        self.use_multi_exit: bool = False
        self.use_fast_slow: bool = False
        self.use_cms: bool = False
        self.use_kd: bool = False

        # FedProx: snapshot of global weights before local training
        self._global_params: Optional[List[torch.Tensor]] = None

        n_fast = sum(p.numel() for p in self.fast_params)
        n_slow = sum(p.numel() for p in self.slow_params)
        logger.info(
            f"Trainer ready | fast={n_fast:,} slow={n_slow:,} | "
            f"amp={self.use_amp} | optimizer={optimizer_type} | "
            f"scheduler={lr_scheduler} | proximal_mu={proximal_mu}"
        )

    # ── Optimizer / Scheduler factory ─────────────────────────────────────────

    def _make_optimizer(self, params, lr):
        if self.optimizer_type == "sgd":
            return torch.optim.SGD(
                params, lr=lr,
                momentum=0.9, weight_decay=self.weight_decay,
            )
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=self.weight_decay,
        )

    def _make_scheduler(self, optimizer, total_steps):
        if self.lr_scheduler_type == "cosine" and total_steps > 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=1e-6,
            )
        if self.lr_scheduler_type == "step" and total_steps > 0:
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, total_steps // 3), gamma=0.5,
            )
        return None

    # ── Phase control ─────────────────────────────────────────────────────────

    def set_phase(self, phase: int) -> None:
        """
        Toggle components based on current training phase.

        Phase 0: backbone warmup -- only Exit4, single optimizer
        Phase 1: + multi-exit CE for all 4 exits
        Phase 2: + fast/slow split, + CMS regularization
        Phase 3: + KD chain (E1<-E2<-E3<-E4)
        Phase 4: full system (same as 3, EDPA is server-side)
        """
        self._phase = phase
        self.use_multi_exit = phase >= 1
        self.use_fast_slow = phase >= 2
        self.use_cms = phase >= 2
        self.use_kd = phase >= 3
        logger.info(
            f"Phase set to {phase}: "
            f"multi_exit={self.use_multi_exit}, "
            f"fast_slow={self.use_fast_slow}, "
            f"cms={self.use_cms}, "
            f"kd={self.use_kd}"
        )

    # ── Loss utilities ────────────────────────────────────────────────────────

    def _multi_exit_ce(
        self,
        logits: Tuple[torch.Tensor, ...],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_multi_exit:
            return F.cross_entropy(logits[-1], labels)

        loss = torch.tensor(0.0, device=self.device)
        for y_k, alpha_k in zip(logits, self.exit_weights):
            loss = loss + alpha_k * F.cross_entropy(y_k, labels)
        return loss

    def _kd_loss(
        self,
        logits: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Self-distillation chain E1<-E2<-E3<-E4."""
        T = self.kd_temperature
        total_kd = torch.tensor(0.0, device=self.device)
        for student, teacher in zip(logits[:-1], logits[1:]):
            teacher_soft = F.softmax(teacher.detach() / T, dim=-1)
            student_log = F.log_softmax(student / T, dim=-1)
            kl = F.kl_div(student_log, teacher_soft, reduction="batchmean")
            total_kd = total_kd + kl * (T ** 2)
        return total_kd

    def _fedprox_loss(self) -> torch.Tensor:
        """FedProx proximal term: (mu/2) * ||w - w_global||^2."""
        if self._global_params is None or self.proximal_mu <= 0:
            return torch.tensor(0.0, device=self.device)

        prox = torch.tensor(0.0, device=self.device)
        for p, gp in zip(self.model.parameters(), self._global_params):
            prox = prox + (p - gp).pow(2).sum()
        return (self.proximal_mu / 2.0) * prox

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        learning_rate: float,
        server_round: int = 1,
        num_rounds: int = 100,
    ) -> Dict[str, float]:
        """
        Run E local epochs of FedEEP training.

        Args:
            train_loader:  Client's local DataLoader.
            epochs:        Number of local epochs.
            learning_rate: Base LR (scheduler adjusts from here).
            server_round:  Current FL round (for scheduler scaling).
            num_rounds:    Total FL rounds (for scheduler scaling).

        Returns dict of training metrics.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.train()
        total_steps = epochs * len(train_loader)

        # Scale LR by round progress for cosine across FL rounds
        round_progress = (server_round - 1) / max(num_rounds, 1)
        if self.lr_scheduler_type == "cosine":
            adjusted_lr = learning_rate * (
                0.5 * (1 + math.cos(math.pi * round_progress))
            )
            adjusted_lr = max(adjusted_lr, 1e-6)
        else:
            adjusted_lr = learning_rate

        if self.use_fast_slow:
            optimizer_fast = self._make_optimizer(
                self.fast_params, adjusted_lr * self.fast_lr_multiplier,
            )
            optimizer_slow = self._make_optimizer(
                self.slow_params, adjusted_lr,
            )
            sched_fast = self._make_scheduler(optimizer_fast, total_steps)
            sched_slow = self._make_scheduler(optimizer_slow, total_steps)
        else:
            optimizer_all = self._make_optimizer(
                list(self.fast_params) + list(self.slow_params),
                adjusted_lr,
            )
            sched_all = self._make_scheduler(optimizer_all, total_steps)

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
                    )
                    if sched_fast:
                        sched_fast.step()
                    if sched_slow and step_counter % self.slow_update_freq == 0:
                        sched_slow.step()
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", enabled=self.use_amp):
                            logits = self.model.forward_all_exits(images)
                        ce = self._multi_exit_ce(logits, labels)
                    batch_loss = ce.item()
                    preds = logits[-1].argmax(1)
                else:
                    batch_loss, preds = self._step_single(
                        images, labels, optimizer_all
                    )
                    if sched_all:
                        sched_all.step()

                total_loss += batch_loss * labels.size(0)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                step_counter += 1

        metrics = {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
            "num_samples": total_samples,
            "phase": self._phase,
            "lr_used": adjusted_lr,
        }
        if self.use_cms:
            metrics.update(self.cms.get_stats())

        logger.info(
            f"Train phase={self._phase}: "
            f"loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f} "
            f"lr={adjusted_lr:.6f}"
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

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            logits = self.model.forward_all_exits(images)
            loss = self._multi_exit_ce(logits, labels)
            if self.use_kd:
                loss = loss + self.kd_weight * self._kd_loss(logits)
            loss = loss + self._fedprox_loss()

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
    ):
        """Phase 2+: fast update every step, slow update every K steps."""
        optimizer_fast.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            logits = self.model.forward_all_exits(images)
            loss_fast = self._multi_exit_ce(logits, labels)
            if self.use_kd:
                loss_fast = loss_fast + self.kd_weight * self._kd_loss(logits)
            loss_fast = loss_fast + self._fedprox_loss()

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

        if step_counter % self.slow_update_freq == 0:
            optimizer_slow.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits_slow = self.model.forward_all_exits(images)
                loss_slow = self._multi_exit_ce(logits_slow, labels)
                if self.use_kd:
                    loss_slow = loss_slow + self.kd_weight * self._kd_loss(logits_slow)
                if self.use_cms:
                    loss_slow = loss_slow + self.cms.compute_loss(self.slow_params)
                loss_slow = loss_slow + self._fedprox_loss()

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

            if self.use_cms:
                self.cms.update(self.slow_params)

    # ── Evaluation ────────────────────────────────────────────────────────────

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
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
            for e in exits.cpu().tolist():
                exit_counts[e] += 1

        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
            "num_samples": total_samples,
            "exit_distribution": exit_counts,
            "avg_exit": sum(
                i * c for i, c in enumerate(exit_counts)
            ) / max(total_samples, 1),
        }

    # ── FL parameter exchange ─────────────────────────────────────────────────

    def get_weights(self) -> OrderedDict:
        """Extract state_dict for FL aggregation (CMS excluded)."""
        return OrderedDict(
            (k, v.detach().cpu())
            for k, v in self.model.state_dict().items()
        )

    def set_weights(self, state_dict: dict) -> None:
        """Load global model weights and snapshot for FedProx."""
        self.model.load_state_dict(state_dict, strict=False)
        if self.proximal_mu > 0:
            self._global_params = [
                p.detach().clone() for p in self.model.parameters()
            ]
