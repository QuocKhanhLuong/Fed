"""
Nested Early-Exit Trainer for Federated Learning

This module implements the Nested Learning paradigm (Google Research, NeurIPS 2025)
combined with Early-Exit Networks for Federated Learning.

Key Innovation: Multi-timescale Optimization
---------------------------------------------
Instead of updating all parameters uniformly, we divide them into:

1. FAST WEIGHTS (Context Flow):
   - Exit classifiers (exit1, exit2, exit3)
   - Updated every step with high learning rate
   - Quickly adapts to local client data

2. SLOW WEIGHTS (Long-term Memory):
   - Backbone stages (stage1, stage2, stage3)
   - Updated every N steps with low learning rate
   - Preserves global knowledge across FL rounds

This addresses the key FL challenge: Catastrophic Forgetting
When aggregating models from non-IID clients, the global knowledge
tends to be overwritten. By using slow updates for the backbone,
we preserve the global structure while allowing fast local adaptation.

Mathematical Formulation:
-------------------------
Let θ = (θ_fast, θ_slow) be the model parameters.

Inner Loop (Fast Update - every step):
    θ_fast ← θ_fast - η_fast · ∇_θ_fast L

Outer Loop (Slow Update - every K steps):
    θ_slow ← θ_slow - η_slow · ∇_θ_slow L

where η_fast >> η_slow (typically 10x)

References:
- Nested Learning (Google Research, NeurIPS 2025)
- "Difficulty-Aware FL with Early-Exit Networks" (IEEE TMC, 2025)

Author: Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import gc

logger = logging.getLogger(__name__)


class NestedEarlyExitTrainer:
    """
    Trainer implementing Nested Learning for Early-Exit Networks in FL.
    
    This trainer uses multi-timescale optimization to:
    1. Quickly adapt exit classifiers to local data (Fast Weights)
    2. Slowly update backbone to preserve global knowledge (Slow Weights)
    
    Algorithm: Nested Optimization for FL
    ─────────────────────────────────────
    Input: Local dataset D_i, global weights w^t, 
           η_fast, η_slow, slow_update_freq K
    
    1. (θ_fast, θ_slow) ← w^t
    2. step ← 0
    3. for epoch = 1 to E do
    4.    for (x, y) ∈ D_i do
    5.       y₁, y₂, y₃ ← f(x; θ)
    6.       L ← Σ α_k · CE(y_k, y)
    7.       
    8.       # Fast Update (Inner Loop)
    9.       θ_fast ← θ_fast - η_fast · ∇_θ_fast L
    10.      
    11.      step ← step + 1
    12.      if step mod K == 0 then
    13.         # Slow Update (Outer Loop)
    14.         θ_slow ← θ_slow - η_slow · ∇_θ_slow L
    15. return (θ_fast, θ_slow)
    
    Args:
        num_classes (int): Number of output classes
        exit_weights (List[float]): α_k for each exit (default: [0.3, 0.3, 0.4])
        fast_lr_multiplier (float): η_fast = η × multiplier (default: 10.0)
        slow_update_freq (int): K - update slow weights every K steps (default: 5)
        device (str): Computation device
        use_mixed_precision (bool): Enable AMP
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        exit_weights: List[float] = [0.3, 0.3, 0.4],
        fast_lr_multiplier: float = 10.0,
        slow_update_freq: int = 5,
        device: str = "auto",
        use_mixed_precision: bool = True,
        use_self_distillation: bool = True,
        distillation_weight: float = 0.5,
        distillation_temp: float = 3.0,
    ):
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Import and create model
        from models.early_exit_mobilevit import EarlyExitMobileViTv2
        self.model = EarlyExitMobileViTv2(num_classes=num_classes)
        self.model.to(self.device)
        
        # Mixed precision setup
        self.use_amp = use_mixed_precision and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Exit loss weights
        self.exit_weights = exit_weights
        self.num_exits = len(exit_weights)
        
        # Nested Learning hyperparameters
        self.fast_lr_multiplier = fast_lr_multiplier
        self.slow_update_freq = slow_update_freq
        
        # Self-Distillation (Exit3 → Exit1, Exit2)
        self.use_self_distillation = use_self_distillation
        self.distillation_weight = distillation_weight
        self.distillation_temp = distillation_temp
        
        # Separate parameter groups
        self._setup_parameter_groups()
        
        # Statistics
        self.stats = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'fast_params': sum(p.numel() for p in self.fast_params),
            'slow_params': sum(p.numel() for p in self.slow_params),
            'fast_lr_multiplier': fast_lr_multiplier,
            'slow_update_freq': slow_update_freq,
        }
        
        logger.info(f"NestedEarlyExitTrainer: device={self.device}")
        logger.info(f"Fast params: {self.stats['fast_params']:,} | "
                    f"Slow params: {self.stats['slow_params']:,}")
        logger.info(f"Fast LR: {fast_lr_multiplier}x | Slow update: every {slow_update_freq} steps")
    
    def _setup_parameter_groups(self):
        """
        Separate model parameters into Fast and Slow groups.
        
        Fast Weights (Context Flow):
            - exit1, exit2, exit3 classifiers
            - These adapt quickly to local client data
        
        Slow Weights (Long-term Memory):
            - stage1, stage2, stage3 backbone
            - These preserve global knowledge
        """
        # Fast: Exit classifiers
        self.fast_params = list(self.model.exit1.parameters()) + \
                          list(self.model.exit2.parameters()) + \
                          list(self.model.exit3.parameters())
        
        # Slow: Backbone stages
        self.slow_params = list(self.model.stage1.parameters()) + \
                          list(self.model.stage2.parameters()) + \
                          list(self.model.stage3.parameters())
        
        logger.info(f"Parameter groups: Fast={len(self.fast_params)}, Slow={len(self.slow_params)}")
    
    def _optimize_memory(self):
        """Clear GPU cache before training."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _compute_distillation_loss(
        self,
        y1: torch.Tensor,
        y2: torch.Tensor,
        y3: torch.Tensor,
    ) -> torch.Tensor:
        """
        Self-Distillation: Exit3 (teacher) → Exit1, Exit2 (students)
        
        L_distill = KL(softmax(y1/T) || softmax(y3/T)) +
                    KL(softmax(y2/T) || softmax(y3/T))
        
        where T is the temperature (higher = softer targets)
        """
        T = self.distillation_temp
        
        # Soft targets from teacher (Exit3)
        teacher_soft = F.softmax(y3.detach() / T, dim=-1)
        
        # KL divergence for students
        student1_log = F.log_softmax(y1 / T, dim=-1)
        student2_log = F.log_softmax(y2 / T, dim=-1)
        
        kl_loss1 = F.kl_div(student1_log, teacher_soft, reduction='batchmean')
        kl_loss2 = F.kl_div(student2_log, teacher_soft, reduction='batchmean')
        
        # Scale by T^2 as per Hinton et al.
        return (kl_loss1 + kl_loss2) * (T ** 2)
    
    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        learning_rate: float = 1e-3,
        fedprox_mu: float = 0.0,
        global_weights: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Nested Learning training with multi-timescale optimization.
        
        Args:
            train_loader: Local training data
            epochs: Number of local epochs
            learning_rate: Base learning rate (η_slow)
            fedprox_mu: FedProx regularization (applied to slow weights only)
            global_weights: Global model weights for FedProx
            
        Returns:
            Training metrics
        """
        self._optimize_memory()
        self.model.train()
        
        # Nested Optimizers with different learning rates
        optimizer_fast = torch.optim.AdamW(
            self.fast_params,
            lr=learning_rate * self.fast_lr_multiplier,  # η_fast = 10x
            weight_decay=0.01,
        )
        optimizer_slow = torch.optim.AdamW(
            self.slow_params,
            lr=learning_rate,  # η_slow = 1x
            weight_decay=0.01,
        )
        
        # Global weights for FedProx (only on slow weights)
        w_global_slow = None
        if fedprox_mu > 0 and global_weights is not None:
            # Extract only backbone (slow) weights from global
            slow_indices = self._get_slow_param_indices()
            w_global_slow = [
                torch.from_numpy(global_weights[i]).to(self.device)
                for i in slow_indices
            ]
        
        # Training metrics
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        step_counter = 0
        
        for epoch in range(epochs):
            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # ═══════════════════════════════════════════════════════
                # INNER LOOP: Fast Update (every step)
                # ═══════════════════════════════════════════════════════
                optimizer_fast.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # Forward all exits
                    y1, y2, y3 = self.model.forward_all_exits(images)
                    
                    # Multi-exit CE loss
                    loss_ce = (
                        self.exit_weights[0] * F.cross_entropy(y1, labels) +
                        self.exit_weights[1] * F.cross_entropy(y2, labels) +
                        self.exit_weights[2] * F.cross_entropy(y3, labels)
                    )
                    
                    # Self-Distillation loss (Exit3 → Exit1, Exit2)
                    if self.use_self_distillation:
                        loss_distill = self._compute_distillation_loss(y1, y2, y3)
                        loss_fast = loss_ce + self.distillation_weight * loss_distill
                    else:
                        loss_fast = loss_ce
                
                # Backward and update fast weights
                if self.scaler is not None:
                    self.scaler.scale(loss_fast).backward(retain_graph=True)
                    self.scaler.step(optimizer_fast)
                    self.scaler.update()
                else:
                    loss_fast.backward(retain_graph=True)
                    optimizer_fast.step()
                
                # ═══════════════════════════════════════════════════════
                # OUTER LOOP: Slow Update (every K steps)
                # ═══════════════════════════════════════════════════════
                step_counter += 1
                if step_counter % self.slow_update_freq == 0:
                    optimizer_slow.zero_grad(set_to_none=True)
                    
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        # Recompute loss for slow weights
                        y1, y2, y3 = self.model.forward_all_exits(images)
                        loss_slow = (
                            self.exit_weights[0] * F.cross_entropy(y1, labels) +
                            self.exit_weights[1] * F.cross_entropy(y2, labels) +
                            self.exit_weights[2] * F.cross_entropy(y3, labels)
                        )
                        
                        # FedProx regularization (only on slow/backbone weights)
                        if fedprox_mu > 0 and w_global_slow is not None:
                            prox_term = sum(
                                ((p - g) ** 2).sum()
                                for p, g in zip(self.slow_params, w_global_slow)
                            )
                            loss_slow += (fedprox_mu / 2) * prox_term
                    
                    if self.scaler is not None:
                        self.scaler.scale(loss_slow).backward()
                        self.scaler.step(optimizer_slow)
                        self.scaler.update()
                    else:
                        loss_slow.backward()
                        optimizer_slow.step()
                
                # Accumulate metrics
                total_loss += loss_fast.item() * labels.size(0)
                total_correct += (y3.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)
        
        metrics = {
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_correct / max(total_samples, 1),
            'num_samples': total_samples,
            'epochs': epochs,
            'fast_updates': step_counter,
            'slow_updates': step_counter // self.slow_update_freq,
        }
        
        logger.info(f"NestedTrain: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}, "
                    f"fast_steps={metrics['fast_updates']}, slow_steps={metrics['slow_updates']}")
        return metrics
    
    def _get_slow_param_indices(self) -> List[int]:
        """Get indices of slow (backbone) parameters in full parameter list."""
        slow_param_ids = {id(p) for p in self.slow_params}
        indices = []
        for i, p in enumerate(self.model.parameters()):
            if id(p) in slow_param_ids:
                indices.append(i)
        return indices
    
    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        threshold: float = 0.8,
    ) -> Dict[str, float]:
        """Evaluate with early exit inference."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        exit_counts = [0] * self.num_exits
        
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
        
        metrics = {
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_correct / max(total_samples, 1),
            'num_samples': total_samples,
            'exit_distribution': exit_counts,
            'avg_exit': sum(i * c for i, c in enumerate(exit_counts)) / max(total_samples, 1),
        }
        
        logger.info(f"Eval: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}, "
                    f"exits={exit_counts}")
        return metrics
    
    def get_parameters(self) -> List[np.ndarray]:
        """Extract all model parameters as NumPy arrays."""
        return [p.detach().cpu().numpy() for p in self.model.parameters()]
    
    def get_slow_parameters(self) -> List[np.ndarray]:
        """Extract only slow (backbone) parameters - for FL aggregation."""
        return [p.detach().cpu().numpy() for p in self.slow_params]
    
    def get_fast_parameters(self) -> List[np.ndarray]:
        """Extract only fast (exit) parameters - for personalization."""
        return [p.detach().cpu().numpy() for p in self.fast_params]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set all model parameters from NumPy arrays."""
        with torch.no_grad():
            for param, value in zip(self.model.parameters(), parameters):
                param.copy_(torch.from_numpy(value))
    
    def set_slow_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set only slow (backbone) parameters - for FL aggregation."""
        with torch.no_grad():
            for param, value in zip(self.slow_params, parameters):
                param.copy_(torch.from_numpy(value))


# =============================================================================
# Utility Functions
# =============================================================================

def create_dummy_dataset(
    num_samples: int = 100,
    num_classes: int = 10,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """Create dummy dataset for testing."""
    x = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, num_classes, (num_samples,))
    
    split = int(0.8 * num_samples)
    train_ds = TensorDataset(x[:split], y[:split])
    test_ds = TensorDataset(x[split:], y[split:])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, test_loader


# =============================================================================
# Unit Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NestedEarlyExitTrainer - Nested Learning Test")
    print("=" * 60)
    
    trainer = NestedEarlyExitTrainer(
        num_classes=10,
        exit_weights=[0.3, 0.3, 0.4],
        fast_lr_multiplier=10.0,
        slow_update_freq=5,
        use_self_distillation=True,
        device="cpu",
        use_mixed_precision=False,
    )
    
    print(f"\nTrainer Statistics:")
    for k, v in trainer.stats.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    
    train_loader, test_loader = create_dummy_dataset(50, 10)
    
    print("\n" + "=" * 60)
    print("Training with Nested Optimization...")
    print("=" * 60)
    
    metrics = trainer.train(train_loader, epochs=2, learning_rate=1e-3)
    print(f"\nTraining Metrics: {metrics}")
    
    print("\n" + "=" * 60)
    print("Evaluation with Early Exit...")
    print("=" * 60)
    
    eval_metrics = trainer.evaluate(test_loader, threshold=0.5)
    print(f"\nEval Metrics: {eval_metrics}")
    
    # Test parameter extraction
    all_params = trainer.get_parameters()
    slow_params = trainer.get_slow_parameters()
    fast_params = trainer.get_fast_parameters()
    
    print(f"\nParameter extraction:")
    print(f"  All: {len(all_params)} arrays")
    print(f"  Slow (backbone): {len(slow_params)} arrays")
    print(f"  Fast (exits): {len(fast_params)} arrays")
    
    print("\n" + "=" * 60)
    print("✅ NestedEarlyExitTrainer test completed!")
    print("=" * 60)
