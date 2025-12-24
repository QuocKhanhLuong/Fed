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

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import gc
from tqdm import tqdm

# PyTorch 1.x/2.x compatibility (for Jetson Nano)
try:
    from nestedfl.utils.torch_compat import get_autocast, get_grad_scaler
except ImportError:
    from utils.torch_compat import get_autocast, get_grad_scaler

logger = logging.getLogger(__name__)


# =============================================================================
# Feature 1: Local Surprise Signal (LSS) - NeurIPS 2025
# =============================================================================

class LocalSurpriseSignal:
    """
    Local Surprise Signal (LSS) from Nested Learning.
    
    LSS measures how "surprising" each sample is to the model, allowing
    the training to focus on difficult/informative samples.
    
    Formula: LSS(x) = loss(x) / E[loss] (normalized per-sample loss)
    
    This implements importance sampling where harder samples get more weight.
    
    Reference: "Nested Learning" (Google Research, NeurIPS 2025)
    """
    
    def __init__(
        self,
        enabled: bool = True,
        temperature: float = 1.0,
        ema_decay: float = 0.99,
    ):
        """
        Args:
            enabled: Whether LSS is active
            temperature: Controls sharpness of weighting (higher = more uniform)
            ema_decay: Decay for running mean loss estimation
        """
        self.enabled = enabled
        self.temperature = temperature
        self.ema_decay = ema_decay
        self.running_mean_loss = None
        
    def compute_weights(
        self,
        per_sample_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sample weights based on surprise signal.
        
        Args:
            per_sample_loss: Loss per sample (batch_size,)
            
        Returns:
            Sample weights (batch_size,) - higher weight for surprising samples
        """
        if not self.enabled:
            return torch.ones_like(per_sample_loss)
        
        # Clamp losses to prevent extreme values
        per_sample_loss = per_sample_loss.clamp(min=1e-6, max=100.0)
        
        # Update running mean
        batch_mean = per_sample_loss.mean().detach()
        if self.running_mean_loss is None:
            self.running_mean_loss = batch_mean
        else:
            self.running_mean_loss = (
                self.ema_decay * self.running_mean_loss + 
                (1 - self.ema_decay) * batch_mean
            )
        
        # Compute normalized surprise (LSS)
        # Higher loss = more surprising = higher weight
        lss = per_sample_loss / (self.running_mean_loss + 1e-6)
        
        # Clamp LSS to prevent extreme outliers from dominating
        lss = lss.clamp(min=0.1, max=10.0)
        
        # Apply temperature (higher = more uniform weights)
        # Use higher temperature (2.0) for stability
        temp = max(self.temperature, 2.0)
        weights = torch.softmax(lss / temp, dim=0) * len(lss)
        
        # Additional clamp on final weights
        weights = weights.clamp(min=0.1, max=5.0)
        
        return weights.detach()
    
    def get_stats(self) -> Dict[str, float]:
        """Get LSS statistics."""
        return {
            'lss_running_mean': self.running_mean_loss.item() if self.running_mean_loss is not None else 0.0,
        }


# =============================================================================
# Feature 2: Deep Momentum GD (DMGD) - NeurIPS 2025
# =============================================================================

class DeepMomentum(nn.Module):
    """
    Deep Momentum using MLP instead of EMA.
    
    Standard momentum: m_t = β*m_{t-1} + g_t
    Deep momentum: m_t = MLP(concat(m_{t-1}, g_t))
    
    This allows the optimizer to learn non-linear gradient aggregation,
    potentially capturing higher-order optimization dynamics.
    
    Reference: "Nested Learning" (Google Research, NeurIPS 2025)
    """
    
    def __init__(
        self,
        param_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        """
        Args:
            param_dim: Dimension of gradient/momentum vectors
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
        """
        super().__init__()
        
        # Small MLP for momentum transformation
        layers = []
        input_dim = param_dim * 2  # concat(momentum, gradient)
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        
        layers.append(nn.Linear(hidden_dim, param_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize to behave like standard momentum initially
        self._init_as_momentum()
        
        # Momentum buffer
        self.register_buffer('momentum', None)
        
    def _init_as_momentum(self):
        """Initialize MLP to approximate standard momentum."""
        with torch.no_grad():
            for module in self.net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self, gradient: torch.Tensor, beta: float = 0.9) -> torch.Tensor:
        """
        Apply deep momentum to gradient.
        
        Args:
            gradient: Current gradient (flattened)
            beta: Momentum coefficient (for fallback)
            
        Returns:
            Updated momentum vector
        """
        if self.momentum is None:
            self.momentum = torch.zeros_like(gradient)
        
        # Fallback to standard momentum for stability
        standard_momentum = beta * self.momentum + gradient
        
        # Deep momentum: learn residual correction
        combined = torch.cat([self.momentum, gradient], dim=-1)
        correction = self.net(combined.unsqueeze(0)).squeeze(0)
        
        # Combine standard + learned correction (with small weight)
        self.momentum = standard_momentum + 0.1 * correction
        
        return self.momentum


# =============================================================================
# Feature 3: Extended Continuum Memory System - NeurIPS 2025
# =============================================================================

class ContinuumMemorySystem(nn.Module):
    """
    Continuum Memory System (CMS) from Google's Nested Learning.
    
    Instead of binary short/long-term memory, CMS uses a SPECTRUM
    of memory modules, each updating at different frequencies.
    
    Memory Levels (Extended to 4+):
    - Fast (τ=1):    Update every step - immediate context
    - Medium (τ=5):  Update every 5 steps - recent patterns
    - Slow (τ=25):   Update every 25 steps - medium-term
    - Anchor (τ=125): Update every 125 steps - long-term knowledge
    
    Memory Budget: O(1) - Fixed size, no growth over FL rounds!
    
    Reference: "Nested Learning" (Google Research, NeurIPS 2025)
    """
    
    def __init__(
        self,
        enabled: bool = True,
        update_freqs: List[int] = None,
        decay_rates: List[float] = None,
        memory_weight: float = 0.001,
        num_levels: int = 4,
        base_freq: int = 5,
    ):
        """
        Args:
            enabled: Whether CMS is active
            update_freqs: Update frequency for each memory level
            decay_rates: EMA decay for each level
            memory_weight: Weight of memory regularization loss
            num_levels: Number of memory levels (default: 4)
            base_freq: Base for exponential frequency spacing
        """
        super().__init__()
        self.enabled = enabled
        self.memory_weight = memory_weight
        
        # Auto-generate exponential frequencies if not provided
        if update_freqs is None:
            update_freqs = self.create_exponential_levels(base_freq, num_levels)
        if decay_rates is None:
            # Decay increases with level (slower update = more decay)
            decay_rates = [1 - 1.0 / (i + 1) for i in range(len(update_freqs))]
            decay_rates[0] = 0.0  # Fast level has no memory decay
        
        self.update_freqs = update_freqs
        self.decay_rates = decay_rates
        self.num_levels = len(update_freqs)
        
        # Memory buffers (initialized on first use)
        self.memories: List[Optional[torch.Tensor]] = [None] * self.num_levels
        # Gradient accumulators per level
        self.grad_accumulators: List[Optional[torch.Tensor]] = [None] * self.num_levels
        self._initialized = False
        
        logger.info(f"CMS initialized: enabled={enabled}, levels={self.num_levels}")
        logger.info(f"  Update freqs: {update_freqs}")
        logger.info(f"  Decay rates: {[f'{d:.2f}' for d in decay_rates]}")
    
    @staticmethod
    def create_exponential_levels(base: int = 5, num_levels: int = 4) -> List[int]:
        """Create exponentially spaced update frequencies."""
        return [base ** i for i in range(num_levels)]  # [1, 5, 25, 125]
    
    def _initialize_memories(self, template: torch.Tensor):
        """Initialize memory buffers from template tensor."""
        if self._initialized:
            return
        for i in range(self.num_levels):
            self.memories[i] = template.detach().clone()
            self.grad_accumulators[i] = torch.zeros_like(template)
        self._initialized = True
        logger.info(f"CMS memories initialized: {template.numel():,} params × {self.num_levels} levels")
    
    @torch.no_grad()
    def update(self, weights: torch.Tensor, step: int, gradient: torch.Tensor = None):
        """
        Update memory buffers based on current step.
        
        Args:
            weights: Current slow (backbone) weights as flat tensor
            step: Current training step
            gradient: Optional gradient for accumulation
        """
        if not self.enabled:
            return
        
        # Initialize on first call
        if not self._initialized:
            self._initialize_memories(weights)
        
        # Accumulate gradients for each level
        if gradient is not None:
            for i in range(self.num_levels):
                self.grad_accumulators[i] += gradient
        
        # Update each memory level based on its frequency
        for i, (freq, decay) in enumerate(zip(self.update_freqs, self.decay_rates)):
            if step % freq == 0:
                # EMA update: m = α*m + (1-α)*w
                self.memories[i] = decay * self.memories[i] + (1 - decay) * weights.detach()
                # Reset gradient accumulator for this level
                if self.grad_accumulators[i] is not None:
                    self.grad_accumulators[i].zero_()
    def get_memory_loss(self, current_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute memory regularization loss.
        
        Encourages model to stay close to slow memories (long-term knowledge).
        Slower memories have higher weight (more important to preserve).
        
        Returns:
            Memory loss term (weighted MSE to each memory level)
        """
        if not self.enabled or not self._initialized:
            return torch.tensor(0.0, device=current_weights.device)
        
        loss = torch.tensor(0.0, device=current_weights.device)
        
        for i, mem in enumerate(self.memories):
            if mem is not None:
                # Slower memories (higher index) have higher weight
                level_weight = (i + 1) / self.num_levels
                loss = loss + level_weight * F.mse_loss(current_weights, mem)
        
        return self.memory_weight * loss
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get statistics about memory divergence."""
        if not self._initialized:
            return {}
        
        stats = {}
        for i, mem in enumerate(self.memories):
            if mem is not None:
                stats[f'memory_{i}_norm'] = mem.norm().item()
        return stats


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
        fast_lr_multiplier: float = 3.0,
        slow_update_freq: int = 5,
        device: str = "auto",
        use_mixed_precision: bool = True,
        use_self_distillation: bool = True,
        distillation_weight: float = 0.1,
        distillation_temp: float = 3.0,
        # CMS (Continuum Memory System) settings
        cms_enabled: bool = True,
        cms_update_freqs: List[int] = None,  # Auto-generate if None
        cms_decay_rates: List[float] = None,
        cms_weight: float = 0.001,
        cms_num_levels: int = 4,  # NEW: Extended from 3 to 4
        # NEW: Local Surprise Signal (LSS)
        use_lss: bool = True,
        lss_temperature: float = 1.0,
        # NEW: Deep Momentum GD (not enabled by default due to overhead)
        use_deep_momentum: bool = False,
        # Use full pretrained backbone from timm
        use_timm_pretrained: bool = True,
    ):
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Import and create model
        if use_timm_pretrained:
            from models.early_exit_mobilevit import TimmPretrainedEarlyExit
            self.model = TimmPretrainedEarlyExit(num_classes=num_classes)
            logger.info("Using TimmPretrainedEarlyExit (full pretrained backbone)")
        else:
            from models.early_exit_mobilevit import EarlyExitMobileViTv2
            self.model = EarlyExitMobileViTv2(num_classes=num_classes)
            logger.info("Using EarlyExitMobileViTv2 (custom architecture)")
        self.model.to(self.device)
        
        # Mixed precision setup (compatible with PyTorch 1.x/2.x)
        self.use_amp = use_mixed_precision and (self.device.type == "cuda")
        self.scaler = get_grad_scaler('cuda') if self.use_amp else None
        
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
        
        # Continuum Memory System (Google Nested Learning) - Extended to N levels
        self.cms = ContinuumMemorySystem(
            enabled=cms_enabled,
            update_freqs=cms_update_freqs,
            decay_rates=cms_decay_rates,
            memory_weight=cms_weight,
            num_levels=cms_num_levels,
        )
        
        # NEW: Local Surprise Signal (LSS) for sample importance weighting
        self.lss = LocalSurpriseSignal(
            enabled=use_lss,
            temperature=lss_temperature,
        )
        self.use_lss = use_lss
        
        # NEW: Deep Momentum (disabled by default)
        self.use_deep_momentum = use_deep_momentum
        self.deep_momentum = None  # Initialized lazily if enabled
        
        # Statistics
        self.stats = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'fast_params': sum(p.numel() for p in self.fast_params),
            'slow_params': sum(p.numel() for p in self.slow_params),
            'fast_lr_multiplier': fast_lr_multiplier,
            'slow_update_freq': slow_update_freq,
            'cms_enabled': cms_enabled,
            'cms_levels': cms_num_levels,
            'lss_enabled': use_lss,
            'dmgd_enabled': use_deep_momentum,
        }
        
        # Calculate FLOPs/GFLOPs using fvcore (doesn't modify model) or fallback to estimate
        try:
            from fvcore.nn import FlopCountAnalysis
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            fca = FlopCountAnalysis(self.model, dummy_input)
            flops = fca.total()
            self.stats['flops'] = flops
            self.stats['gflops'] = flops / 1e9
            self.stats['mflops'] = flops / 1e6
            logger.info(f"Model FLOPs: {flops/1e6:.2f}M ({flops/1e9:.4f} GFLOPs)")
        except ImportError:
            # Fallback: estimate for MobileViTv2-100 on 32x32 input
            # MobileViTv2-100 has ~1.8 GFLOPs on 256x256, scale by (32/256)^2 = 1/64
            estimated_gflops = 1.8 / 64 * 0.5  # ~14 MFLOPs for CIFAR
            self.stats['flops'] = estimated_gflops * 1e9
            self.stats['gflops'] = estimated_gflops
            self.stats['mflops'] = estimated_gflops * 1e3
            logger.info(f"Model FLOPs (estimated): {estimated_gflops*1e3:.2f}M ({estimated_gflops:.4f} GFLOPs)")
        except Exception as e:
            logger.warning(f"FLOPs calculation failed: {e}")
            self.stats['flops'] = 0
            self.stats['gflops'] = 0
        
        logger.info(f"NestedEarlyExitTrainer: device={self.device}")
        logger.info(f"Fast params: {self.stats['fast_params']:,} | "
                    f"Slow params: {self.stats['slow_params']:,}")
        logger.info(f"Fast LR: {fast_lr_multiplier}x | Slow update: every {slow_update_freq} steps")
        logger.info(f"CMS: enabled={cms_enabled}, levels={cms_num_levels}")
        logger.info(f"LSS: enabled={use_lss} | DMGD: enabled={use_deep_momentum}")
    
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
        
        # Slow: Backbone (handle both model types)
        if hasattr(self.model, 'backbone'):
            # TimmPretrainedEarlyExit uses 'backbone'
            self.slow_params = list(self.model.backbone.parameters())
        else:
            # EarlyExitMobileViTv2 uses stage1, stage2, stage3
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
    
    def _apply_deep_momentum_slow(self):
        """
        Apply Deep Momentum GD to slow parameter gradients.
        
        This modifies the gradients in-place using a learned MLP that
        combines current gradient with momentum history.
        """
        with torch.no_grad():
            # Initialize deep momentum MLP lazily
            if self.deep_momentum is None:
                # Use a smaller dimension for efficiency
                grad_dim = min(1024, sum(p.numel() for p in self.slow_params if p.grad is not None))
                self.deep_momentum = DeepMomentum(param_dim=grad_dim).to(self.device)
                logger.info(f"DMGD initialized: dim={grad_dim}")
            
            # Collect gradients from slow params
            grads = []
            for p in self.slow_params:
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
            
            if not grads:
                return
            
            # Concatenate and truncate/pad to fixed size
            full_grad = torch.cat(grads)
            target_dim = self.deep_momentum.momentum.shape[0] if self.deep_momentum.momentum is not None else 1024
            
            if full_grad.numel() > target_dim:
                # Sample subset for deep momentum computation
                indices = torch.randperm(full_grad.numel(), device=self.device)[:target_dim]
                sampled_grad = full_grad[indices]
            else:
                sampled_grad = F.pad(full_grad, (0, target_dim - full_grad.numel()))
            
            # Apply deep momentum
            modified_grad = self.deep_momentum(sampled_grad)
            
            # Scale factor for gradient modification (small to maintain stability)
            scale = 0.1
            
            # Apply modification back to original gradients
            if full_grad.numel() > target_dim:
                # Only modify sampled positions
                full_grad[indices] += scale * (modified_grad[:len(indices)] - sampled_grad)
            else:
                # Modify all gradients
                full_grad[:modified_grad.numel()] += scale * (modified_grad[:full_grad.numel()] - sampled_grad[:full_grad.numel()])
            
            # Copy modified gradients back
            offset = 0
            for p in self.slow_params:
                if p.grad is not None:
                    numel = p.grad.numel()
                    p.grad.copy_(full_grad[offset:offset+numel].view_as(p.grad))
                    offset += numel
    
    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        learning_rate: float = 1e-3,
        fedprox_mu: float = 0.0,
        global_weights: Optional[List[np.ndarray]] = None,
        use_scheduler: bool = True,
        warmup_epochs: int = 5,
    ) -> Dict[str, float]:
        """
        Nested Learning training with multi-timescale optimization.
        
        Args:
            train_loader: Local training data
            epochs: Number of local epochs
            learning_rate: Base learning rate (η_slow)
            fedprox_mu: FedProx regularization (applied to slow weights only)
            global_weights: Global model weights for FedProx
            use_scheduler: Use Cosine Annealing + Warmup (MobileViT official)
            warmup_epochs: Linear warmup epochs before cosine decay
            
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
        
        # Use DeepMomentumGD for slow weights if enabled
        if self.use_deep_momentum:
            try:
                from nestedfl.nested_learning.optimizers import DeepMomentumGD
                optimizer_slow = DeepMomentumGD(
                    params=self.slow_params,
                    lr=learning_rate,
                    momentum=0.9,
                    memory_lr=1e-4,
                    use_shared_memory=True,
                    gradient_checkpointing=True,
                    use_factorized_memory=True,
                    internal_loss_mode='surrogate',  # Paper: cosine + magnitude + temporal
                )
                logger.info("Using DeepMomentumGD for slow weights (official nested-learning)")
            except ImportError as e:
                logger.warning(f"DeepMomentumGD not available: {e}, falling back to AdamW")
                optimizer_slow = torch.optim.AdamW(
                    self.slow_params,
                    lr=learning_rate,
                    weight_decay=0.01,
                )
        else:
            optimizer_slow = torch.optim.AdamW(
                self.slow_params,
                lr=learning_rate,  # η_slow = 1x
                weight_decay=0.01,
            )
        
        # Cosine Annealing + Linear Warmup Scheduler (MobileViT official)
        scheduler_fast = None
        scheduler_slow = None
        if use_scheduler and epochs > 1:
            # LambdaLR with warmup + cosine decay
            import math
            def lr_lambda(current_epoch):
                if current_epoch < warmup_epochs:
                    # Linear warmup
                    return float(current_epoch + 1) / float(warmup_epochs)
                else:
                    # Cosine annealing
                    progress = float(current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
            
            scheduler_fast = torch.optim.lr_scheduler.LambdaLR(optimizer_fast, lr_lambda)
            scheduler_slow = torch.optim.lr_scheduler.LambdaLR(optimizer_slow, lr_lambda)
            logger.info(f"Scheduler: Cosine + {warmup_epochs} warmup epochs")
        
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
            # Progress bar for batches
            pbar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=False,
                ncols=100
            )
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # ═══════════════════════════════════════════════════════
                # INNER LOOP: Fast Update (every step)
                # ═══════════════════════════════════════════════════════
                optimizer_fast.zero_grad(set_to_none=True)
                
                with get_autocast('cuda', enabled=self.use_amp):
                    # Forward all exits
                    y1, y2, y3 = self.model.forward_all_exits(images)
                    
                    # Per-sample losses for LSS
                    loss1_per_sample = F.cross_entropy(y1, labels, reduction='none')
                    loss2_per_sample = F.cross_entropy(y2, labels, reduction='none')
                    loss3_per_sample = F.cross_entropy(y3, labels, reduction='none')
                    
                    # Combined per-sample loss (weighted)
                    combined_per_sample = (
                        self.exit_weights[0] * loss1_per_sample +
                        self.exit_weights[1] * loss2_per_sample +
                        self.exit_weights[2] * loss3_per_sample
                    )
                    
                    # Apply LSS weighting (harder samples get more weight)
                    if self.use_lss:
                        lss_weights = self.lss.compute_weights(combined_per_sample)
                        loss_ce = (combined_per_sample * lss_weights).mean()
                    else:
                        loss_ce = combined_per_sample.mean()
                    
                    # Self-Distillation loss (Exit3 → Exit1, Exit2)
                    if self.use_self_distillation:
                        loss_distill = self._compute_distillation_loss(y1, y2, y3)
                        loss_fast = loss_ce + self.distillation_weight * loss_distill
                    else:
                        loss_fast = loss_ce
                
                # Backward and update fast weights
                if self.scaler is not None:
                    self.scaler.scale(loss_fast).backward(retain_graph=True)
                    self.scaler.unscale_(optimizer_fast)
                    torch.nn.utils.clip_grad_norm_(self.fast_params, max_norm=1.0)
                    self.scaler.step(optimizer_fast)
                    self.scaler.update()
                else:
                    loss_fast.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.fast_params, max_norm=1.0)
                    optimizer_fast.step()
                
                # ═══════════════════════════════════════════════════════
                # OUTER LOOP: Slow Update (every K steps)
                # ═══════════════════════════════════════════════════════
                step_counter += 1
                if step_counter % self.slow_update_freq == 0:
                    optimizer_slow.zero_grad(set_to_none=True)
                    
                    with get_autocast('cuda', enabled=self.use_amp):
                        # Recompute loss for slow weights (use same LSS weights as fast)
                        y1, y2, y3 = self.model.forward_all_exits(images)
                        
                        # Per-sample losses (same as fast update)
                        loss1_ps = F.cross_entropy(y1, labels, reduction='none')
                        loss2_ps = F.cross_entropy(y2, labels, reduction='none')
                        loss3_ps = F.cross_entropy(y3, labels, reduction='none')
                        combined_ps = (
                            self.exit_weights[0] * loss1_ps +
                            self.exit_weights[1] * loss2_ps +
                            self.exit_weights[2] * loss3_ps
                        )
                        
                        # Apply same LSS weighting for consistency
                        if self.use_lss:
                            # Reuse weights from fast loop (already computed)
                            lss_w = self.lss.compute_weights(combined_ps)
                            loss_slow = (combined_ps * lss_w).mean()
                        else:
                            loss_slow = combined_ps.mean()
                        
                        # FedProx regularization (only on slow/backbone weights)
                        if fedprox_mu > 0 and w_global_slow is not None:
                            prox_term = sum(
                                ((p - g) ** 2).sum()
                                for p, g in zip(self.slow_params, w_global_slow)
                            )
                            loss_slow += (fedprox_mu / 2) * prox_term
                        
                        # CMS: Disabled by default - causes issues similar to EWC
                        # Only enable if cms_weight > 0 and after sufficient warmup
                        if self.cms.enabled and step_counter > 100:
                            slow_tensor = torch.cat([p.view(-1) for p in self.slow_params])
                            loss_memory = self.cms.get_memory_loss(slow_tensor)
                            # Use very small weight to avoid suppressing learning
                            loss_slow = loss_slow + 0.001 * loss_memory
                    
                    if self.scaler is not None:
                        self.scaler.scale(loss_slow).backward()
                        self.scaler.unscale_(optimizer_slow)
                        torch.nn.utils.clip_grad_norm_(self.slow_params, max_norm=1.0)
                        # DeepMomentumGD handles memory/momentum internally
                        self.scaler.step(optimizer_slow)
                        self.scaler.update()
                    else:
                        loss_slow.backward()
                        torch.nn.utils.clip_grad_norm_(self.slow_params, max_norm=1.0)
                        # DeepMomentumGD handles memory/momentum internally
                        optimizer_slow.step()
                    
                    # ═══════════════════════════════════════════════════════
                    # CMS: Update Memory Buffers (after optimizer step)
                    # ═══════════════════════════════════════════════════════
                    if self.cms.enabled:
                        with torch.no_grad():
                            slow_tensor = torch.cat([p.view(-1) for p in self.slow_params])
                            self.cms.update(slow_tensor, step_counter)
                
                # Accumulate metrics
                total_loss += loss_fast.item() * labels.size(0)
                total_correct += (y3.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)
                
                # Collect predictions for F1/Precision/Recall
                if 'all_preds' not in locals():
                    all_preds = []
                    all_labels = []
                all_preds.extend(y3.argmax(1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss / total_samples:.4f}',
                    'acc': f'{100 * total_correct / total_samples:.1f}%'
                })
            
            # Calculate per-epoch metrics
            epoch_acc = total_correct / max(total_samples, 1)
            epoch_loss = total_loss / max(total_samples, 1)
            
            # Calculate F1/Precision/Recall using sklearn
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score
                precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
                logger.info(f"  Epoch {epoch+1}/{epochs}: acc={epoch_acc:.4f}, loss={epoch_loss:.4f}, "
                           f"F1={f1:.4f}, Prec={precision:.4f}, Recall={recall:.4f}")
            except ImportError:
                logger.info(f"  Epoch {epoch+1}/{epochs}: acc={epoch_acc:.4f}, loss={epoch_loss:.4f}")
            
            # Step schedulers at end of epoch
            if scheduler_fast is not None:
                scheduler_fast.step()
            if scheduler_slow is not None:
                scheduler_slow.step()
        
        metrics = {
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_correct / max(total_samples, 1),
            'num_samples': total_samples,
            'epochs': epochs,
            'fast_updates': step_counter,
            'slow_updates': step_counter // self.slow_update_freq,
        }
        
        # Add CMS stats if enabled
        if self.cms.enabled:
            cms_stats = self.cms.get_memory_stats()
            metrics.update(cms_stats)
        
        # Add LSS stats if enabled
        if self.use_lss:
            lss_stats = self.lss.get_stats()
            metrics.update(lss_stats)
        
        logger.info(f"NestedTrain: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}, "
                    f"fast_steps={metrics['fast_updates']}, slow_steps={metrics['slow_updates']}")
        if self.cms.enabled and 'memory_0_norm' in metrics:
            mem_norms = [f"{metrics.get('memory_' + str(i) + '_norm', 0):.2f}" for i in range(4)]
            logger.info(f"  CMS Memory norms: {mem_norms}")
        if self.use_lss and 'lss_running_mean' in metrics:
            logger.info(f"  LSS running mean loss: {metrics.get('lss_running_mean', 0):.4f}")
        
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
    
    def load_model_state_dict(self, state_dict: dict) -> None:
        """
        Load model weights from state_dict (proper key-based matching).
        
        This is the recommended way to load weights in FL as it ensures
        proper parameter matching by name instead of order.
        Filters out thop-added keys (total_ops, total_params).
        """
        # Filter out thop-added keys
        filtered_dict = {k: v for k, v in state_dict.items() 
                         if 'total_ops' not in k and 'total_params' not in k}
        self.model.load_state_dict(filtered_dict, strict=False)
    
    def get_model_state_dict(self) -> dict:
        """
        Get model state_dict for FL aggregation.
        Filters out thop-added keys (total_ops, total_params).
        
        Returns:
            State dict with parameter names as keys
        """
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()
                if 'total_ops' not in k and 'total_params' not in k}


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
