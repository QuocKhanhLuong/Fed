"""
Early-Exit Trainer for Federated Learning

This module implements the local training procedure for:

    "Difficulty-Aware Federated Learning with Early-Exit Networks"
    IEEE Transactions on Mobile Computing, 2025

Training Algorithm (Section IV-A):
---------------------------------
The multi-exit training uses weighted cross-entropy loss:

    L_total = Σ_{k=1}^{K} α_k · CE(f_k(x), y)

where α_k are learnable or fixed exit weights.

Optimization (Section IV-B):
- Optimizer: AdamW with weight decay λ = 0.01
- Mixed Precision: FP16 forward, FP32 gradients
- Gradient Checkpointing: Optional memory optimization

FedProx Integration (Optional):
    L_prox = L_total + (μ/2) ||w - w_global||²

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

# PyTorch 1.x/2.x compatibility (for Jetson Nano)
from utils.torch_compat import get_autocast, get_grad_scaler

logger = logging.getLogger(__name__)


class EarlyExitTrainer:
    """
    Trainer for Early-Exit Networks in Federated Learning.
    
    Implements Algorithm 3: Local Training
    ─────────────────────────────────────
    Input: Local dataset D_i, global weights w^t, learning rate η,
           exit weights α, epochs E
    
    1. w ← w^t                          # Initialize from global
    2. for e = 1 to E do
    3.    for (x, y) ∈ D_i do
    4.       y₁, y₂, y₃ ← f(x; w)       # Forward all exits
    5.       L ← Σ α_k · CE(y_k, y)      # Multi-exit loss
    6.       w ← w - η · ∇L             # Update
    7. return w
    
    Args:
        num_classes (int): Number of output classes
        exit_weights (List[float]): α_k for each exit
        device (str): Computation device
        use_mixed_precision (bool): Enable AMP
        use_gradient_checkpointing (bool): Enable memory optimization
        
    Attributes:
        model (nn.Module): EarlyExitMobileViTv2 instance
        scaler (GradScaler): AMP scaler for mixed precision
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        exit_weights: List[float] = [0.3, 0.3, 0.4],
        device: str = "auto",
        use_mixed_precision: bool = True,
        use_gradient_checkpointing: bool = True,
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
        
        # Enable gradient checkpointing (reduces memory ~40%)
        if use_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Mixed precision setup (compatible with PyTorch 1.x/2.x)
        self.use_amp = use_mixed_precision and (self.device.type == "cuda")
        self.scaler = get_grad_scaler('cuda') if self.use_amp else None
        
        # Loss configuration
        self.exit_weights = exit_weights
        self.num_exits = len(exit_weights)
        
        # Statistics (Table III in paper)
        self.stats = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() 
                                    if p.requires_grad),
        }
        self.stats.update(self.model.count_parameters())
        
        logger.info(f"EarlyExitTrainer: device={self.device}, AMP={self.use_amp}")
        logger.info(f"Parameters: {self.stats['trainable_params']:,} trainable")
    
    def _optimize_memory(self):
        """Clear GPU cache before training."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        learning_rate: float = 1e-3,
        fedprox_mu: float = 0.0,
        global_weights: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Local training with multi-exit loss.
        
        Implements Algorithm 3 from Section IV-A.
        
        Args:
            train_loader: Local training data
            epochs: Number of local epochs E
            learning_rate: Learning rate η
            fedprox_mu: FedProx regularization μ (0 = disabled)
            global_weights: w^t for FedProx
            
        Returns:
            Training metrics {loss, accuracy, num_samples, epochs}
        """
        self._optimize_memory()
        self.model.train()
        
        # Optimizer: AdamW (Section IV-B)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        # Global weights for FedProx
        w_global = None
        if fedprox_mu > 0 and global_weights is not None:
            w_global = [torch.from_numpy(w).to(self.device) for w in global_weights]
        
        # Training metrics
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward with mixed precision
                with get_autocast('cuda', enabled=self.use_amp):
                    # Step 4: Forward all exits
                    y1, y2, y3 = self.model.forward_all_exits(images)
                    
                    # Step 5: Multi-exit loss (Eq. 3)
                    # L = Σ α_k · CE(y_k, y)
                    loss = (
                        self.exit_weights[0] * F.cross_entropy(y1, labels) +
                        self.exit_weights[1] * F.cross_entropy(y2, labels) +
                        self.exit_weights[2] * F.cross_entropy(y3, labels)
                    )
                    
                    # FedProx: L += (μ/2)||w - w_global||²
                    if fedprox_mu > 0 and w_global is not None:
                        prox_term = sum(
                            ((p - g) ** 2).sum()
                            for p, g in zip(self.model.parameters(), w_global)
                        )
                        loss += (fedprox_mu / 2) * prox_term
                
                # Step 6: Gradient update
                optimizer.zero_grad(set_to_none=True)
                
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Accumulate metrics
                total_loss += loss.item() * labels.size(0)
                total_correct += (y3.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)
        
        metrics = {
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_correct / max(total_samples, 1),
            'num_samples': total_samples,
            'epochs': epochs,
        }
        
        logger.info(f"Train: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        threshold: float = 0.8,
    ) -> Dict[str, float]:
        """
        Evaluate with early exit inference.
        
        Args:
            test_loader: Test data
            threshold: Confidence threshold τ
            
        Returns:
            Metrics {loss, accuracy, exit_distribution, avg_exit}
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        exit_counts = [0] * self.num_exits
        
        for images, labels in test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Early exit inference
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
        """Extract model parameters as NumPy arrays."""
        return [p.detach().cpu().numpy() for p in self.model.parameters()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from NumPy arrays."""
        with torch.no_grad():
            for param, value in zip(self.model.parameters(), parameters):
                param.copy_(torch.from_numpy(value))


# =============================================================================
# Utility Functions
# =============================================================================

def create_dummy_dataset(
    num_samples: int = 100,
    num_classes: int = 10,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dummy dataset for testing.
    
    Args:
        num_samples: Total number of samples
        num_classes: Number of classes
        batch_size: Batch size
        
    Returns:
        (train_loader, test_loader)
    """
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
    print("EarlyExitTrainer - IEEE Format Test")
    print("=" * 60)
    
    trainer = EarlyExitTrainer(
        num_classes=10,
        exit_weights=[0.3, 0.3, 0.4],
        device="cpu",
        use_mixed_precision=False,
    )
    
    print(f"\nTrainer Statistics (Table III):")
    for k, v in trainer.stats.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    
    train_loader, test_loader = create_dummy_dataset(50, 10)
    
    metrics = trainer.train(train_loader, epochs=1)
    print(f"\nTraining: {metrics}")
    
    eval_metrics = trainer.evaluate(test_loader, threshold=0.5)
    print(f"Evaluation: {eval_metrics}")
    
    params = trainer.get_parameters()
    print(f"\nExtracted {len(params)} parameter arrays")
