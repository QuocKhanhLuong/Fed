"""
Integration with official nested-learning library.

This module provides wrappers to use the official nested-learning library
components (DeepMomentumGD, ContinuumMemorySystem) in our FL training.

Reference: https://github.com/erikl2/nested-learning
The source code has been copied to nestedfl/nested_learning/
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)

# Import from local nested_learning module
try:
    from nestedfl.nested_learning.optimizers import DeepMomentumGD
    from nestedfl.nested_learning.memory import ContinuumMemorySystem
    HAS_NESTED_LEARNING = True
    logger.info("✓ Local nested-learning module loaded successfully")
except ImportError as e:
    logger.warning(f"nested-learning module not found: {e}")
    HAS_NESTED_LEARNING = False
    DeepMomentumGD = None
    ContinuumMemorySystem = None


def create_nested_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    momentum: float = 0.9,
    memory_lr: float = 1e-4,
    use_cms: bool = True,
    use_dmgd: bool = True,
    internal_loss_mode: str = 'surrogate',
) -> torch.optim.Optimizer:
    """
    Create optimizer with nested learning features.
    
    If nested-learning library is available, uses official DeepMomentumGD.
    Otherwise, falls back to standard AdamW.
    
    Args:
        model: PyTorch model
        lr: Learning rate
        momentum: Momentum coefficient
        memory_lr: Learning rate for memory modules
        use_cms: Enable ContinuumMemorySystem (ignored if library not available)
        use_dmgd: Enable DeepMomentumGD (ignored if library not available)
        internal_loss_mode: 'surrogate' or 'l2_regression'
    
    Returns:
        Optimizer instance
    """
    if HAS_NESTED_LEARNING and use_dmgd:
        logger.info(f"Using DeepMomentumGD (lr={lr}, memory_lr={memory_lr}, mode={internal_loss_mode})")
        optimizer = DeepMomentumGD(
            params=model.parameters(),
            lr=lr,
            momentum=momentum,
            memory_lr=memory_lr,
            use_shared_memory=True,
            gradient_checkpointing=True,
            use_factorized_memory=True,
            internal_loss_mode=internal_loss_mode,
        )
        return optimizer
    else:
        logger.info(f"Using standard AdamW (lr={lr})")
        return torch.optim.AdamW(model.parameters(), lr=lr)


def create_cms_layer(
    dim: int,
    hidden_dim: Optional[int] = None,
    num_levels: int = 3,
    chunk_sizes: Optional[List[int]] = None,
) -> Optional[nn.Module]:
    """
    Create ContinuumMemorySystem layer.
    
    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (default: 4 * dim)
        num_levels: Number of frequency levels
        chunk_sizes: List of chunk sizes for each level
    
    Returns:
        CMS module or None if library not available
    """
    if not HAS_NESTED_LEARNING:
        logger.warning("CMS not available - nested-learning library not installed")
        return None
    
    if chunk_sizes is None:
        # Default: exponentially increasing (8, 32, 128)
        chunk_sizes = [2 ** (i + 3) for i in range(num_levels)]
    
    logger.info(f"Creating CMS: dim={dim}, levels={num_levels}, chunks={chunk_sizes}")
    
    return ContinuumMemorySystem(
        dim=dim,
        hidden_dim=hidden_dim,
        num_levels=num_levels,
        chunk_sizes=chunk_sizes,
    )


class NestedLearningWrapper:
    """
    Convenience wrapper for nested learning components.
    
    Usage:
        nl = NestedLearningWrapper(model, use_dmgd=True, use_cms=True)
        optimizer = nl.get_optimizer(lr=1e-3)
        
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()  # Trains both model AND memory modules
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_dmgd: bool = True,
        use_cms: bool = False,  # CMS is for model architecture, not optimizer
        internal_loss_mode: str = 'surrogate',
    ):
        self.model = model
        self.use_dmgd = use_dmgd and HAS_NESTED_LEARNING
        self.use_cms = use_cms and HAS_NESTED_LEARNING
        self.internal_loss_mode = internal_loss_mode
        
        self.optimizer = None
        self.cms_layers = []
        
        logger.info(f"NestedLearningWrapper: DMGD={self.use_dmgd}, CMS={self.use_cms}")
    
    def get_optimizer(
        self,
        lr: float = 1e-3,
        momentum: float = 0.9,
        memory_lr: float = 1e-4,
    ) -> torch.optim.Optimizer:
        """Get optimizer with nested learning features."""
        self.optimizer = create_nested_optimizer(
            model=self.model,
            lr=lr,
            momentum=momentum,
            memory_lr=memory_lr,
            use_dmgd=self.use_dmgd,
            internal_loss_mode=self.internal_loss_mode,
        )
        return self.optimizer
    
    def add_cms_to_model(self, dim: int, num_levels: int = 3) -> Optional[nn.Module]:
        """
        Create CMS layer to add to model.
        
        Note: You need to manually integrate this into your model's forward pass.
        """
        if not self.use_cms:
            return None
        
        cms = create_cms_layer(dim=dim, num_levels=num_levels)
        if cms is not None:
            self.cms_layers.append(cms)
        return cms
    
    @property
    def is_using_nested_learning(self) -> bool:
        """Check if we're using any nested learning features."""
        return self.use_dmgd or self.use_cms
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about nested learning components."""
        stats = {
            'has_nested_learning_lib': HAS_NESTED_LEARNING,
            'using_dmgd': self.use_dmgd,
            'using_cms': self.use_cms,
            'num_cms_layers': len(self.cms_layers),
        }
        
        if self.use_dmgd and self.optimizer is not None:
            stats['optimizer_type'] = type(self.optimizer).__name__
        
        return stats
