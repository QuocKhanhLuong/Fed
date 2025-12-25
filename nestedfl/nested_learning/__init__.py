"""
Nested Learning: The Illusion of Deep Learning Architectures

Implementation of the NeurIPS 2025 paper by Behrouz et al.

This implementation provides:
1. DeepMomentumGD - Optimizer with TRUE nested optimization (memory modules trained)
2. ContinuumMemorySystem - Multi-frequency memory with different update timescales
3. FullyNestedCMS - Paper-exact nested composition (Equation 30)

Key concepts:
- Outer loop: Model parameter updates via learned optimizer
- Inner loop: Memory module training via internal loss
- Multi-frequency: CMS levels update at different rates (fast vs slow memory)
"""

__version__ = "0.2.0"

from nested_learning.optimizers import (
    DeepMomentumGD,
    SimpleMomentumGD,
    DeltaRuleMomentum,
    PreconditionedMomentum,
)
from nested_learning.memory import (
    AssociativeMemory,
    ContinuumMemorySystem,
    FullyNestedCMS,
    LinearAttention,
)

__all__ = [
    # Optimizers
    "DeepMomentumGD",
    "SimpleMomentumGD",
    "DeltaRuleMomentum",
    "PreconditionedMomentum",
    # Memory
    "AssociativeMemory",
    "ContinuumMemorySystem",
    "FullyNestedCMS",
    "LinearAttention",
]
