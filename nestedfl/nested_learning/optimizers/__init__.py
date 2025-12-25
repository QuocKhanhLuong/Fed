"""
Deep Optimizers: Gradient-based optimizers as associative memory modules.

This module implements optimizers from the Nested Learning perspective,
where optimizers are viewed as associative memories that compress gradients.

Key classes:
- DeepMomentumGD: Optimizer with TRUE nested optimization (memory modules trained via internal loss)
- SimpleMomentumGD: Standard momentum for comparison
- DeltaRuleMomentum: Delta-rule based momentum
- PreconditionedMomentum: Momentum with preconditioning
"""

from .deep_momentum import DeepMomentumGD, SimpleMomentumGD, MemoryMLP, SharedMemoryPool
from .delta_rule import DeltaRuleMomentum
from .preconditioned import PreconditionedMomentum
from .nested_dmgd import (
    NestedDeepMomentumGD,
    LearnedMemoryModule,
    create_meta_learning_task_distribution,
)

__all__ = [
    # Main optimizer with nested learning
    "DeepMomentumGD",
    "SimpleMomentumGD",
    "MemoryMLP",
    "SharedMemoryPool",
    # Other optimizer variants
    "DeltaRuleMomentum",
    "PreconditionedMomentum",
    # Meta-learning variant
    "NestedDeepMomentumGD",
    "LearnedMemoryModule",
    "create_meta_learning_task_distribution",
]
