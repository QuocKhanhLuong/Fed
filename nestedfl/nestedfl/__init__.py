"""
nestedfl: FedEEP — Federated Early-Exit with Progressive phases
"""

__version__ = "2.0.0"

try:
    from nestedfl.nested_trainer import NestedEarlyExitTrainer, ContinuumMemorySystem
except ImportError:
    pass

__all__ = ["NestedEarlyExitTrainer", "ContinuumMemorySystem"]
