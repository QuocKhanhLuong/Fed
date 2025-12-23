"""
nestedfl: Nested Early-Exit Federated Learning with Flower 1.18+

A complete Flower application for federated learning with:
- Early-Exit Networks (MobileViTv2)
- Nested Learning (Fast/Slow parameter separation)
- Local Surprise Signal (LSS)
- Continuum Memory System (CMS)
"""

__version__ = "1.0.0"

# Import main components
from nestedfl.task import get_model, get_trainer, load_data, train, test

# Optional imports
try:
    from nestedfl.nested_trainer import NestedEarlyExitTrainer
except ImportError:
    pass

try:
    from nestedfl.aggregators import FedAvgAggregator, FedDynAggregator
except ImportError:
    pass

try:
    from nestedfl.serializer import ModelSerializer
except ImportError:
    pass

__all__ = [
    "get_model",
    "get_trainer", 
    "load_data",
    "train",
    "test",
]
