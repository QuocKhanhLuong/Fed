"""Client components for Federated Learning"""

from .nested_trainer import NestedEarlyExitTrainer, create_dummy_dataset
from .fl_client import FLClient, create_fl_client

__all__ = [
    'NestedEarlyExitTrainer',
    'create_dummy_dataset',
    'FLClient',
    'create_fl_client',
]
