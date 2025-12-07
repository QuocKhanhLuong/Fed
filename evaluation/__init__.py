"""
FL Evaluation Framework

IEEE-standard metrics for Federated Learning with Early-Exit Networks.
"""

from .fl_evaluator import (
    FLEvaluator,
    ExperimentConfig,
    RoundMetrics,
    ExperimentComparison,
)

__all__ = [
    'FLEvaluator',
    'ExperimentConfig', 
    'RoundMetrics',
    'ExperimentComparison',
]
