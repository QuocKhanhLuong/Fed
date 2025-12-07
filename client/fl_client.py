"""
FL Client for Federated Learning with Early-Exit Networks

This module provides the Flower-compatible FL client interface.

Author: Research Team - FL-QUIC-LoRA Project
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

try:
    import flwr as fl
    from flwr.common import NDArrays, Scalar
    HAS_FLOWER = True
except ImportError:
    HAS_FLOWER = False
    fl = None
    NDArrays = List[np.ndarray]
    Scalar = Union[bool, bytes, float, int, str]

try:
    from torch.utils.data import DataLoader
except ImportError:
    DataLoader = Any

from .early_exit_trainer import EarlyExitTrainer, create_dummy_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FLClient(fl.client.NumPyClient if HAS_FLOWER else object):
    """
    Flower-compatible FL Client with Early-Exit trainer.
    
    Implements NumPyClient interface for local training and evaluation.
    """
    
    def __init__(
        self, 
        trainer: EarlyExitTrainer, 
        train_loader, 
        test_loader, 
        local_epochs: int = 3, 
        learning_rate: float = 1e-3
    ):
        self.trainer = trainer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        logger.info(f"FLClient init: epochs={local_epochs}, lr={learning_rate}")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters as NumPy arrays."""
        return self.trainer.get_parameters()
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from NumPy arrays."""
        self.trainer.set_parameters(parameters)
    
    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Local training with Early-Exit model.
        
        Args:
            parameters: Global model weights
            config: Training configuration from server
            
        Returns:
            (updated_weights, num_samples, metrics)
        """
        logger.info(f"Starting local training (Round {config.get('round', '?')})")
        self.set_parameters(parameters)
        global_weights_copy = [np.copy(w) for w in parameters]
        
        metrics = self.trainer.train(
            self.train_loader,
            epochs=int(config.get('local_epochs', self.local_epochs)),
            learning_rate=float(config.get('learning_rate', self.learning_rate)),
            fedprox_mu=float(config.get('fedprox_mu', 0.0)),
            global_weights=global_weights_copy
        )
        
        updated_parameters = self.trainer.get_parameters()
        num_samples = int(metrics['num_samples'])
        fl_metrics = {
            'loss': float(metrics['loss']), 
            'accuracy': float(metrics['accuracy'])
        }
        
        logger.info(f"Training complete: loss={fl_metrics['loss']:.4f}, acc={fl_metrics['accuracy']:.4f}")
        return updated_parameters, num_samples, fl_metrics
    
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate with Early-Exit inference.
        
        Args:
            parameters: Model weights
            config: Evaluation configuration
            
        Returns:
            (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)
        threshold = float(config.get('exit_threshold', 0.8))
        metrics = self.trainer.evaluate(self.test_loader, threshold=threshold)
        
        loss = float(metrics['loss'])
        num_samples = int(metrics['num_samples'])
        fl_metrics = {
            'accuracy': float(metrics['accuracy']),
            'avg_exit': float(metrics.get('avg_exit', 2.0)),
        }
        
        logger.info(f"Evaluation: loss={loss:.4f}, acc={fl_metrics['accuracy']:.4f}")
        return loss, num_samples, fl_metrics


def create_fl_client(
    num_classes: int = 10,
    local_epochs: int = 3,
    learning_rate: float = 1e-3,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    use_mixed_precision: bool = True,
    # Legacy parameters (ignored for backwards compatibility)
    lora_r: int = 8,
    network_monitor: Optional[Any] = None,
    use_sam: bool = False,
    use_tta: bool = False,
) -> FLClient:
    """
    Factory function to create FL client with Early-Exit trainer.
    
    Args:
        num_classes: Number of output classes
        local_epochs: Local training epochs per round
        learning_rate: Learning rate
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        test_loader: Test data loader
        use_mixed_precision: Enable AMP
        
    Returns:
        Configured FLClient instance
    """
    trainer = EarlyExitTrainer(
        num_classes=num_classes,
        use_mixed_precision=use_mixed_precision,
    )
    
    if train_loader is None:
        train_loader, dummy_test = create_dummy_dataset(
            num_samples=100, 
            num_classes=num_classes
        )
        if test_loader is None: 
            test_loader = dummy_test
    
    if val_loader is None: 
        eval_loader = train_loader
    else: 
        eval_loader = val_loader
    
    if test_loader is None: 
        test_loader = eval_loader
    
    return FLClient(trainer, train_loader, test_loader, local_epochs, learning_rate)


if __name__ == "__main__":
    if not HAS_FLOWER:
        logger.error("Flower not installed")
        exit(1)
    
    train_loader, test_loader = create_dummy_dataset(num_samples=100, num_classes=10)
    client = create_fl_client(
        num_classes=10, 
        local_epochs=2, 
        train_loader=train_loader, 
        test_loader=test_loader
    )
    
    logger.info("Simulating FL round...")
    initial_params = client.get_parameters({})
    config = {'round': 1, 'local_epochs': 2, 'learning_rate': 1e-3}
    updated_params, num_samples, metrics = client.fit(initial_params, config)
    loss, num_samples, eval_metrics = client.evaluate(updated_params, {})
    
    logger.info("Demo completed")
