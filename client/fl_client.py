"""
Flower Client Implementation for FL-QUIC-LoRA
Integrates MobileViT trainer with Flower's NumPyClient

Author: Research Team - FL-QUIC-LoRA Project
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, TYPE_CHECKING
import logging

# Runtime imports first
try:
    import flwr as fl
    from flwr.common import NDArrays, Scalar
    HAS_FLOWER = True
except ImportError:
    HAS_FLOWER = False
    fl = None  # type: ignore
    # Define fallback type aliases when Flower not available
    NDArrays = List[np.ndarray]
    Scalar = Union[bool, bytes, float, int, str]
    logging.warning("Flower not installed - client will not work")

from .model_trainer import MobileViTLoRATrainer, create_dummy_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FLClient(fl.client.NumPyClient if HAS_FLOWER else object):  # type: ignore
    """
    Federated Learning Client using Flower framework.
    Wraps MobileViTLoRATrainer for FL operations.
    """
    
    def __init__(
        self,
        trainer: MobileViTLoRATrainer,
        train_loader,
        test_loader,
        local_epochs: int = 3,
        learning_rate: float = 1e-3,
    ):
        """
        Initialize FL client.
        
        Args:
            trainer: Model trainer instance
            train_loader: Training data loader
            test_loader: Test data loader
            local_epochs: Number of local training epochs per round
            learning_rate: Learning rate for training
        """
        self.trainer = trainer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        
        logger.info(f"FLClient initialized: epochs={local_epochs}, lr={learning_rate}")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:  # type: ignore
        """
        Get model parameters.
        
        Args:
            config: Configuration from server
            
        Returns:
            List of NumPy arrays (model weights)
        """
        logger.info("Extracting model parameters...")
        parameters = self.trainer.get_parameters()
        logger.info(f"Extracted {len(parameters)} parameter arrays")
        return parameters
    
    def set_parameters(self, parameters: NDArrays) -> None:  # type: ignore
        """
        Set model parameters (LoRA weights).
        
        Args:
            parameters: List of NumPy arrays from server
        """
        logger.info(f"Setting {len(parameters)} parameter arrays...")
        self.trainer.set_parameters(parameters)
        logger.info("Parameters updated")
    
    def fit(
        self,
        parameters: NDArrays,  # type: ignore
        config: Dict[str, Scalar]  # type: ignore
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:  # type: ignore
        """
        Train the model locally.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration from server
            
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting local training (Round {config.get('round', '?')})")
        logger.info(f"{'='*60}")
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Lưu lại global weights trước khi set vào model (để tính FedProx)
        global_weights_copy = [np.copy(w) for w in parameters]
        
        # Get training config
        epochs = int(config.get('local_epochs', self.local_epochs))
        lr = float(config.get('learning_rate', self.learning_rate))
        mu = float(config.get('fedprox_mu', 0.01)) # Lấy từ config server gửi xuống
        
        # Train locally
        metrics = self.trainer.train(
            self.train_loader,
            epochs=epochs,
            learning_rate=lr,
            fedprox_mu=mu,
            global_weights=global_weights_copy
        )
        
        # Extract updated parameters (Adaptive Pruning)
        updated_parameters = self.trainer.get_adaptive_parameters()
        
        num_samples = int(metrics['num_samples'])
        
        # Convert metrics to Flower format
        fl_metrics: Dict[str, Scalar] = {  # type: ignore
            'loss': float(metrics['loss']),
            'accuracy': float(metrics['accuracy']),
        }
        
        logger.info(f"Local training complete: {num_samples} samples, "
                   f"loss={fl_metrics['loss']:.4f}, acc={fl_metrics['accuracy']:.4f}")
        
        return updated_parameters, num_samples, fl_metrics
    
    def evaluate(
        self,
        parameters: NDArrays,  # type: ignore
        config: Dict[str, Scalar]  # type: ignore
    ) -> Tuple[float, int, Dict[str, Scalar]]:  # type: ignore
        """
        Evaluate the model locally.
        
        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        logger.info("Evaluating model...")
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Evaluate
        metrics = self.trainer.evaluate(self.test_loader)
        
        loss = float(metrics['loss'])
        num_samples = int(metrics['num_samples'])
        
        fl_metrics: Dict[str, Scalar] = {  # type: ignore
            'accuracy': float(metrics['accuracy']),
        }
        
        logger.info(f"Evaluation: loss={loss:.4f}, acc={fl_metrics['accuracy']:.4f}")
        
        return loss, num_samples, fl_metrics


def create_fl_client(
    num_classes: int = 10,
    lora_r: int = 8,
    local_epochs: int = 3,
    learning_rate: float = 1e-3,
    use_dummy_data: bool = True,
    network_monitor: Optional[Any] = None,  # <--- NEW
) -> FLClient:
    """
    Factory function to create FL client.
    
    Args:
        num_classes: Number of output classes
        lora_r: LoRA rank
        local_epochs: Epochs per round
        learning_rate: Learning rate
        use_dummy_data: Use dummy dataset (for testing)
        network_monitor: NetworkMonitor instance
        
    Returns:
        FLClient instance
    """
    # Create trainer
    logger.info("Creating MobileViT + LoRA trainer...")
    trainer = MobileViTLoRATrainer(
        num_classes=num_classes,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,
        use_mixed_precision=True,
        network_monitor=network_monitor,  # <--- Pass network_monitor
    )
    
    # Create data loaders
    if use_dummy_data:
        logger.info("Creating dummy dataset...")
        train_loader, test_loader = create_dummy_dataset(num_samples=100)
    else:
        # TODO: Load real dataset (CIFAR-10, ImageNet, etc.)
        raise NotImplementedError("Real dataset loading not implemented yet")
    
    # Create FL client
    client = FLClient(
        trainer=trainer,
        train_loader=train_loader,
        test_loader=test_loader,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
    )
    
    logger.info("FL Client created successfully")
    return client


# Example usage
if __name__ == "__main__":
    if not HAS_FLOWER:
        logger.error("Flower not installed! Install with: pip install flwr")
        exit(1)
    
    logger.info("="*60)
    logger.info("Flower Client Demo")
    logger.info("="*60)
    
    # Create client
    client = create_fl_client(
        num_classes=10,
        lora_r=4,
        local_epochs=2,
        use_dummy_data=True,
    )
    
    # Simulate FL round
    logger.info("\nSimulating FL round...")
    
    # Get initial parameters
    initial_params = client.get_parameters({})
    logger.info(f"Initial parameters: {len(initial_params)} arrays")
    
    # Simulate training
    config = {'round': 1, 'local_epochs': 2, 'learning_rate': 1e-3}
    updated_params, num_samples, metrics = client.fit(initial_params, config)
    
    logger.info(f"\nTraining results:")
    logger.info(f"  Samples: {num_samples}")
    logger.info(f"  Loss: {metrics['loss']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # Simulate evaluation
    loss, num_samples, eval_metrics = client.evaluate(updated_params, {})
    
    logger.info(f"\nEvaluation results:")
    logger.info(f"  Loss: {loss:.4f}")
    logger.info(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Flower Client demo completed!")
    logger.info("="*60)
