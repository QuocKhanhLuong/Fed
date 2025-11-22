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

# PyTorch imports
try:
    from torch.utils.data import DataLoader
except ImportError:
    DataLoader = Any  # type: ignore

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
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    network_monitor: Optional[Any] = None,
    use_sam: bool = False,
    use_tta: bool = False,
) -> FLClient:
    """
    Factory function to create FL client with real datasets.
    
    Args:
        num_classes: Number of output classes
        lora_r: LoRA rank
        local_epochs: Epochs per round
        learning_rate: Learning rate
        train_loader: Training data loader (REQUIRED)
        val_loader: Validation data loader (optional, uses train if None)
        test_loader: Test data loader (optional, uses val if None)
        network_monitor: NetworkMonitor instance
        use_sam: Enable SAM optimizer
        use_tta: Enable Test-Time Adaptation
        
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
        network_monitor=network_monitor,
        use_sam=use_sam,
        use_tta=use_tta,
    )
    
    # Use provided loaders or create dummy data as fallback
    if train_loader is None:
        logger.warning("No train_loader provided - creating dummy dataset for testing")
        train_loader, dummy_test = create_dummy_dataset(num_samples=100, num_classes=num_classes)
        if test_loader is None:
            test_loader = dummy_test
    
    # Use val_loader for evaluation if provided, otherwise use train_loader
    if val_loader is None:
        logger.info("No val_loader provided - using train_loader for local evaluation")
        eval_loader = train_loader
    else:
        eval_loader = val_loader
    
    # Use test_loader if provided, otherwise use eval_loader
    if test_loader is None:
        logger.info("No test_loader provided - using val_loader for testing")
        test_loader = eval_loader
    
    # Create FL client
    client = FLClient(
        trainer=trainer,
        train_loader=train_loader,
        test_loader=test_loader,  # Use test_loader for global evaluation
        local_epochs=local_epochs,
        learning_rate=learning_rate,
    )
    
    logger.info(f"FL Client created: {num_classes} classes, LoRA-r={lora_r}, SAM={use_sam}, TTA={use_tta}")
    return client


# Example usage
if __name__ == "__main__":
    if not HAS_FLOWER:
        logger.error("Flower not installed! Install with: pip install flwr")
        exit(1)
    
    logger.info("="*60)
    logger.info("Flower Client Demo")
    logger.info("="*60)
    
    # Create dummy data for demo
    from .model_trainer import create_dummy_dataset
    train_loader, test_loader = create_dummy_dataset(num_samples=100, num_classes=10)
    
    # Create client
    client = create_fl_client(
        num_classes=10,
        lora_r=4,
        local_epochs=2,
        train_loader=train_loader,
        test_loader=test_loader,
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
