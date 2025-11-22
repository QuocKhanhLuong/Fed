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

from .model_trainer import MobileViTLoRATrainer, create_dummy_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FLClient(fl.client.NumPyClient if HAS_FLOWER else object):
    def __init__(self, trainer: MobileViTLoRATrainer, train_loader, test_loader, local_epochs=3, learning_rate=1e-3):
        self.trainer = trainer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        logger.info(f"FLClient init: epochs={local_epochs}, lr={learning_rate}")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.trainer.get_parameters()
    
    def set_parameters(self, parameters: NDArrays) -> None:
        self.trainer.set_parameters(parameters)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        logger.info(f"Starting local training (Round {config.get('round', '?')})")
        self.set_parameters(parameters)
        global_weights_copy = [np.copy(w) for w in parameters]
        
        metrics = self.trainer.train(
            self.train_loader,
            epochs=int(config.get('local_epochs', self.local_epochs)),
            learning_rate=float(config.get('learning_rate', self.learning_rate)),
            fedprox_mu=float(config.get('fedprox_mu', 0.01)),
            global_weights=global_weights_copy
        )
        
        updated_parameters = self.trainer.get_adaptive_parameters()
        num_samples = int(metrics['num_samples'])
        fl_metrics = {'loss': float(metrics['loss']), 'accuracy': float(metrics['accuracy'])}
        
        logger.info(f"Training complete: loss={fl_metrics['loss']:.4f}, acc={fl_metrics['accuracy']:.4f}")
        return updated_parameters, num_samples, fl_metrics
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        metrics = self.trainer.evaluate(self.test_loader)
        loss = float(metrics['loss'])
        num_samples = int(metrics['num_samples'])
        fl_metrics = {'accuracy': float(metrics['accuracy'])}
        
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
    trainer = MobileViTLoRATrainer(
        num_classes=num_classes,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,
        use_mixed_precision=True,
        network_monitor=network_monitor,
        use_sam=use_sam,
        use_tta=use_tta,
    )
    
    if train_loader is None:
        train_loader, dummy_test = create_dummy_dataset(num_samples=100, num_classes=num_classes)
        if test_loader is None: test_loader = dummy_test
    
    if val_loader is None: eval_loader = train_loader
    else: eval_loader = val_loader
    
    if test_loader is None: test_loader = eval_loader
    
    return FLClient(trainer, train_loader, test_loader, local_epochs, learning_rate)


if __name__ == "__main__":
    if not HAS_FLOWER:
        logger.error("Flower not installed")
        exit(1)
    
    train_loader, test_loader = create_dummy_dataset(num_samples=100, num_classes=10)
    client = create_fl_client(num_classes=10, lora_r=4, local_epochs=2, train_loader=train_loader, test_loader=test_loader)
    
    logger.info("Simulating FL round...")
    initial_params = client.get_parameters({})
    config = {'round': 1, 'local_epochs': 2, 'learning_rate': 1e-3}
    updated_params, num_samples, metrics = client.fit(initial_params, config)
    loss, num_samples, eval_metrics = client.evaluate(updated_params, {})
    
    logger.info("Demo completed")
