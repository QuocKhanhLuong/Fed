"""
Custom Flower Strategy for FL-QUIC-LoRA
Implements FedAvg aggregation with custom logic

Author: Research Team - FL-QUIC-LoRA Project
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging

try:
    import flwr as fl
    from flwr.common import (
        FitRes,
        Parameters,
        Scalar,
        parameters_to_ndarrays,
        ndarrays_to_parameters,
    )
    from flwr.server.client_proxy import ClientProxy
    HAS_FLOWER = True
except ImportError:
    HAS_FLOWER = False
    logging.warning("Flower not installed")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FLStrategy(fl.server.strategy.FedAvg if HAS_FLOWER else object):
    """
    Custom Federated Averaging Strategy.
    Extends Flower's FedAvg with custom logic for FL-QUIC-LoRA.
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ):
        """
        Initialize custom strategy.
        
        Args:
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum available clients
            evaluate_fn: Server-side evaluation function
            on_fit_config_fn: Function to configure training
            on_evaluate_config_fn: Function to configure evaluation
            accept_failures: Whether to accept failures
            initial_parameters: Initial global model parameters
        """
        if not HAS_FLOWER:
            logger.error("Flower not installed!")
            return
        
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        
        self.current_round = 0
        
        logger.info(f"FLStrategy initialized: min_fit={min_fit_clients}, "
                   f"min_eval={min_evaluate_clients}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate training results from clients.
        
        Args:
            server_round: Current round number
            results: List of (client, fit_result) tuples
            failures: List of failures
            
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        self.current_round = server_round
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {server_round}: Aggregating {len(results)} client updates")
        logger.info(f"{'='*60}")
        
        if failures:
            logger.warning(f"  {len(failures)} clients failed")
        
        # Call parent's aggregation (FedAvg)
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # Calculate additional metrics
            total_samples = sum(r.num_examples for _, r in results)
            avg_loss = np.mean([r.metrics.get('loss', 0.0) for _, r in results])
            avg_accuracy = np.mean([r.metrics.get('accuracy', 0.0) for _, r in results])
            
            logger.info(f"Aggregation complete:")
            logger.info(f"  Clients: {len(results)}")
            logger.info(f"  Total samples: {total_samples}")
            logger.info(f"  Avg loss: {avg_loss:.4f}")
            logger.info(f"  Avg accuracy: {avg_accuracy:.4f}")
            
            # Add custom metrics
            metrics['total_samples'] = total_samples
            metrics['avg_loss'] = avg_loss
            metrics['avg_accuracy'] = avg_accuracy
            metrics['num_clients'] = len(results)
        
        return aggregated_parameters, metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.
        
        Args:
            server_round: Current round number
            results: List of (client, eval_result) tuples
            failures: List of failures
            
        Returns:
            Tuple of (aggregated_loss, metrics)
        """
        logger.info(f"Aggregating evaluation from {len(results)} clients...")
        
        # Call parent's aggregation
        loss_aggregated, metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        if results:
            avg_accuracy = np.mean([r.metrics.get('accuracy', 0.0) for _, r in results])
            metrics['avg_accuracy'] = avg_accuracy
            
            logger.info(f"Evaluation aggregated: loss={loss_aggregated:.4f}, "
                       f"accuracy={avg_accuracy:.4f}")
        
        return loss_aggregated, metrics


def create_fit_config(server_round: int, local_epochs: int = 3) -> Dict[str, Scalar]:
    """
    Create configuration for client training.
    
    Args:
        server_round: Current round number
        local_epochs: Number of local epochs
        
    Returns:
        Configuration dictionary
    """
    config = {
        'round': server_round,
        'local_epochs': local_epochs,
        'learning_rate': 1e-3,  # Could be adaptive
    }
    
    # Adaptive learning rate (optional)
    if server_round > 5:
        config['learning_rate'] = 5e-4  # Reduce after 5 rounds
    if server_round > 10:
        config['learning_rate'] = 1e-4  # Further reduce
    
    return config


def create_evaluate_config(server_round: int) -> Dict[str, Scalar]:
    """
    Create configuration for client evaluation.
    
    Args:
        server_round: Current round number
        
    Returns:
        Configuration dictionary
    """
    return {
        'round': server_round,
    }


def create_strategy(
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    local_epochs: int = 3,
) -> FLStrategy:
    """
    Factory function to create FL strategy.
    
    Args:
        min_fit_clients: Minimum clients for training
        min_evaluate_clients: Minimum clients for evaluation
        min_available_clients: Minimum available clients
        local_epochs: Epochs per round
        
    Returns:
        FLStrategy instance
    """
    if not HAS_FLOWER:
        raise RuntimeError("Flower not installed!")
    
    strategy = FLStrategy(
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=lambda round: create_fit_config(round, local_epochs),
        on_evaluate_config_fn=create_evaluate_config,
        accept_failures=True,
    )
    
    logger.info("FL Strategy created")
    return strategy


# Example usage
if __name__ == "__main__":
    if not HAS_FLOWER:
        logger.error("Flower not installed! Install with: pip install flwr")
        exit(1)
    
    logger.info("="*60)
    logger.info("Flower Strategy Demo")
    logger.info("="*60)
    
    # Create strategy
    strategy = create_strategy(
        min_fit_clients=2,
        min_evaluate_clients=2,
        local_epochs=3,
    )
    
    logger.info(f"\nStrategy configuration:")
    logger.info(f"  Fraction fit: {strategy.fraction_fit}")
    logger.info(f"  Min fit clients: {strategy.min_fit_clients}")
    logger.info(f"  Min available clients: {strategy.min_available_clients}")
    
    # Test config generation
    logger.info(f"\nFit config for round 1: {create_fit_config(1)}")
    logger.info(f"Fit config for round 6: {create_fit_config(6)}")
    logger.info(f"Fit config for round 11: {create_fit_config(11)}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Strategy demo completed!")
    logger.info("="*60)
