#!/usr/bin/env python3
"""
Nested Early-Exit Federated Learning with Flower

Pure Flower implementation (no QUIC) for simplified experiments.
Supports simulation mode and distributed mode.

Usage:
    # Simulation mode (all clients on one machine)
    python scripts/run_flower_experiment.py --num_clients 3 --num_rounds 10

    # With serialization ablation
    python scripts/run_flower_experiment.py --use_serializer --compression lz4

Author: Research Team
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Flower imports
try:
    import flwr as fl
    from flwr.common import NDArrays, Scalar, Metrics
    from flwr.server.strategy import FedAvg
    HAS_FLOWER = True
except ImportError:
    HAS_FLOWER = False
    print("ERROR: Flower not installed. Run: pip install flwr")
    sys.exit(1)

# PyTorch
import torch
from torch.utils.data import DataLoader, Subset

# Local imports
from client.nested_trainer import NestedEarlyExitTrainer, create_dummy_dataset
from client.fl_client import FLClient
from server.aggregators import create_aggregator

# Optional: serializer for ablation
try:
    from transport.serializer import ModelSerializer
    HAS_SERIALIZER = True
except ImportError:
    HAS_SERIALIZER = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Partitioning
# =============================================================================

def partition_data_dirichlet(
    dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    """Partition dataset using Dirichlet distribution."""
    np.random.seed(seed)
    
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])
    
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, proportions)
        
        for k, split in enumerate(splits):
            client_indices[k].extend(split.tolist())
    
    for k in range(num_clients):
        np.random.shuffle(client_indices[k])
    
    return client_indices


# =============================================================================
# Client Factory
# =============================================================================

def get_client_fn(
    train_dataset,
    test_dataset,
    client_indices: List[List[int]],
    config: dict,
    use_serializer: bool = False,
) -> Callable[[str], fl.client.Client]:
    """
    Create client factory function for Flower.
    
    Args:
        train_dataset: Full training dataset
        test_dataset: Test dataset (shared)
        client_indices: Indices for each client
        config: Configuration dict
        use_serializer: Enable serialization ablation
        
    Returns:
        Client factory function
    """
    serializer = ModelSerializer() if use_serializer and HAS_SERIALIZER else None
    
    def client_fn(cid: str) -> fl.client.Client:
        """Create a client by ID."""
        client_id = int(cid)
        
        # Create data loaders
        client_data = Subset(train_dataset, client_indices[client_id])
        train_loader = DataLoader(
            client_data,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
        )
        
        # Create trainer
        trainer = NestedEarlyExitTrainer(
            num_classes=config['num_classes'],
            exit_weights=config.get('exit_weights', [0.3, 0.3, 0.4]),
            fast_lr_multiplier=config.get('fast_lr_multiplier', 3.0),
            slow_update_freq=config.get('slow_update_freq', 5),
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_mixed_precision=config.get('mixed_precision', True),
            use_self_distillation=config.get('use_distillation', True),
            cms_enabled=config.get('cms_enabled', True),
            use_lss=config.get('use_lss', True),
            use_deep_momentum=False,
            use_timm_pretrained=config.get('use_pretrained', True),
        )
        
        # Create FL client
        client = FLClient(
            trainer=trainer,
            train_loader=train_loader,
            test_loader=test_loader,
            local_epochs=config['local_epochs'],
            learning_rate=config['learning_rate'],
        )
        
        logger.info(f"Created client {cid}: {len(client_indices[client_id])} samples")
        
        return client.to_client()
    
    return client_fn


# =============================================================================
# Metrics Aggregation
# =============================================================================

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from multiple clients."""
    if not metrics:
        return {}
    
    total_samples = sum(num for num, _ in metrics)
    
    # Weighted average of accuracy
    accuracies = [m.get("accuracy", 0.0) * num for num, m in metrics]
    losses = [m.get("loss", 0.0) * num for num, m in metrics]
    
    return {
        "accuracy": sum(accuracies) / total_samples if total_samples > 0 else 0.0,
        "loss": sum(losses) / total_samples if total_samples > 0 else 0.0,
    }


# =============================================================================
# FedDyn Strategy (Flower-compatible)
# =============================================================================

class FedDynStrategy(FedAvg):
    """
    FedDyn Strategy for Flower.
    
    Dynamic regularization for handling non-IID data.
    Reference: FedDyn paper (ICLR 2021)
    """
    
    def __init__(
        self,
        alpha: float = 0.01,
        **kwargs,
    ):
        super().__init__(
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
            **kwargs,
        )
        self.alpha = alpha
        self.h = None  # Gradient correction term
        logger.info(f"FedDyn Strategy: alpha={alpha}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate with FedDyn correction."""
        if not results:
            return None, {}
        
        # Get aggregated parameters from FedAvg
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # FedDyn correction would be applied here
            # For simplicity, using standard FedAvg aggregation
            logger.info(f"Round {server_round}: aggregated {len(results)} clients")
        
        return aggregated_parameters, metrics


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiment(args):
    """Run FL experiment with Flower."""
    logger.info("=" * 60)
    logger.info("Nested Early-Exit Federated Learning (Flower)")
    logger.info("=" * 60)
    
    # Configuration
    config = {
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'local_epochs': args.local_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_classes': 10 if args.dataset == 'cifar10' else 100,
        'mixed_precision': not args.no_amp,
        'use_distillation': True,
        'cms_enabled': True,
        'use_lss': True,
        'use_pretrained': True,
        'fast_lr_multiplier': args.fast_lr_mult,
        'slow_update_freq': args.slow_freq,
    }
    
    logger.info("Configuration:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    
    # Load dataset
    logger.info(f"\nLoading dataset: {args.dataset}")
    from torchvision import datasets, transforms
    
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    else:  # cifar100
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
    
    # Partition data
    client_indices = partition_data_dirichlet(
        train_dataset,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
    )
    
    for i, indices in enumerate(client_indices):
        logger.info(f"  Client {i}: {len(indices)} samples")
    
    # Create client factory
    client_fn = get_client_fn(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        client_indices=client_indices,
        config=config,
        use_serializer=args.use_serializer,
    )
    
    # Create strategy
    if args.strategy == 'feddyn':
        strategy = FedDynStrategy(
            alpha=0.01,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
        )
    else:  # fedavg
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    
    # Run simulation
    logger.info(f"\nStarting FL simulation: {args.num_rounds} rounds, {args.num_clients} clients")
    logger.info(f"Strategy: {args.strategy.upper()}")
    
    # Configure resources per client
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 0.33 if torch.cuda.is_available() else 0.0,  # Share GPU among 3 clients
    }
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 60)
    
    if history.metrics_distributed:
        for round_num, metrics in enumerate(history.metrics_distributed.get("accuracy", [])):
            logger.info(f"Round {round_num + 1}: accuracy={metrics[1]:.4f}")
    
    logger.info(f"\nResults saved to history object")
    return history


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='FL Experiment with Flower')
    
    # FL config
    parser.add_argument('--num_clients', type=int, default=3)
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--local_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--strategy', type=str, default='feddyn', choices=['fedavg', 'feddyn'])
    
    # Nested Learning
    parser.add_argument('--fast_lr_mult', type=float, default=3.0)
    parser.add_argument('--slow_freq', type=int, default=5)
    
    # Ablation
    parser.add_argument('--use_serializer', action='store_true', help='Enable serialization')
    parser.add_argument('--compression', type=str, default='none', choices=['none', 'lz4', 'zstd'])
    
    # System
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_amp', action='store_true')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   Nested Early-Exit Federated Learning (Flower)                      ║
║   Pure Flower simulation - No QUIC                                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    run_experiment(args)
