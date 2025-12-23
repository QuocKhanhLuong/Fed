#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nested Early-Exit Federated Learning Experiment Runner

This script implements the experimental evaluation for:
    "Difficulty-Aware Federated Learning with Nested Early-Exit Networks"
    IEEE Transactions on Mobile Computing, 2025

The experiment supports two modes:
    1. Simulation Mode: All clients run on a single GPU (for development)
    2. Distributed Mode: Server + real Jetson Nano clients via QUIC

Algorithm: Nested Federated Optimization with Early-Exit Networks
─────────────────────────────────────────────────────────────────
Input: Dataset D, num_clients K, rounds T, local_epochs E
Output: Trained global model θ*

1. Initialize θ₀ from pretrained MobileViTv2
2. for t = 1 to T do
3.     S_t ← sample(K clients)
4.     for k ∈ S_t in parallel do
5.         θ_k ← LocalTrain(θ_{t-1}, D_k, E)    # Nested optimization
6.     θ_t ← Aggregate({θ_k : k ∈ S_t})         # FedAvg/FedProx
7. return θ_T

Usage:
    # Simulation mode (all on server)
    python scripts/run_experiment.py --mode simulation --num_clients 10

    # Distributed mode (server side)  
    python scripts/run_experiment.py --mode server --port 4433

    # Distributed mode (Jetson client)
    python scripts/run_experiment.py --mode client --server_ip 192.168.1.100

Author: Research Team
Date: 2025
"""

import argparse
import copy
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Local imports
from client.nested_trainer import NestedEarlyExitTrainer, create_dummy_dataset
from client.data_manager import load_dataset
from utils.torch_compat import get_device_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Experiment Configuration (Table I in paper)
# =============================================================================

DEFAULT_CONFIG = {
    # Federated Learning
    'num_clients': 3,            # K: Number of clients (default: 3 for parallel)
    'num_rounds': 50,            # T: Communication rounds
    'client_fraction': 1.0,      # C: Fraction of clients per round
    'local_epochs': 5,           # E: Local epochs
    'parallel_clients': 3,       # Max clients to train in parallel (RTX 4070: 3)
    
    # Dataset
    'dataset': 'cifar100',       # Dataset name
    'partition': 'dirichlet',    # Data partition: iid, dirichlet
    'alpha': 0.5,                # Dirichlet concentration (lower = more non-IID)
    'batch_size': 32,            # Local batch size
    
    # Model
    'num_classes': 100,          # Number of output classes
    'use_pretrained': True,      # Use ImageNet pretrained backbone
    
    # Optimization
    'learning_rate': 1e-3,       # η: Base learning rate
    'weight_decay': 0.01,        # λ: L2 regularization
    'fedprox_mu': 0.01,          # μ: FedProx regularization
    
    # Nested Learning (Algorithm 1)
    'fast_lr_multiplier': 3.0,   # η_fast = η × multiplier
    'slow_update_freq': 5,       # K: Update slow weights every K steps
    
    # Early Exit
    'exit_weights': [0.3, 0.3, 0.4],  # α_k: Loss weights for exits
    'exit_threshold': 0.8,            # τ: Confidence threshold
    
    # Self-Distillation
    'use_distillation': True,    # Enable knowledge distillation
    'distillation_weight': 0.1,  # β: Distillation loss weight
    'distillation_temp': 3.0,    # T: Temperature
    
    # CMS (Continuum Memory System) - Extended
    'cms_enabled': True,         # Enable memory system
    'cms_weight': 0.001,         # Memory regularization weight
    'cms_num_levels': 4,         # NEW: Number of memory levels
    
    # NEW: Local Surprise Signal (LSS)
    'use_lss': True,             # Enable sample importance weighting
    'lss_temperature': 1.0,      # Temperature for weighting
    
    # NEW: Deep Momentum GD (disabled by default)
    'use_deep_momentum': False,  # Enable MLP-based momentum
    
    # System
    'device': 'auto',            # cuda, cpu, or auto
    'mixed_precision': True,     # Use FP16
    'seed': 42,                  # Random seed
}


# =============================================================================
# Data Partitioning (Section IV-A)
# =============================================================================

def partition_data(
    dataset,
    num_clients: int,
    partition_type: str = 'dirichlet',
    alpha: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    """
    Partition dataset among clients.
    
    Implements statistical heterogeneity via Dirichlet distribution:
        p_k ~ Dir(α)
        
    where α controls the heterogeneity level:
        - α → ∞: IID distribution
        - α → 0: Each client has single class
        
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients K
        partition_type: 'iid' or 'dirichlet'
        alpha: Dirichlet concentration parameter
        seed: Random seed
        
    Returns:
        List of sample indices for each client
    """
    np.random.seed(seed)
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])
    
    num_samples = len(labels)
    num_classes = len(np.unique(labels))
    
    if partition_type == 'iid':
        # IID: Random shuffle and split
        indices = np.random.permutation(num_samples)
        splits = np.array_split(indices, num_clients)
        return [list(s) for s in splits]
    
    elif partition_type == 'dirichlet':
        # Non-IID: Dirichlet distribution
        client_indices = [[] for _ in range(num_clients)]
        
        for c in range(num_classes):
            class_indices = np.where(labels == c)[0]
            np.random.shuffle(class_indices)
            
            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = proportions / proportions.sum()
            
            # Assign samples to clients
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            splits = np.split(class_indices, proportions)
            
            for k, split in enumerate(splits):
                client_indices[k].extend(split.tolist())
        
        # Shuffle each client's data
        for k in range(num_clients):
            np.random.shuffle(client_indices[k])
        
        return client_indices
    
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")


# =============================================================================
# Federated Aggregation (Section IV-B)
# =============================================================================

def federated_averaging(
    global_params: List[np.ndarray],
    client_params: List[List[np.ndarray]],
    client_weights: List[int],
) -> List[np.ndarray]:
    """
    FedAvg aggregation (McMahan et al., 2017).
    
    θ_t = Σ (n_k / n) · θ_k
    
    where n_k is the number of samples at client k.
    """
    total_weight = sum(client_weights)
    
    aggregated = []
    for param_idx in range(len(global_params)):
        weighted_sum = np.zeros_like(global_params[param_idx])
        for client_idx, params in enumerate(client_params):
            weight = client_weights[client_idx] / total_weight
            weighted_sum += weight * params[param_idx]
        aggregated.append(weighted_sum)
    
    return aggregated


def fedprox_aggregation(
    global_params: List[np.ndarray],
    client_params: List[List[np.ndarray]],
    client_weights: List[int],
    mu: float = 0.01,
) -> List[np.ndarray]:
    """
    FedProx aggregation with proximal regularization.
    
    Same as FedAvg for aggregation; the regularization is applied
    during local training via the proximal term:
    
    L_prox = L + (μ/2) ||θ - θ_global||²
    """
    return federated_averaging(global_params, client_params, client_weights)


# =============================================================================
# Experiment Runner (Algorithm 2)
# =============================================================================

class FederatedExperiment:
    """
    Federated Learning Experiment Runner.
    
    Implements the main FL training loop with:
    - Nested Early-Exit model architecture
    - Multi-timescale optimization (fast/slow weights)
    - Continuum Memory System for catastrophic forgetting
    - Self-distillation for exit consistency
    
    Attributes:
        config (dict): Experiment configuration
        device (torch.device): Computation device
        global_model (NestedEarlyExitTrainer): Global model
    """
    
    def __init__(self, config: dict):
        """Initialize experiment with configuration."""
        self.config = config
        self.setup_device()
        self.setup_logging()
        
    def setup_device(self):
        """Setup computation device."""
        if self.config['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config['device'])
        
        logger.info(f"Device: {self.device}")
        
        # Log device info
        info = get_device_info()
        for k, v in info.items():
            logger.info(f"  {k}: {v}")
    
    def setup_logging(self):
        """Setup experiment logging directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_name = f"{self.config['dataset']}_{self.config['num_clients']}clients_{timestamp}"
        self.log_dir = PROJECT_ROOT / 'experiments' / 'logs' / self.exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Experiment: {self.exp_name}")
        logger.info(f"Log directory: {self.log_dir}")
    
    def load_data(self) -> Tuple[any, any, List[List[int]]]:
        """
        Load and partition dataset.
        
        Returns:
            train_dataset: Training dataset
            test_dataset: Test dataset  
            client_indices: List of sample indices per client
        """
        logger.info(f"Loading dataset: {self.config['dataset']}")
        
        # Load dataset
        try:
            from torchvision import datasets, transforms
            
            if self.config['dataset'] == 'cifar10':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                        (0.2470, 0.2435, 0.2616)),
                ])
                train_dataset = datasets.CIFAR10(
                    root='./data', train=True, download=True, transform=transform
                )
                test_dataset = datasets.CIFAR10(
                    root='./data', train=False, download=True, transform=transform
                )
                self.config['num_classes'] = 10
                
            elif self.config['dataset'] == 'cifar100':
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408),
                                        (0.2675, 0.2565, 0.2761)),
                ])
                train_dataset = datasets.CIFAR100(
                    root='./data', train=True, download=True, transform=transform
                )
                test_dataset = datasets.CIFAR100(
                    root='./data', train=False, download=True, transform=transform
                )
                self.config['num_classes'] = 100
            else:
                raise ValueError(f"Unknown dataset: {self.config['dataset']}")
                
        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}")
            logger.info("Using dummy dataset for testing")
            train_loader, test_loader = create_dummy_dataset(
                num_samples=500,
                num_classes=self.config['num_classes'],
                batch_size=self.config['batch_size'],
            )
            return train_loader.dataset, test_loader.dataset, None
        
        # Partition data
        client_indices = partition_data(
            train_dataset,
            num_clients=self.config['num_clients'],
            partition_type=self.config['partition'],
            alpha=self.config['alpha'],
            seed=self.config['seed'],
        )
        
        # Log partition statistics
        for k, indices in enumerate(client_indices):
            logger.info(f"  Client {k}: {len(indices)} samples")
        
        return train_dataset, test_dataset, client_indices
    
    def create_trainer(self) -> NestedEarlyExitTrainer:
        """Create Nested Early-Exit trainer instance."""
        return NestedEarlyExitTrainer(
            num_classes=self.config['num_classes'],
            exit_weights=self.config['exit_weights'],
            fast_lr_multiplier=self.config['fast_lr_multiplier'],
            slow_update_freq=self.config['slow_update_freq'],
            device=str(self.device),
            use_mixed_precision=self.config['mixed_precision'],
            use_self_distillation=self.config['use_distillation'],
            distillation_weight=self.config['distillation_weight'],
            distillation_temp=self.config['distillation_temp'],
            cms_enabled=self.config['cms_enabled'],
            cms_weight=self.config['cms_weight'],
            cms_num_levels=self.config['cms_num_levels'],
            use_lss=self.config['use_lss'],
            lss_temperature=self.config['lss_temperature'],
            use_deep_momentum=self.config['use_deep_momentum'],
            use_timm_pretrained=self.config['use_pretrained'],
        )
    
    def run_simulation(self):
        """
        Run FL simulation on single GPU with parallel client training.
        
        Algorithm 2: Simulated Federated Training (Parallel)
        ─────────────────────────────────────────────────────
        Clients train in parallel using ThreadPoolExecutor.
        RTX 4070 (12GB VRAM) supports 3 concurrent models.
        """
        logger.info("=" * 60)
        logger.info("Starting Federated Learning Simulation (PARALLEL MODE)")
        logger.info("=" * 60)
        
        # Load data
        train_dataset, test_dataset, client_indices = self.load_data()
        
        if client_indices is None:
            logger.error("Failed to partition data")
            return
        
        # Create global model
        global_trainer = self.create_trainer()
        global_params = global_trainer.get_parameters()
        
        # Test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2,
        )
        
        # Training history
        history = {
            'round': [],
            'loss': [],
            'accuracy': [],
            'exit_distribution': [],
        }
        
        # Get parallel config
        max_parallel = self.config.get('parallel_clients', 3)
        logger.info(f"Parallel training enabled: max {max_parallel} clients simultaneously")
        
        # ═══════════════════════════════════════════════════════════════
        # Helper function for parallel client training
        # ═══════════════════════════════════════════════════════════════
        def train_single_client(client_id: int) -> Tuple[int, List[np.ndarray], int, Dict]:
            """Train a single client and return results."""
            # Create client data loader
            client_data = Subset(train_dataset, client_indices[client_id])
            client_loader = DataLoader(
                client_data,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=0,  # No workers in thread to avoid fork issues
            )
            
            # Create client trainer (each thread gets its own model)
            client_trainer = self.create_trainer()
            client_trainer.set_parameters(copy.deepcopy(global_params))
            
            # Local training
            train_metrics = client_trainer.train(
                client_loader,
                epochs=self.config['local_epochs'],
                learning_rate=self.config['learning_rate'],
                fedprox_mu=self.config['fedprox_mu'],
                global_weights=global_params,
            )
            
            # Get parameters before cleanup
            params = client_trainer.get_parameters()
            num_samples = len(client_indices[client_id])
            
            # Clean up
            del client_trainer
            torch.cuda.empty_cache()
            
            return client_id, params, num_samples, train_metrics
        
        # Main FL loop
        for round_idx in range(self.config['num_rounds']):
            round_start = time.time()
            
            logger.info(f"\n{'─' * 60}")
            logger.info(f"Round {round_idx + 1}/{self.config['num_rounds']}")
            logger.info(f"{'─' * 60}")
            
            # Sample clients
            num_selected = max(1, int(self.config['client_fraction'] * self.config['num_clients']))
            selected_clients = np.random.choice(
                self.config['num_clients'], 
                num_selected, 
                replace=False
            )
            
            # ═══════════════════════════════════════════════════════════════
            # PARALLEL LOCAL TRAINING
            # ═══════════════════════════════════════════════════════════════
            client_params = []
            client_weights = []
            
            logger.info(f"Training {len(selected_clients)} clients in parallel (max {max_parallel} concurrent)...")
            
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                # Submit all client training jobs
                futures = {
                    executor.submit(train_single_client, client_id): client_id 
                    for client_id in selected_clients
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    client_id = futures[future]
                    try:
                        cid, params, num_samples, train_metrics = future.result()
                        client_params.append(params)
                        client_weights.append(num_samples)
                        logger.info(f"  ✓ Client {cid}: loss={train_metrics['loss']:.4f}, "
                                   f"acc={train_metrics['accuracy']:.4f}")
                    except Exception as e:
                        logger.error(f"  ✗ Client {client_id} failed: {e}")
            
            # Aggregate
            if client_params:
                global_params = fedprox_aggregation(
                    global_params,
                    client_params,
                    client_weights,
                    mu=self.config['fedprox_mu'],
                )
                global_trainer.set_parameters(global_params)
            
            # Evaluate
            eval_metrics = global_trainer.evaluate(
                test_loader,
                threshold=self.config['exit_threshold'],
            )
            
            round_time = time.time() - round_start
            
            logger.info(f"\nRound {round_idx + 1} Summary:")
            logger.info(f"  Test Loss: {eval_metrics['loss']:.4f}")
            logger.info(f"  Test Accuracy: {eval_metrics['accuracy']:.4f}")
            logger.info(f"  Exit Distribution: {eval_metrics['exit_distribution']}")
            logger.info(f"  Round Time: {round_time:.1f}s")
            
            # Record history
            history['round'].append(round_idx + 1)
            history['loss'].append(eval_metrics['loss'])
            history['accuracy'].append(eval_metrics['accuracy'])
            history['exit_distribution'].append(eval_metrics['exit_distribution'])
        
        # Save results
        self.save_results(history)
        
        logger.info("\n" + "=" * 60)
        logger.info("Experiment Completed!")
        logger.info(f"Best Accuracy: {max(history['accuracy']):.4f}")
        logger.info(f"Results saved to: {self.log_dir}")
        logger.info("=" * 60)
    
    def save_results(self, history: dict):
        """Save experiment results."""
        import json
        
        # Save config
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save history
        history_path = self.log_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Results saved to {self.log_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Nested Early-Exit Federated Learning Experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Mode
    parser.add_argument('--mode', type=str, default='simulation',
                        choices=['simulation', 'server', 'client'],
                        help='Experiment mode')
    
    # FL Configuration
    parser.add_argument('--num_clients', type=int, default=3,
                        help='Number of clients K')
    parser.add_argument('--num_rounds', type=int, default=50,
                        help='Number of communication rounds T')
    parser.add_argument('--local_epochs', type=int, default=5,
                        help='Local training epochs E')
    parser.add_argument('--client_fraction', type=float, default=1.0,
                        help='Fraction of clients per round C')
    parser.add_argument('--parallel_clients', type=int, default=3,
                        help='Max clients to train in parallel (RTX 4070: 3)')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset name')
    parser.add_argument('--partition', type=str, default='dirichlet',
                        choices=['iid', 'dirichlet'],
                        help='Data partition method')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet concentration α')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Local batch size')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate η')
    parser.add_argument('--fedprox_mu', type=float, default=0.01,
                        help='FedProx regularization μ')
    
    # Nested Learning
    parser.add_argument('--fast_lr_mult', type=float, default=3.0,
                        help='Fast learning rate multiplier')
    parser.add_argument('--slow_freq', type=int, default=5,
                        help='Slow weight update frequency K')
    
    # Early Exit
    parser.add_argument('--exit_threshold', type=float, default=0.8,
                        help='Early exit confidence threshold τ')
    
    # NEW: Nested Learning Features (NeurIPS 2025)
    parser.add_argument('--use_lss', action='store_true', default=True,
                        help='Enable Local Surprise Signal (LSS)')
    parser.add_argument('--no_lss', action='store_true',
                        help='Disable Local Surprise Signal')
    parser.add_argument('--lss_temp', type=float, default=1.0,
                        help='LSS temperature (higher = more uniform)')
    parser.add_argument('--use_dmgd', action='store_true',
                        help='Enable Deep Momentum GD (adds overhead)')
    parser.add_argument('--cms_levels', type=int, default=4,
                        help='Number of CMS memory levels')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')
    
    # Distributed mode
    parser.add_argument('--server_ip', type=str, default='localhost',
                        help='Server IP (client mode)')
    parser.add_argument('--port', type=int, default=4433,
                        help='Server port')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Print header
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   Nested Early-Exit Federated Learning                              ║
║   IEEE Transactions on Mobile Computing, 2025                       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    config.update({
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'local_epochs': args.local_epochs,
        'client_fraction': args.client_fraction,
        'parallel_clients': args.parallel_clients,
        'dataset': args.dataset,
        'partition': args.partition,
        'alpha': args.alpha,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'fedprox_mu': args.fedprox_mu,
        'fast_lr_multiplier': args.fast_lr_mult,
        'slow_update_freq': args.slow_freq,
        'exit_threshold': args.exit_threshold,
        'device': args.device,
        'seed': args.seed,
        'mixed_precision': not args.no_amp,
        # Nested Learning Features (NeurIPS 2025)
        'use_lss': args.use_lss and not args.no_lss,
        'lss_temperature': args.lss_temp,
        'use_deep_momentum': args.use_dmgd,
        'cms_num_levels': args.cms_levels,
    })
    
    # Log configuration
    logger.info("Configuration:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    
    # Create experiment
    experiment = FederatedExperiment(config)
    
    if args.mode == 'simulation':
        experiment.run_simulation()
    elif args.mode == 'server':
        logger.info("Starting FL server...")
        logger.info(f"Listening on port {args.port}")
        # TODO: Integrate with QUIC server
        logger.warning("Distributed mode not yet implemented. Use simulation mode.")
    elif args.mode == 'client':
        logger.info(f"Connecting to server at {args.server_ip}:{args.port}")
        # TODO: Integrate with QUIC client
        logger.warning("Distributed mode not yet implemented. Use jetson/run_client.sh")


if __name__ == '__main__':
    main()
