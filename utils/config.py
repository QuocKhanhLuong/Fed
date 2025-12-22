"""
Configuration Module for FL-QUIC Project
Central configuration for hyperparameters and system settings

Author: Research Team - FL-QUIC Project
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class NetworkConfig:
    """Network and QUIC configuration"""
    server_host: str = "0.0.0.0"
    server_port: int = 4433
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    
    # QUIC settings
    enable_0rtt: bool = True
    max_stream_data: int = 10 * 1024 * 1024  # 10 MB
    idle_timeout: float = 600.0  # Increased for long training (10 min)
    
    # Connection settings
    connection_timeout: float = 10.0
    round_timeout: float = 300.0  # 5 minutes per round


@dataclass
class FederatedConfig:
    """Federated Learning configuration"""
    num_rounds: int = 10
    min_clients: int = 2
    min_available_clients: int = 3
    client_fraction: float = 1.0  # Fraction of clients to sample per round
    
    # Aggregation strategy
    aggregation_strategy: str = "FedProx"  # FedAvg, FedProx, FedDyn
    feddyn_alpha: float = 0.01  # FedDyn regularization strength (α)
    
    # Convergence tracking
    target_accuracy: float = 0.80  # Target accuracy for convergence detection
    convergence_patience: int = 3  # Rounds to wait for improvement


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Model settings
    model_name: str = "mobilevitv2_100"  # timm model name
    num_classes: int = 100  # CIFAR-100
    image_size: int = 32  # Input image size
    
    # Early Exit settings
    exit_weights: List[float] = None  # Weights for each exit [exit1, exit2, exit3]
    exit_threshold: float = 0.8  # Confidence threshold for early exit
    
    # Nested Learning settings
    fast_lr_multiplier: float = 3.0  # Learning rate multiplier for fast weights
    slow_update_freq: int = 5  # Update slow weights every N steps
    
    # Self-Distillation
    use_self_distillation: bool = True
    distillation_weight: float = 0.1
    distillation_temp: float = 3.0
    
    # Continuum Memory System (CMS)
    cms_enabled: bool = True
    cms_update_freqs: List[int] = None  # [1, 5, 25]
    cms_decay_rates: List[float] = None  # [0.0, 0.9, 0.99]
    cms_weight: float = 0.001
    
    def __post_init__(self):
        if self.exit_weights is None:
            self.exit_weights = [0.3, 0.3, 0.4]
        if self.cms_update_freqs is None:
            self.cms_update_freqs = [1, 5, 25]
        if self.cms_decay_rates is None:
            self.cms_decay_rates = [0.0, 0.9, 0.99]


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Optimizer
    learning_rate: float = 1e-3
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    fedprox_mu: float = 0.01
    
    # Training
    local_epochs: int = 50
    batch_size: int = 16
    num_workers: int = 2
    
    # Device
    device: str = "cuda"  # Will be auto-detected
    mixed_precision: bool = True  # Use FP16 on GPU
    
    # Data
    train_split: float = 0.8
    val_split: float = 0.2
    
    # Dataset Configuration
    dataset_name: str = "cifar100"  # cifar10, cifar100, pathmnist, etc.
    data_dir: str = "./data"  # Directory to store datasets
    partition_alpha: float = 0.5  # Dirichlet concentration parameter


@dataclass
class CompressionConfig:
    """Model compression configuration"""
    # Quantization
    enable_quantization: bool = True
    quantization_dtype: str = "int8"  # int8, int4
    
    # LZ4 compression
    compression_level: int = 4  # 0-16, higher = better compression
    
    # Pruning (optional)
    enable_pruning: bool = False
    pruning_ratio: float = 0.3


@dataclass
class SystemConfig:
    """System and logging configuration"""
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    tensorboard_dir: str = "./runs"
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 5  # Save every N rounds
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_port: int = 9090


class Config:
    """Master configuration class"""
    
    def __init__(self):
        self.network = NetworkConfig()
        self.federated = FederatedConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.compression = CompressionConfig()
        self.system = SystemConfig()
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            'network': self.network.__dict__,
            'federated': self.federated.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'compression': self.compression.__dict__,
            'system': self.system.__dict__,
        }
    
    def update_from_dict(self, config_dict: dict):
        """Update configuration from dictionary"""
        for category, values in config_dict.items():
            if hasattr(self, category):
                config_obj = getattr(self, category)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)


# Default configuration instance
default_config = Config()


# Jetson Nano optimized configuration
def get_jetson_config() -> Config:
    """Get optimized configuration for Jetson Nano"""
    config = Config()
    
    # Reduce batch size for limited memory (4GB RAM)
    config.training.batch_size = 8
    config.training.mixed_precision = True
    config.training.num_workers = 2
    
    # Smaller image size for faster inference
    config.model.image_size = 32
    
    # Aggressive compression for limited bandwidth
    config.compression.enable_quantization = True
    config.compression.compression_level = 9
    
    # Simpler early exit config
    config.model.exit_threshold = 0.7  # Lower threshold for faster inference
    config.model.fast_lr_multiplier = 2.0
    config.model.slow_update_freq = 10
    
    # Disable CMS for memory saving
    config.model.cms_enabled = False
    
    return config


# High-performance server configuration
def get_server_config() -> Config:
    """Get configuration for high-performance server"""
    config = Config()
    
    # Larger batch sizes
    config.training.batch_size = 64
    config.training.num_workers = 8
    
    # More clients per round
    config.federated.min_clients = 5
    config.federated.min_available_clients = 3
    
    # Full features
    config.model.cms_enabled = True
    config.model.use_self_distillation = True
    
    return config


def get_rtx4070_config() -> Config:
    """Get optimized configuration for RTX 4070 (12GB VRAM)"""
    config = Config()
    
    # Memory optimized for 12GB VRAM
    config.training.batch_size = 64
    config.training.mixed_precision = True
    config.training.num_workers = 4
    
    # FL settings
    config.federated.min_clients = 3
    config.federated.num_rounds = 50
    config.federated.aggregation_strategy = "FedDyn"
    
    # Full features enabled
    config.model.cms_enabled = True
    config.model.use_self_distillation = True
    
    # Compression
    config.compression.compression_level = 9
    config.compression.enable_quantization = True
    
    return config
