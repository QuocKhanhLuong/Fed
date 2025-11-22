"""
Configuration Module for FL-QUIC-LoRA Project
Central configuration for hyperparameters and system settings

Author: Research Team - FL-QUIC-LoRA Project
"""

from dataclasses import dataclass
from typing import Optional


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
    idle_timeout: float = 60.0
    
    # Connection settings
    connection_timeout: float = 10.0
    round_timeout: float = 300.0  # 5 minutes per round


@dataclass
class FederatedConfig:
    """Federated Learning configuration"""
    num_rounds: int = 10
    min_clients: int = 2
    min_available_clients: int = 2
    client_fraction: float = 1.0  # Fraction of clients to sample per round
    
    # Aggregation strategy
    aggregation_strategy: str = "FedAvg"  # FedAvg, FedProx, etc.


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # MobileViT settings
    model_name: str = "apple/mobilevit-small"
    num_classes: int = 10
    image_size: int = 224
    
    # LoRA settings (Feature D)
    use_lora: bool = False  # Use LoRA for parameter-efficient fine-tuning
    lora_r: int = 8  # LoRA rank
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.1
    lora_target_modules: Optional[list] = None  # Will be set to ["qkv"] for ViT
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["qkv"]  # Target attention layers


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Optimizer
    learning_rate: float = 1e-3
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    fedprox_mu: float = 0.01
    
    # Training
    local_epochs: int = 3
    batch_size: int = 32
    num_workers: int = 2
    
    # Device
    device: str = "cuda"  # Will be auto-detected
    mixed_precision: bool = True  # Use FP16 on Jetson Nano
    
    # Data
    train_split: float = 0.8
    val_split: float = 0.2
    
    # Advanced Training Techniques
    # Feature A: Sharpness-Aware Minimization (SAM)
    use_sam: bool = False  # Enable SAM optimizer for better generalization
    sam_rho: float = 0.05  # Neighborhood size for SAM
    
    # Feature C: Test-Time Adaptation (TTA)
    use_tta: bool = False  # Enable TTA during evaluation
    tta_steps: int = 1  # Number of TTA adaptation steps
    tta_lr: float = 1e-4  # Learning rate for TTA


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
    
    # Reduce batch size for limited memory
    config.training.batch_size = 16
    config.training.mixed_precision = True
    config.training.num_workers = 2
    
    # Aggressive compression for limited bandwidth
    config.compression.enable_quantization = True
    config.compression.compression_level = 9
    
    # Smaller LoRA rank for faster training
    config.model.lora_r = 4
    config.model.lora_alpha = 8
    
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
    
    # Higher LoRA rank for better performance
    config.model.lora_r = 16
    config.model.lora_alpha = 32
    
    return config
