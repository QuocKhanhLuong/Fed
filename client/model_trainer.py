"""
MobileViT + LoRA Model Trainer
Optimized for Edge Devices (Jetson Nano)

Features:
- MobileViT backbone from Hugging Face
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Mixed precision training (FP16)
- Only extracts LoRA weights for transmission

Author: Research Team - FL-QUIC-LoRA Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import logging
from pathlib import Path

# Hugging Face Transformers
try:
    from transformers import MobileViTForImageClassification, MobileViTConfig
    from peft import LoraConfig, get_peft_model
    from peft.utils import TaskType
    HAS_TRANSFORMERS = True
except ImportError as e:
    HAS_TRANSFORMERS = False
    logging.warning(f"transformers/peft not installed - using mock model: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileViTLoRATrainer:
    """
    Trainer for MobileViT with LoRA adaptation.
    Designed for Federated Learning on edge devices.
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        model_name: str = "apple/mobilevit-small",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        device: str = "auto",
        use_mixed_precision: bool = True,
        network_monitor: Optional[Any] = None,  # <--- NEW
    ):
        """
        Initialize the trainer.
        
        Args:
            num_classes: Number of output classes
            model_name: Pretrained MobileViT model name
            lora_r: LoRA rank (lower = fewer parameters)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout for LoRA layers
            device: Device to use ('cuda', 'cpu', or 'auto')
            use_mixed_precision: Use FP16 training
            network_monitor: NetworkMonitor instance for adaptive logic
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_mixed_precision = use_mixed_precision
        self.network_monitor = network_monitor  # Store reference
        
        # Auto-detect device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing MobileViTLoRATrainer on {self.device}")
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Check if using mock model
        self.is_mock_model = not HAS_TRANSFORMERS or isinstance(self.model, nn.Module) and self.model.__class__.__name__ == 'SimpleCNN'
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and self.device.type == "cuda" else None
        
        # Statistics
        self.stats = {
            'total_params': self._count_parameters(),
            'trainable_params': self._count_trainable_parameters(),
            'lora_params': self._count_lora_parameters(),
        }
        
        logger.info(f"Model initialized: {self.stats}")
    
    def _build_model(self) -> nn.Module:
        """Build MobileViT model with LoRA adaptation"""
        
        if not HAS_TRANSFORMERS:
            logger.warning("Using mock model - install transformers and peft")
            return self._build_mock_model()
        
        try:
            # Load pretrained MobileViT
            logger.info(f"Loading {self.model_name}...")
            model = MobileViTForImageClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
            )
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # Use FEATURE_EXTRACTION for ViT models
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["query", "key", "value"],  # Target attention Q, K, V
                bias="none",
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            
            # Wrap with Adaptive Model
            from .adaptive_model import AdaptiveMobileViT
            model = AdaptiveMobileViT(model)
            
            logger.info("LoRA applied and wrapped with AdaptiveMobileViT")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Falling back to mock model")
            return self._build_mock_model()
    
    def _build_mock_model(self) -> nn.Module:
        """Build a simple CNN for testing without transformers"""
        
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * 56 * 56, 512)
                self.fc2 = nn.Linear(512, num_classes)
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, pixel_values):
                x = pixel_values
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                logits = self.fc2(x)
                return {"logits": logits}
        
        logger.info("Created SimpleCNN mock model")
        return SimpleCNN(self.num_classes)
    
    def _count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def _count_trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _count_lora_parameters(self) -> int:
        """Count LoRA-specific parameters"""
        if HAS_TRANSFORMERS:
            try:
                return sum(p.numel() for n, p in self.model.named_parameters() 
                          if 'lora' in n.lower() and p.requires_grad)
            except:
                pass
        return self._count_trainable_parameters()
    
    def get_parameters(self) -> List[np.ndarray]:
        """
        Extract model parameters as NumPy arrays.
        For LoRA models, only extracts LoRA weights.
        
        Returns:
            List of NumPy arrays (weights)
        """
        parameters = []
        
        if HAS_TRANSFORMERS:
            # Extract only LoRA parameters
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    parameters.append(param.detach().cpu().numpy())
        else:
            # Extract all trainable parameters (mock model)
            for param in self.model.parameters():
                if param.requires_grad:
                    parameters.append(param.detach().cpu().numpy())
        
        logger.debug(f"Extracted {len(parameters)} parameter arrays")
        return parameters

    def get_adaptive_parameters(self) -> List[np.ndarray]:
        """
        Extract and Prune parameters based on Network Quality.
        Returns sparse weights (zeros) to be compressed by LZ4.
        """
        # 1. Determine Network Score
        score = 1.0
        if self.network_monitor:
            score = self.network_monitor.get_network_score()
        
        # 2. Calculate Keep Ratio (Dynamic Pruning)
        # Good Network (>0.8) -> Keep 100%
        # Poor Network (0.0) -> Keep min 10%
        if score >= 0.8:
            keep_ratio = 1.0
        else:
            # Linear interpolation: 0.0 -> 0.1, 0.8 -> 1.0
            keep_ratio = 0.1 + (0.9 * (score / 0.8))
            
        logger.info(f"Network Score: {score:.2f} -> Pruning Keep Ratio: {keep_ratio:.2%}")

        parameters = []
        
        # 3. Iterate through params (Logic similar to get_parameters but with Pruning)
        # Note: Only process LoRA params or trainable params
        if HAS_TRANSFORMERS and not self.is_mock_model:
            # Real transformers model - only extract LoRA params
            iterator = [(n, p) for n, p in self.model.named_parameters() 
                       if 'lora' in n.lower() and p.requires_grad]
        else:
            # Mock model or no transformers - extract all trainable params
            iterator = [(f"param_{i}", p) for i, p in enumerate(self.model.parameters()) 
                       if p.requires_grad]

        for name, param in iterator:
            # Work on a copy to avoid affecting the original model
            tensor = param.detach().clone()
            
            # Perform Pruning if needed
            if keep_ratio < 1.0:
                # Calculate Threshold at k-th percentile
                # Example: keep 20% -> Threshold is the value at top 20%
                numel = tensor.numel()
                k = int(numel * keep_ratio)
                
                if k < 1: k = 1 # Always keep at least 1 element
                
                # Find top-k value by absolute magnitude
                # flatten() to 1D, abs() for magnitude
                threshold_val = torch.kthvalue(
                    tensor.abs().flatten(), 
                    numel - k + 1
                ).values
                
                # Create mask: Keep values >= threshold
                mask = tensor.abs() >= threshold_val
                
                # Set unimportant values to 0
                tensor = tensor * mask
                
            # Convert to Numpy and add to list
            parameters.append(tensor.cpu().numpy())
        
        # Log pruning statistics
        total_params = sum(p.size for p in parameters)
        if keep_ratio < 1.0:
            zero_params = sum(np.count_nonzero(p == 0) for p in parameters)
            sparsity = zero_params / total_params if total_params > 0 else 0
            logger.info(f"Adaptive Pruning: {len(parameters)} arrays, "
                       f"{total_params:,} params, {sparsity:.1%} zeros (LZ4 will compress these)")
        else:
            logger.debug(f"No pruning applied: {len(parameters)} arrays, {total_params:,} params")

        return parameters
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from NumPy arrays.
        For LoRA models, only updates LoRA weights.
        
        Args:
            parameters: List of NumPy arrays
        """
        param_idx = 0
        
        if HAS_TRANSFORMERS:
            # Update only LoRA parameters
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    if param_idx >= len(parameters):
                        logger.error(f"Not enough parameters provided")
                        break
                    
                    param.data = torch.from_numpy(parameters[param_idx]).to(self.device)
                    param_idx += 1
        else:
            # Update all trainable parameters (mock model)
            for param in self.model.parameters():
                if param.requires_grad:
                    if param_idx >= len(parameters):
                        break
                    param.data = torch.from_numpy(parameters[param_idx]).to(self.device)
                    param_idx += 1
        
        logger.debug(f"Set {param_idx} parameter arrays")
    
    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        fedprox_mu: float = 0.0,       
        global_weights: List[np.ndarray] = None,
        network_monitor: Optional[Any] = None  # <--- NEW
    ) -> Dict[str, float]:
        """
        Train the model locally.
        
        Args:
            train_loader: Training data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Training loop
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if self.use_mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        # Get network score
                        quality_score = 1.0
                        if network_monitor:
                            quality_score = network_monitor.get_network_score()
                            if hasattr(self.model, 'set_quality_score'):
                                self.model.set_quality_score(quality_score)

                        outputs = self.model(pixel_values=images, quality_score=quality_score)
                        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                        task_loss = F.cross_entropy(logits, labels)
                        
                        loss = task_loss
                        if fedprox_mu > 0 and global_weights is not None:
                            prox_loss = self._compute_proximal_loss(global_weights, fedprox_mu)
                            loss += prox_loss
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(pixel_values=images)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    task_loss = F.cross_entropy(logits, labels)
                    
                    loss = task_loss
                    if fedprox_mu > 0 and global_weights is not None:
                        prox_loss = self._compute_proximal_loss(global_weights, fedprox_mu)
                        loss += prox_loss
                    
                    loss.backward()
                    optimizer.step()
                
                # Statistics
                with torch.no_grad():
                    predictions = logits.argmax(dim=1)
                    correct = (predictions == labels).sum().item()
                    
                    epoch_loss += loss.item() * images.size(0)
                    epoch_correct += correct
                    epoch_samples += images.size(0)
            
            # Epoch statistics
            avg_loss = epoch_loss / epoch_samples
            accuracy = epoch_correct / epoch_samples
            
            logger.info(f"Epoch {epoch + 1}/{epochs}: "
                       f"Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        # Final metrics
        metrics = {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'num_samples': total_samples,
            'num_epochs': epochs,
        }
        
        return metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(pixel_values=images)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                loss = F.cross_entropy(logits, labels)
                
                predictions = logits.argmax(dim=1)
                correct = (predictions == labels).sum().item()
                
                total_loss += loss.item() * images.size(0)
                total_correct += correct
                total_samples += images.size(0)
        
        metrics = {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'num_samples': total_samples,
        }
        
        logger.info(f"Evaluation: Loss={metrics['loss']:.4f}, "
                   f"Accuracy={metrics['accuracy']:.4f}")
        
        return metrics
    
    def _compute_proximal_loss(self, global_weights: List[np.ndarray], mu: float) -> torch.Tensor:
        """Tính toán FedProx regularization term"""
        proximal_term = 0.0
        
        # Lấy các tham số hiện tại của model (đang train)
        # Lưu ý: get_parameters() của bạn trả về list numpy, nhưng ở đây ta cần tensor
        # để giữ được gradient graph. Vì vậy ta duyệt qua model.parameters() trực tiếp.
        
        # Tạo iterator cho global weights (đã flatten hoặc theo layer)
        # Ở đây ta giả định thứ tự parameters khớp với get_parameters()
        
        # 1. Chỉ lấy các tham số cần train (LoRA)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 2. Kiểm tra khớp số lượng
        if len(trainable_params) != len(global_weights):
            return torch.tensor(0.0).to(self.device)

        # 3. Tính khoảng cách L2
        for param, global_w in zip(trainable_params, global_weights):
            # Chuyển global weight (numpy) sang tensor trên cùng device
            global_w_tensor = torch.tensor(global_w).to(self.device)
            
            # Cộng dồn norm^2
            proximal_term += (param - global_w_tensor).norm(2) ** 2

        return (mu / 2.0) * proximal_term


def create_dummy_dataset(
    num_samples: int = 100,
    num_classes: int = 10,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dummy dataset for testing.
    
    Args:
        num_samples: Number of samples
        num_classes: Number of classes
        image_size: Image size
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Generate random images and labels
    train_images = torch.randn(num_samples, 3, image_size, image_size)
    train_labels = torch.randint(0, num_classes, (num_samples,))
    
    test_images = torch.randn(num_samples // 5, 3, image_size, image_size)
    test_labels = torch.randint(0, num_classes, (num_samples // 5,))
    
    # Create datasets
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("MobileViT + LoRA Trainer Demo")
    logger.info("="*60)
    
    # Create trainer
    trainer = MobileViTLoRATrainer(
        num_classes=10,
        lora_r=4,  # Small rank for testing
        lora_alpha=8,
        use_mixed_precision=False,  # Disable for CPU testing
    )
    
    # Create dummy data
    logger.info("\nCreating dummy dataset...")
    train_loader, test_loader = create_dummy_dataset(num_samples=100)
    
    # Extract initial parameters
    logger.info("\nExtracting initial parameters...")
    initial_params = trainer.get_parameters()
    logger.info(f"Number of parameter arrays: {len(initial_params)}")
    logger.info(f"Total parameters: {sum(p.size for p in initial_params):,}")
    
    # Train
    logger.info("\nTraining...")
    metrics = trainer.train(train_loader, epochs=2, learning_rate=1e-3)
    logger.info(f"Training metrics: {metrics}")
    
    # Evaluate
    logger.info("\nEvaluating...")
    eval_metrics = trainer.evaluate(test_loader)
    logger.info(f"Evaluation metrics: {eval_metrics}")
    
    # Extract trained parameters
    logger.info("\nExtracting trained parameters...")
    trained_params = trainer.get_parameters()
    
    # Verify parameters changed
    param_diff = np.mean([np.mean(np.abs(p1 - p2)) 
                         for p1, p2 in zip(initial_params, trained_params)])
    logger.info(f"Average parameter change: {param_diff:.6f}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ Demo completed successfully!")
    logger.info("="*60)
