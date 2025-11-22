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

try:
    import sys
    import os
    import timm
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.gated_mobilevit import GatedBlockWrapper
except ImportError as e:
    logging.error(f"Failed to import dependencies: {e}")
    raise

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
        
        # Inject Gating
        self._inject_gating()
        
        # Freeze Backbone
        self._freeze_backbone()
        
        # No mock model check needed
        self.is_mock_model = False
        
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
        """Build MobileViTv2 model using timm"""
        logger.info(f"Initializing MobileViTv2 (timm) for {self.num_classes} classes")
        # Create model using timm
        model = timm.create_model('mobilevitv2_050.cvnets_in1k', pretrained=True, num_classes=self.num_classes)
        return model

    def _inject_gating(self):
        """Inject GatedBlockWrapper into the model"""
        logger.info("Injecting GatedBlockWrapper into MobileViT blocks...")
        from timm.models.mobilevit import MobileVitV2Block
        
        modules_to_replace = []
        for name, module in self.model.named_modules():
            if isinstance(module, MobileVitV2Block):
                modules_to_replace.append((name, module))
        
        for name, module in modules_to_replace:
            # Create wrapper
            wrapped_block = GatedBlockWrapper(module)
            
            # Replace in parent
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = self.model.get_submodule(parent_name)
                setattr(parent, child_name, wrapped_block)
            else:
                # Handle top-level if necessary (unlikely for blocks)
                pass
            
        logger.info(f"Replaced {len(modules_to_replace)} blocks with GatedBlockWrapper")

    def _freeze_backbone(self):
        """Freeze backbone, keep expert/gate/head trainable"""
        logger.info("Freezing backbone parameters...")
        
        # Freeze all first
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze specific parts
        for name, module in self.model.named_modules():
            if isinstance(module, GatedBlockWrapper):
                for param in module.expert.parameters():
                    param.requires_grad = True
                for param in module.gate.parameters():
                    param.requires_grad = True
            
        # Unfreeze head
        if hasattr(self.model, 'head'):
            for param in self.model.head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
                
        trainable = self._count_trainable_parameters()
        total = self._count_parameters()
        logger.info(f"Frozen backbone. Trainable params: {trainable}/{total} ({trainable/total:.1%})")
    
    # Mock model removed
    
    def _count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def _count_trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _count_lora_parameters(self) -> int:
        """Count LoRA-specific parameters (Deprecated)"""
        return 0
    
    def get_parameters(self) -> List[np.ndarray]:
        """
        Extract model parameters as NumPy arrays.
        For LoRA models, only extracts LoRA weights.
        
        Returns:
            List of NumPy arrays (weights)
        """
        parameters = []
        
        # Extract all trainable parameters
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
        # 3. Iterate through params
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
        
        # Update all trainable parameters
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
                        # Get network score from self.network_monitor
                        quality_score = 1.0
                        if self.network_monitor:
                            quality_score = self.network_monitor.get_network_score()
                        
                        # Propagate score
                        self.model.apply(lambda m: m.set_quality_score(quality_score) if hasattr(m, 'set_quality_score') else None)

                        outputs = self.model(images)
                        logits = outputs
                        task_loss = F.cross_entropy(logits, labels)
                        
                        loss = task_loss
                        if fedprox_mu > 0 and global_weights is not None:
                            prox_loss = self._compute_proximal_loss(global_weights, fedprox_mu)
                            loss += prox_loss
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Get network score from self.network_monitor
                    quality_score = 1.0
                    if self.network_monitor:
                        quality_score = self.network_monitor.get_network_score()
                    
                    # Propagate score
                    self.model.apply(lambda m: m.set_quality_score(quality_score) if hasattr(m, 'set_quality_score') else None)

                    outputs = self.model(images)
                    logits = outputs
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
                
                outputs = self.model(images)
                logits = outputs
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
        """Compute FedProx regularization term"""
        proximal_term = 0.0
        
        # Get current model parameters (training)
        # Note: self.get_parameters() returns numpy list, but here we need tensors
        # to keep the gradient graph. So we iterate model.parameters() directly.
        
        # Create iterator for global weights
        # We assume the order matches get_parameters()
        
        # 1. Get only trainable parameters (LoRA)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 2. Check length mismatch
        if len(trainable_params) != len(global_weights):
            raise ValueError(f"Parameter length mismatch: model={len(trainable_params)}, global={len(global_weights)}")

        # 3. Compute L2 distance
        for param, global_w in zip(trainable_params, global_weights):
            # Check shape
            if param.shape != global_w.shape:
                raise ValueError(f"Shape mismatch: model={param.shape}, global={global_w.shape}")

            # Convert global weight (numpy) to tensor on same device
            global_w_tensor = torch.tensor(global_w).to(self.device)
            
            # Accumulate norm^2
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
