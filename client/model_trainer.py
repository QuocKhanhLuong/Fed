import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import logging
import sys
import os
import timm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import ClassificationEvaluator, ResourceTracker

# GatedBlockWrapper for network-adaptive gating
class GatedBlockWrapper(nn.Module):
    """Wraps a MobileViT block with adaptive gating based on network quality."""
    def __init__(self, block: nn.Module, expert_dim: int = 64):
        super().__init__()
        self.block = block
        self.quality_score = 1.0
        
        # Lightweight expert for low-quality scenarios
        self.expert = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LazyLinear(expert_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.LazyLinear(1),
            nn.Sigmoid(),
        )
        self._initialized = False
    
    def set_quality_score(self, score: float):
        self.quality_score = score
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize lazy layers
        if not self._initialized:
            with torch.no_grad():
                _ = self.expert(x)
                expert_out = self.expert(x)
                _ = self.gate(expert_out)
            self._initialized = True
        
        # High quality: use full block
        if self.quality_score >= 0.8:
            return self.block(x)
        
        # Low quality: blend with expert path
        block_out = self.block(x)
        expert_feat = self.expert(x)
        gate_val = self.gate(expert_feat)
        
        # gate_val determines how much to trust the block output
        return block_out * gate_val.view(-1, 1, 1, 1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    LoraConfig = None
    get_peft_model = None
    PeftModel = None

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("SAM requires first_step() and second_step()")
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), p=2
        )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class MobileViTLoRATrainer:
    def __init__(
        self,
        num_classes: int = 10,
        model_name: str = "apple/mobilevit-small",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        device: str = "auto",
        use_mixed_precision: bool = True,
        network_monitor: Optional[Any] = None,
        use_sam: bool = False,
        sam_rho: float = 0.05,
        use_tta: bool = False,
        tta_steps: int = 1,
        tta_lr: float = 1e-4,
        use_lora: bool = False,
        checkpoint_dir: str = "checkpoints",
    ):
        self.num_classes = num_classes
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_mixed_precision = use_mixed_precision
        self.network_monitor = network_monitor
        self.use_sam = use_sam
        self.sam_rho = sam_rho
        self.use_tta = use_tta
        self.tta_steps = tta_steps
        self.tta_lr = tta_lr
        self.use_lora = use_lora
        self.checkpoint_dir = checkpoint_dir
        
        # Best model tracking
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Trainer init: SAM={use_sam}, TTA={use_tta}, LoRA={use_lora}")
        
        self.model = self._build_model()
        self._inject_gating()
        self.model.to(self.device)
        self._freeze_backbone()
        
        self.scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and self.device.type == "cuda" else None
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _build_model(self) -> nn.Module:
        base_model = timm.create_model('mobilevitv2_050.cvnets_in1k', pretrained=True, num_classes=self.num_classes)
        
        if self.use_lora and HAS_PEFT and LoraConfig is not None:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["head.fc"],
                modules_to_save=[],
            )
            model = get_peft_model(base_model, lora_config)
            return model
        
        return base_model

    def _inject_gating(self):
        from timm.models.mobilevit import MobileVitV2Block
        modules_to_replace = []
        for name, module in self.model.named_modules():
            if isinstance(module, MobileVitV2Block):
                modules_to_replace.append((name, module))
        
        for name, module in modules_to_replace:
            wrapped_block = GatedBlockWrapper(module)
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = self.model.get_submodule(parent_name)
                setattr(parent, child_name, wrapped_block)

    def _freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
            
        for name, module in self.model.named_modules():
            if isinstance(module, GatedBlockWrapper):
                for param in module.expert.parameters(): param.requires_grad = True
                for param in module.gate.parameters(): param.requires_grad = True
            
        if hasattr(self.model, 'head'):
            for param in self.model.head.parameters(): param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters(): param.requires_grad = True
                
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
    
    def _count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def save_checkpoint(
        self,
        filename: str = "best_model.pth",
        epoch: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Lưu checkpoint của model.
        
        Args:
            filename: Tên file checkpoint (mặc định: "best_model.pth")
            epoch: Epoch hiện tại (optional)
            optimizer: Optimizer state (optional)
            metrics: Các metrics của model (optional)
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'use_lora': self.use_lora,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss,
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # Nếu sử dụng LoRA, lưu thêm config
        if self.use_lora and HAS_PEFT:
            checkpoint['lora_config'] = {
                'lora_r': self.lora_r,
                'lora_alpha': self.lora_alpha,
                'lora_dropout': self.lora_dropout,
            }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(
        self,
        filename: str = "best_model.pth",
        load_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint của model.
        
        Args:
            filename: Tên file checkpoint
            load_optimizer: Có load optimizer state không
            optimizer: Optimizer để load state vào (nếu load_optimizer=True)
        
        Returns:
            Dict chứa thông tin checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore best metrics
        if 'best_accuracy' in checkpoint:
            self.best_accuracy = checkpoint['best_accuracy']
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
        
        # Load optimizer state nếu cần
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint
    
    def save_best_model(
        self,
        accuracy: float,
        loss: float,
        epoch: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> bool:
        """
        Lưu model nếu là model tốt nhất (dựa trên accuracy).
        
        Args:
            accuracy: Accuracy hiện tại
            loss: Loss hiện tại
            epoch: Epoch hiện tại
            optimizer: Optimizer state
        
        Returns:
            True nếu đã lưu model mới, False nếu không
        """
        is_best = accuracy > self.best_accuracy
        
        if is_best:
            self.best_accuracy = accuracy
            self.best_loss = loss
            
            metrics = {
                'accuracy': accuracy,
                'loss': loss,
            }
            
            self.save_checkpoint(
                filename="best_model.pth",
                epoch=epoch,
                optimizer=optimizer,
                metrics=metrics,
            )
            
            logger.info(f"New best model saved! Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            return True
        
        return False
    
    def get_parameters(self) -> List[np.ndarray]:
        parameters = []
        if self.use_lora and HAS_PEFT and PeftModel is not None and isinstance(self.model, PeftModel):
            for name, param in self.model.named_parameters():
                if param.requires_grad and ('lora' in name.lower() or 'adapter' in name.lower()):
                    parameters.append(param.detach().cpu().numpy())
        else:
            for param in self.model.parameters():
                if param.requires_grad:
                    parameters.append(param.detach().cpu().numpy())
        return parameters

    def get_adaptive_parameters(self) -> List[np.ndarray]:
        score = self.network_monitor.get_network_score() if self.network_monitor else 1.0
        keep_ratio = 1.0 if score >= 0.8 else 0.1 + (0.9 * (score / 0.8))
            
        parameters = []
        iterator = [(f"param_{i}", p) for i, p in enumerate(self.model.parameters()) if p.requires_grad]

        for name, param in iterator:
            tensor = param.detach().clone()
            if keep_ratio < 1.0:
                numel = tensor.numel()
                k = int(numel * keep_ratio)
                if k < 1: k = 1
                threshold_val = torch.kthvalue(tensor.abs().flatten(), numel - k + 1).values
                mask = tensor.abs() >= threshold_val
                tensor = tensor * mask
            parameters.append(tensor.cpu().numpy())
        
        return parameters
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        param_idx = 0
        if self.use_lora and HAS_PEFT and PeftModel is not None and isinstance(self.model, PeftModel):
            for name, param in self.model.named_parameters():
                if param.requires_grad and ('lora' in name.lower() or 'adapter' in name.lower()):
                    if param_idx >= len(parameters): break
                    new_param = torch.from_numpy(parameters[param_idx]).to(self.device)
                    if new_param.shape == param.shape:
                        param.data = new_param
                    param_idx += 1
        else:
            for param in self.model.parameters():
                if param.requires_grad:
                    if param_idx >= len(parameters): break
                    param.data = torch.from_numpy(parameters[param_idx]).to(self.device)
                    param_idx += 1
    
    def _simulate_network_artifacts(self, images: torch.Tensor, score: float) -> torch.Tensor:
        if score >= 0.9: return images
        noise_level = (0.9 - score) * 0.5
        noise = torch.randn_like(images) * noise_level
        return torch.clamp(images + noise, 0.0, 1.0)

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        fedprox_mu: float = 0.0,       
        global_weights: Optional[List[np.ndarray]] = None,
        val_loader: Optional[DataLoader] = None,
        save_best: bool = False,
    ) -> Dict[str, float]:
        self.model.train()
        tracker = ResourceTracker(device=str(self.device))
        tracker.__enter__()
        
        if self.use_sam:
            optimizer = SAM(self.model.parameters(), torch.optim.AdamW, rho=self.sam_rho, lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        for epoch in range(epochs):
            epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                simulated_score = np.random.uniform(0.0, 0.8) if np.random.rand() < 0.3 else 1.0
                self.model.apply(lambda m: m.set_quality_score(simulated_score) if hasattr(m, 'set_quality_score') else None)
                if simulated_score < 0.9:
                    images = self._simulate_network_artifacts(images, simulated_score)

                optimizer.zero_grad()
                
                if self.use_sam:
                    with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                        outputs = self.model(images)
                        loss = F.cross_entropy(outputs, labels)
                        if fedprox_mu > 0 and global_weights:
                            loss += self._compute_proximal_loss(global_weights, fedprox_mu)
                    
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer.base_optimizer)
                        optimizer.first_step(zero_grad=True)
                    else:
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        
                    with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                        outputs = self.model(images)
                        loss = F.cross_entropy(outputs, labels)
                        if fedprox_mu > 0 and global_weights:
                            loss += self._compute_proximal_loss(global_weights, fedprox_mu)
                    
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer.base_optimizer)
                        self.scaler.update()
                        optimizer.second_step(zero_grad=True)
                    else:
                        loss.backward()
                        optimizer.second_step(zero_grad=True)
                else:
                    with torch.amp.autocast(enabled=self.use_mixed_precision):
                        outputs = self.model(images)
                        loss = F.cross_entropy(outputs, labels)
                        if fedprox_mu > 0 and global_weights:
                            loss += self._compute_proximal_loss(global_weights, fedprox_mu)
                    
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                
                with torch.no_grad():
                    predictions = outputs.argmax(dim=1)
                    correct = (predictions == labels).sum().item()
                    epoch_loss += loss.item() * images.size(0)
                    epoch_correct += correct
                    epoch_samples += images.size(0)
            
            logger.info(f"Epoch {epoch + 1}/{epochs}: Loss={epoch_loss/epoch_samples:.4f}, Acc={epoch_correct/epoch_samples:.4f}")
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
            
            # Evaluate and save best model if validation loader is provided
            if save_best and val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.save_best_model(
                    accuracy=val_metrics['accuracy'],
                    loss=val_metrics['loss'],
                    epoch=epoch + 1,
                    optimizer=optimizer,
                )
        
        tracker.__exit__(None, None, None)
        resource_metrics = tracker.get_metrics()
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'num_samples': total_samples,
            'num_epochs': epochs,
            'train_time_s': resource_metrics['time_s'],
            'gpu_mem_peak_mb': resource_metrics['gpu_mem_peak_mb'],
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        original_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        try:
            real_score = self.network_monitor.get_network_score() if self.network_monitor else 1.0
            self.model.apply(lambda m: m.set_quality_score(real_score) if hasattr(m, 'set_quality_score') else None)
            
            if self.use_tta:
                self._test_time_adaptation(test_loader)
            
            self.model.eval()
            total_loss = 0.0
            all_predictions, all_labels, all_probabilities = [], [], []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = F.cross_entropy(outputs, labels)
                    probabilities = F.softmax(outputs, dim=1)
                    predictions = outputs.argmax(dim=1)
                    
                    total_loss += loss.item() * images.size(0)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.append(probabilities.cpu().numpy())
            
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            all_probabilities = np.vstack(all_probabilities) if all_probabilities else None
            
            eval_metrics = ClassificationEvaluator.compute(
                predictions=all_predictions,
                targets=all_labels,
                probabilities=all_probabilities,
                num_classes=self.num_classes,
            )
            eval_metrics['loss'] = total_loss / max(1, eval_metrics['num_samples'])
            
            logger.info(f"Eval (Score={real_score:.2f}): Loss={eval_metrics['loss']:.4f}, Acc={eval_metrics['accuracy']:.4f}")
            return eval_metrics
            
        finally:
            if self.use_tta:
                self.model.load_state_dict(original_state)
    
    def _test_time_adaptation(self, test_loader: DataLoader) -> None:
        original_training = self.model.training
        self.model.train()
        
        frozen_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                frozen_params.append((name, param))
                param.requires_grad = False
        
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for param in module.parameters(): param.requires_grad = True
        
        bn_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(bn_params) > 0:
            optimizer = torch.optim.SGD(bn_params, lr=self.tta_lr)
            for step in range(self.tta_steps):
                for images, _ in test_loader:
                    images = images.to(self.device)
                    optimizer.zero_grad()
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = self._entropy_loss(outputs)
                    else:
                        outputs = self.model(images)
                        loss = self._entropy_loss(outputs)
                    
                    if self.use_mixed_precision and self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
        
        for name, param in frozen_params: param.requires_grad = True
        self.model.train(original_training)
    
    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
    
    def _compute_proximal_loss(self, global_weights: List[np.ndarray], mu: float) -> torch.Tensor:
        proximal_term = torch.tensor(0.0, device=self.device)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) != len(global_weights): return torch.tensor(0.0, device=self.device)
        
        for param, global_w in zip(trainable_params, global_weights):
            proximal_term = proximal_term + (param - torch.tensor(global_w, device=self.device)).norm(2) ** 2
        return torch.tensor(mu / 2.0, device=self.device) * proximal_term

def create_dummy_dataset(num_samples=100, num_classes=10, image_size=224):
    train_images = torch.randn(num_samples, 3, image_size, image_size)
    train_labels = torch.randint(0, num_classes, (num_samples,))
    test_images = torch.randn(num_samples // 5, 3, image_size, image_size)
    test_labels = torch.randint(0, num_classes, (num_samples // 5,))
    return DataLoader(TensorDataset(train_images, train_labels), batch_size=32, shuffle=True), \
           DataLoader(TensorDataset(test_images, test_labels), batch_size=32, shuffle=False)

if __name__ == "__main__":
    # Ví dụ sử dụng với checkpoint
    trainer = MobileViTLoRATrainer(
        num_classes=10, 
        lora_r=4, 
        use_mixed_precision=False,
        checkpoint_dir="checkpoints"
    )
    train_loader, test_loader = create_dummy_dataset(num_samples=50)
    
    # Training với validation và auto-save best model
    metrics = trainer.train(
        train_loader, 
        epochs=3,
        val_loader=test_loader,
        save_best=True
    )
    
    print(f"Training completed. Best accuracy: {trainer.best_accuracy:.4f}")
    
    # Load best model
    try:
        checkpoint = trainer.load_checkpoint("best_model.pth")
        print(f"Loaded best model from epoch {checkpoint.get('epoch', 'N/A')}")
    except FileNotFoundError:
        print("No checkpoint found")