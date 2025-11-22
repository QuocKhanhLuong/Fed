import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import logging
from pathlib import Path
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import sys
    import os
    import timm
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.gated_mobilevit import GatedBlockWrapper
    
    # Try to import PEFT for LoRA
    try:
        from peft import LoraConfig, get_peft_model, PeftModel
        HAS_PEFT = True
    except ImportError:
        HAS_PEFT = False
        LoraConfig = None  # type: ignore
        get_peft_model = None  # type: ignore
        PeftModel = None  # type: ignore
        logger.warning("peft library not installed - LoRA features disabled")
        
except ImportError as e:
    logger.error(f"Failed to import dependencies: {e}")
    raise


# ============================================================
# SAM Optimizer Implementation (Feature A)
# ============================================================
class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) Optimizer
    Paper: https://arxiv.org/abs/2010.01412
    
    Improves model generalization by minimizing both loss value and loss sharpness.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        """
        Args:
            params: Model parameters
            base_optimizer: Base optimizer class (e.g., torch.optim.AdamW)
            rho: Neighborhood size (default: 0.05)
            adaptive: Use adaptive SAM (default: False)
        """
        assert rho >= 0.0, f"Invalid rho: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step: Compute gradient and climb to local maximum.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # Climb to the local maximum "w + e(w)"
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step: Update weights with base optimizer.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # Get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # Do the actual "sharpness-aware" update
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Standard step (not used in SAM workflow).
        SAM requires first_step() and second_step() to be called explicitly.
        """
        raise NotImplementedError("SAM requires first_step() and second_step() to be called explicitly.")
    
    def _grad_norm(self):
        """
        Compute gradient norm for scaling.
        """
        shared_device = self.param_groups[0]["params"][0].device  # Put everything on the same device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class MobileViTLoRATrainer:
    """
    Trainer for MobileViT with LoRA/AdaLoRA adaptation.
    Designed for Federated Learning on edge devices.
    
    Features:
    - Feature A: Sharpness-Aware Minimization (SAM) for better generalization
    - Feature C: Test-Time Adaptation (TTA) for handling distribution shifts
    - Feature D: AdaLoRA for dynamic rank allocation
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
        network_monitor: Optional[Any] = None,
        # Feature A: SAM
        use_sam: bool = False,
        sam_rho: float = 0.05,
        # Feature C: TTA
        use_tta: bool = False,
        tta_steps: int = 1,
        tta_lr: float = 1e-4,
        # Feature D: LoRA parameters
        use_lora: bool = False,
    ):
        self.num_classes = num_classes
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_mixed_precision = use_mixed_precision
        self.network_monitor = network_monitor
        
        # Feature flags
        self.use_sam = use_sam
        self.sam_rho = sam_rho
        self.use_tta = use_tta
        self.tta_steps = tta_steps
        self.tta_lr = tta_lr
        self.use_lora = use_lora
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing MobileViTLoRATrainer on {self.device}")
        logger.info(f"  SAM: {use_sam}, TTA: {use_tta}, LoRA: {use_lora}")
        
        self.model = self._build_model()
        self.model.to(self.device)
        self._inject_gating()
        self._freeze_backbone()
        
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and self.device.type == "cuda" else None
        
        self.stats = {
            'total_params': self._count_parameters(),
            'trainable_params': self._count_trainable_parameters(),
        }
        logger.info(f"Model initialized: {self.stats}")
    
    def _build_model(self) -> nn.Module:
        """
        Build model with optional AdaLoRA integration.
        
        Feature D: If use_lora=True, wraps the base model with LoRA adapter.
        Otherwise, uses standard model (with manual gating injection later).
        """
        logger.info(f"Initializing MobileViTv2 (timm) for {self.num_classes} classes")
        base_model = timm.create_model('mobilevitv2_050.cvnets_in1k', pretrained=True, num_classes=self.num_classes)
        
        # Feature D: LoRA Integration
        if self.use_lora and HAS_PEFT and LoraConfig is not None:
            logger.info(f"Applying LoRA: r={self.lora_r}, alpha={self.lora_alpha}")
            
            # Configure LoRA
            # Note: LoRA only supports Linear layers, not Conv2d
            # MobileViTv2 uses mostly Conv2d layers with only head.fc as Linear
            # We target the classifier head which is the main trainable component
            lora_config = LoraConfig(  # type: ignore
                r=self.lora_r,  # LoRA rank
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["head.fc"],  # Only Linear layer in MobileViTv2
                modules_to_save=[],  # head.fc is already targeted
            )
            
            # Wrap model with PEFT
            model = get_peft_model(base_model, lora_config)  # type: ignore
            logger.info("LoRA applied successfully (Note: MobileViTv2 has limited Linear layers)")
            model.print_trainable_parameters()  # type: ignore
            
            return model
        
        elif self.use_lora and not HAS_PEFT:
            logger.warning("LoRA requested but peft not installed. Using standard model.")
        
        return base_model

    def _inject_gating(self):
        logger.info("Injecting GatedBlockWrapper into MobileViT blocks...")
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
            
        logger.info(f"Replaced {len(modules_to_replace)} blocks with GatedBlockWrapper")

    def _freeze_backbone(self):
        logger.info("Freezing backbone parameters...")
        for param in self.model.parameters():
            param.requires_grad = False
            
        for name, module in self.model.named_modules():
            if isinstance(module, GatedBlockWrapper):
                for param in module.expert.parameters():
                    param.requires_grad = True
                for param in module.gate.parameters():
                    param.requires_grad = True
            
        if hasattr(self.model, 'head'):
            for param in self.model.head.parameters():  # type: ignore
                param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():  # type: ignore
                param.requires_grad = True
                
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
    
    def _count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_parameters(self) -> List[np.ndarray]:
        """
        Extract trainable parameters.
        
        Feature D: For LoRA, correctly extracts adapter weights.
        """
        parameters = []
        
        # For LoRA models wrapped with PEFT
        if self.use_lora and HAS_PEFT and PeftModel is not None and isinstance(self.model, PeftModel):
            # Extract only adapter parameters (LoRA weights)
            for name, param in self.model.named_parameters():
                if param.requires_grad and ('lora' in name.lower() or 'adapter' in name.lower()):
                    parameters.append(param.detach().cpu().numpy())
            logger.debug(f"Extracted {len(parameters)} LoRA adapter parameters")
        else:
            # Standard extraction for non-AdaLoRA models
            for param in self.model.parameters():
                if param.requires_grad:
                    parameters.append(param.detach().cpu().numpy())
        
        return parameters

    def get_adaptive_parameters(self) -> List[np.ndarray]:
        score = 1.0
        if self.network_monitor:
            score = self.network_monitor.get_network_score()
        
        if score >= 0.8:
            keep_ratio = 1.0
        else:
            keep_ratio = 0.1 + (0.9 * (score / 0.8))
            
        logger.info(f"Network Score: {score:.2f} -> Pruning Keep Ratio: {keep_ratio:.2%}")

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
        """
        Set trainable parameters.
        
        Feature D: For LoRA, correctly sets adapter weights.
        Ensures shape compatibility before assignment.
        """
        param_idx = 0
        
        # For LoRA models wrapped with PEFT
        if self.use_lora and HAS_PEFT and PeftModel is not None and isinstance(self.model, PeftModel):
            for name, param in self.model.named_parameters():
                if param.requires_grad and ('lora' in name.lower() or 'adapter' in name.lower()):
                    if param_idx >= len(parameters):
                        logger.warning(f"Not enough parameters to set. Expected more than {param_idx}")
                        break
                    
                    new_param = torch.from_numpy(parameters[param_idx]).to(self.device)
                    
                    # Verify shape compatibility
                    if new_param.shape == param.shape:
                        param.data = new_param
                    else:
                        logger.warning(f"Shape mismatch for {name}: expected {param.shape}, got {new_param.shape}. Skipping.")
                    
                    param_idx += 1
        else:
            # Standard setting for non-AdaLoRA models
            for param in self.model.parameters():
                if param.requires_grad:
                    if param_idx >= len(parameters):
                        break
                    param.data = torch.from_numpy(parameters[param_idx]).to(self.device)
                    param_idx += 1
        
        logger.debug(f"Set {param_idx} parameters")
    
    def _simulate_network_artifacts(self, images: torch.Tensor, score: float) -> torch.Tensor:
        """
        Simulate compression artifacts (noise/blur) based on quality score.
        Lower score = More noise.
        """
        if score >= 0.9:
            return images
            
        # Intensity of noise (0.0 to 0.2)
        noise_level = (0.9 - score) * 0.5
        
        # Add Gaussian Noise
        noise = torch.randn_like(images) * noise_level
        noisy_images = images + noise
        
        return torch.clamp(noisy_images, 0.0, 1.0)

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        fedprox_mu: float = 0.0,       
        global_weights: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Train the model with optional SAM optimizer.
        
        Feature A: If use_sam=True, uses SAM wrapper around AdamW for better generalization.
        """
        self.model.train()
        
        # Feature A: SAM Optimizer
        if self.use_sam:
            logger.info(f"Using SAM optimizer (rho={self.sam_rho})")
            base_optimizer = torch.optim.AdamW
            optimizer: Any = SAM(
                self.model.parameters(), 
                base_optimizer,
                rho=self.sam_rho,
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        else:
            optimizer: Any = torch.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
        
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        for epoch in range(epochs):
            epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # --- SIMULATED GATING TRAINING ---
                # Randomly simulate network conditions to train the Gating Mechanism
                # 30% chance to simulate poor network during training
                if np.random.rand() < 0.3:
                    simulated_score = np.random.uniform(0.0, 0.8)
                else:
                    simulated_score = 1.0
                
                # Apply simulated score to model (to activate Gate)
                self.model.apply(lambda m: m.set_quality_score(simulated_score) if hasattr(m, 'set_quality_score') else None)  # type: ignore
                
                # Apply artifacts to input images (so Expert learns to handle noise)
                if simulated_score < 0.9:
                    images = self._simulate_network_artifacts(images, simulated_score)
                # ---------------------------------

                optimizer.zero_grad()
                
                # Feature A: SAM requires two forward-backward passes
                if self.use_sam:
                    # First forward-backward: Compute gradient and climb to local maximum
                    if self.use_mixed_precision and self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = F.cross_entropy(outputs, labels)
                            if fedprox_mu > 0 and global_weights:
                                loss += self._compute_proximal_loss(global_weights, fedprox_mu)
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer.base_optimizer)
                        optimizer.first_step(zero_grad=True)
                        
                        # Second forward-backward: Update at the perturbed point
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = F.cross_entropy(outputs, labels)
                            if fedprox_mu > 0 and global_weights:
                                loss += self._compute_proximal_loss(global_weights, fedprox_mu)
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer.base_optimizer)
                        self.scaler.update()
                        optimizer.second_step(zero_grad=True)
                    else:
                        # First forward-backward
                        outputs = self.model(images)
                        loss = F.cross_entropy(outputs, labels)
                        if fedprox_mu > 0 and global_weights:
                            loss += self._compute_proximal_loss(global_weights, fedprox_mu)
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        
                        # Second forward-backward
                        outputs = self.model(images)
                        loss = F.cross_entropy(outputs, labels)
                        if fedprox_mu > 0 and global_weights:
                            loss += self._compute_proximal_loss(global_weights, fedprox_mu)
                        loss.backward()
                        optimizer.second_step(zero_grad=True)
                
                # Standard optimization (non-SAM)
                else:
                    if self.use_mixed_precision and self.scaler:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            loss = F.cross_entropy(outputs, labels)
                            if fedprox_mu > 0 and global_weights:
                                loss += self._compute_proximal_loss(global_weights, fedprox_mu)
                        
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        outputs = self.model(images)
                        loss = F.cross_entropy(outputs, labels)
                        if fedprox_mu > 0 and global_weights:
                            loss += self._compute_proximal_loss(global_weights, fedprox_mu)
                        loss.backward()
                        optimizer.step()
                
                with torch.no_grad():
                    predictions = outputs.argmax(dim=1)
                    correct = (predictions == labels).sum().item()
                    epoch_loss += loss.item() * images.size(0)
                    epoch_correct += correct
                    epoch_samples += images.size(0)
            
            avg_loss = epoch_loss / epoch_samples
            accuracy = epoch_correct / epoch_samples
            logger.info(f"Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'num_samples': total_samples,
            'num_epochs': epochs,
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model with optional Test-Time Adaptation.
        
        Feature C: If use_tta=True, adapts BatchNorm statistics during inference
        to handle distribution shifts between training and test data.
        """
        # During evaluation, use ACTUAL network score if available
        real_score = 1.0
        if self.network_monitor:
            real_score = self.network_monitor.get_network_score()
        
        self.model.apply(lambda m: m.set_quality_score(real_score) if hasattr(m, 'set_quality_score') else None)  # type: ignore
        
        # Feature C: Test-Time Adaptation
        if self.use_tta:
            logger.info(f"Performing Test-Time Adaptation (steps={self.tta_steps}, lr={self.tta_lr})")
            self._test_time_adaptation(test_loader)
        
        # Standard evaluation
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                predictions = outputs.argmax(dim=1)
                correct = (predictions == labels).sum().item()
                total_loss += loss.item() * images.size(0)
                total_correct += correct
                total_samples += images.size(0)
        
        metrics = {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'num_samples': total_samples,
        }
        logger.info(f"Evaluation (Score={real_score:.2f}, TTA={self.use_tta}): Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}")
        return metrics
    
    def _test_time_adaptation(self, test_loader: DataLoader) -> None:
        """
        Feature C: Test-Time Adaptation
        
        Adapts BatchNorm running statistics to the test distribution.
        This helps the model handle distribution shifts without full retraining.
        
        Strategy:
        1. Set model to train mode (to update BN running stats)
        2. Freeze all parameters except BN parameters
        3. Run a few passes through test data
        4. Restore model to eval mode
        """
        # Save original training state
        original_training = self.model.training
        
        # Set to train mode for BN updates
        self.model.train()
        
        # Freeze all parameters except BN
        frozen_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                frozen_params.append((name, param))
                param.requires_grad = False
        
        # Unfreeze BN parameters
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for param in module.parameters():
                    param.requires_grad = True
        
        # Create lightweight optimizer for BN parameters
        bn_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(bn_params) > 0:
            optimizer = torch.optim.SGD(bn_params, lr=self.tta_lr)
            
            # Run TTA passes
            for step in range(self.tta_steps):
                for images, _ in test_loader:
                    images = images.to(self.device)
                    optimizer.zero_grad()
                    
                    # Forward pass to update BN stats (need gradients enabled!)
                    if self.use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(images)
                            # Minimize entropy for unsupervised adaptation
                            loss = self._entropy_loss(outputs)
                    else:
                        outputs = self.model(images)
                        # Minimize entropy for unsupervised adaptation
                        loss = self._entropy_loss(outputs)
                    
                    if self.use_mixed_precision and self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
        
        # Restore original parameter states
        for name, param in frozen_params:
            param.requires_grad = True
        
        # Restore original training mode
        self.model.train(original_training)
        
        logger.info(f"TTA completed: Updated {len(bn_params)} BN parameters")
    
    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy loss for test-time adaptation.
        Minimizing entropy encourages confident predictions.
        """
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        return entropy.mean()
    
    def _compute_proximal_loss(self, global_weights: List[np.ndarray], mu: float) -> torch.Tensor:
        proximal_term = torch.tensor(0.0, device=self.device)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) != len(global_weights): 
            return torch.tensor(0.0, device=self.device)
        
        for param, global_w in zip(trainable_params, global_weights):
            global_w_tensor = torch.tensor(global_w, device=self.device)
            proximal_term = proximal_term + (param - global_w_tensor).norm(2) ** 2
        return torch.tensor(mu / 2.0, device=self.device) * proximal_term

def create_dummy_dataset(num_samples=100, num_classes=10, image_size=224):
    train_images = torch.randn(num_samples, 3, image_size, image_size)
    train_labels = torch.randint(0, num_classes, (num_samples,))
    test_images = torch.randn(num_samples // 5, 3, image_size, image_size)
    test_labels = torch.randint(0, num_classes, (num_samples // 5,))
    return DataLoader(TensorDataset(train_images, train_labels), batch_size=32, shuffle=True), \
           DataLoader(TensorDataset(test_images, test_labels), batch_size=32, shuffle=False)

if __name__ == "__main__":
    trainer = MobileViTLoRATrainer(num_classes=10, lora_r=4, use_mixed_precision=False)
    train_loader, test_loader = create_dummy_dataset(num_samples=50)
    metrics = trainer.train(train_loader, epochs=1)