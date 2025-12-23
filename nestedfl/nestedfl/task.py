"""
nestedfl: Nested Early-Exit Federated Learning with Flower 1.18+

This module provides model, training, and evaluation utilities for
Nested Learning in Federated settings.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import from local nestedfl package
try:
    from nestedfl.nested_trainer import NestedEarlyExitTrainer
    HAS_TRAINER = True
except ImportError:
    HAS_TRAINER = False

# Import model from models folder (copied to nestedfl/)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))  # nestedfl/
    from models.early_exit_mobilevit import TimmPretrainedEarlyExit
    HAS_CUSTOM_MODEL = True
except ImportError:
    HAS_CUSTOM_MODEL = False
    print("Warning: Custom model not found, using simple CNN")


# =============================================================================
# Model Definition
# =============================================================================

class SimpleCNN(nn.Module):
    """Fallback simple CNN for testing."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def get_model(num_classes: int = 100, use_pretrained: bool = True):
    """
    Get model for FL training.
    
    Returns TimmPretrainedEarlyExit if available, else SimpleCNN.
    """
    if HAS_CUSTOM_MODEL and use_pretrained:
        return TimmPretrainedEarlyExit(num_classes=num_classes)
    else:
        return SimpleCNN(num_classes=num_classes)


def get_trainer(num_classes: int = 100, **kwargs):
    """
    Get NestedEarlyExitTrainer for FL.
    """
    if HAS_CUSTOM_MODEL:
        return NestedEarlyExitTrainer(
            num_classes=num_classes,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_mixed_precision=True,
            use_timm_pretrained=True,
            use_self_distillation=kwargs.get('use_distillation', True),
            cms_enabled=kwargs.get('cms_enabled', True),
            use_lss=kwargs.get('use_lss', True),
            fast_lr_multiplier=kwargs.get('fast_lr_mult', 3.0),
            slow_update_freq=kwargs.get('slow_update_freq', 5),
        )
    return None


# =============================================================================
# Data Loading with Dirichlet Partitioning
# =============================================================================

def load_data(partition_id: int, num_partitions: int, dataset: str = "cifar100"):
    """
    Load partitioned data for FL client.
    
    Uses Dirichlet partitioning for non-IID simulation.
    """
    # Transforms
    if dataset == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_data = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
        num_classes = 100
    else:  # cifar10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
        ])
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        num_classes = 10
    
    # Dirichlet partitioning
    import numpy as np
    np.random.seed(42)
    
    labels = np.array(train_data.targets)
    client_indices = [[] for _ in range(num_partitions)]
    alpha = 0.5  # Non-IID severity
    
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet([alpha] * num_partitions)
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, proportions)
        for k, split in enumerate(splits):
            client_indices[k].extend(split.tolist())
    
    # Get this client's partition
    indices = client_indices[partition_id]
    client_data = torch.utils.data.Subset(train_data, indices)
    
    trainloader = DataLoader(client_data, batch_size=32, shuffle=True, num_workers=0)
    testloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)
    
    return trainloader, testloader


# =============================================================================
# Training and Evaluation Functions
# =============================================================================

def train(model, trainloader, epochs: int, lr: float, device):
    """
    Train model on local data.
    
    Returns average training loss.
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        for batch in trainloader:
            # Handle both tuple and dict formats
            if isinstance(batch, dict):
                images = batch.get("img", batch.get("image"))
                labels = batch.get("label", batch.get("labels"))
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device).long()  # Ensure Long type
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle early-exit model: returns (logits, exit_indices) tuple
            if isinstance(outputs, (list, tuple)):
                if len(outputs) == 2 and outputs[1].dtype == torch.long:
                    outputs = outputs[0]  # (logits, exit_indices) -> logits
                else:
                    outputs = outputs[-1]  # [y1, y2, y3] -> y3
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def test(model, testloader, device):
    """
    Evaluate model on test data.
    
    Returns (loss, accuracy).
    """
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in testloader:
            # Handle both tuple and dict formats
            if isinstance(batch, dict):
                images = batch.get("img", batch.get("image"))
                labels = batch.get("label", batch.get("labels"))
            else:
                images, labels = batch
            
            images = images.to(device)
            labels = labels.to(device).long()  # Ensure Long type for CrossEntropyLoss
            
            outputs = model(images)
            
            # Handle early-exit model: returns (logits, exit_indices) tuple
            # or list of [exit1, exit2, exit3] logits from forward_all_exits()
            if isinstance(outputs, (list, tuple)):
                # If it's (logits, exit_indices) from forward(), take first element
                # If it's [y1, y2, y3] from forward_all_exits(), take last element
                if len(outputs) == 2 and outputs[1].dtype == torch.long:
                    outputs = outputs[0]  # (logits, exit_indices) -> logits
                else:
                    outputs = outputs[-1]  # [y1, y2, y3] -> y3
            
            # Ensure outputs are 2D (batch_size, num_classes)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0
    
    return avg_loss, accuracy


# For compatibility with template
Net = lambda: get_model(num_classes=100, use_pretrained=True)


if __name__ == "__main__":
    # Quick test
    print("Testing task.py...")
    model = get_model(num_classes=10, use_pretrained=False)
    print(f"Model: {model.__class__.__name__}")
    
    trainloader, testloader = load_data(partition_id=0, num_partitions=3, dataset="cifar10")
    print(f"Train samples: {len(trainloader.dataset)}")
    print(f"Test samples: {len(testloader.dataset)}")
    
    print("Done!")
