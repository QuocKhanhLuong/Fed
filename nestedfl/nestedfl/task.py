"""
FedEEP task.py — Model and data utilities (slim version)

Delegates to:
  - models.convnext_early_exit.ConvNeXtEarlyExit
  - nestedfl.data.cifar100.CIFAR100FederatedDataset
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_model(num_classes: int = 100, pretrained: bool = True) -> nn.Module:
    """Return ConvNeXtEarlyExit model."""
    from models.convnext_early_exit import ConvNeXtEarlyExit
    return ConvNeXtEarlyExit(num_classes=num_classes, pretrained=pretrained)


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset: str = "cifar100",
    alpha: float = 0.5,
    batch_size: int = 32,
):
    """Return (trainloader, testloader) for the given client partition."""
    if dataset == "cifar100":
        from nestedfl.data.cifar100 import CIFAR100FederatedDataset
        data = CIFAR100FederatedDataset(
            num_partitions=num_partitions,
            alpha=alpha,
            batch_size=batch_size,
        )
        return data.get_partition(partition_id)
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'. Only 'cifar100' supported.")


def test(
    model: nn.Module,
    testloader: DataLoader,
    device: torch.device,
    threshold: float = 0.8,
):
    """Evaluate model on test set. Returns (loss, accuracy)."""
    model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            if hasattr(model, "forward"):
                logits, _ = model(images, threshold=threshold)
            else:
                logits = model(images)

            total_loss += F.cross_entropy(logits, labels).item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy
