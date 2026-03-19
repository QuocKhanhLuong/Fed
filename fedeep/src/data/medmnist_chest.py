"""
ChestMNIST federated dataset loader (multi-label -> single-label adaptation).

ChestMNIST: 28x28 grayscale chest X-rays, 14 pathology labels.
For classification we use the multi-label setup directly or argmax
depending on config.

Images are resized to 32x32 and replicated to 3 channels for ConvNeXt.

Usage:
    train_loaders, test_loader = make_federated_chestmnist(
        num_clients=10, alpha=0.5
    )
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .partition import dirichlet_partition

NUM_CLASSES = 14


class _ChestMNISTWrapper(Dataset):
    """Wraps medmnist ChestMNIST to return (image_3ch_32x32, label_int)."""

    def __init__(self, medmnist_dataset, transform=None):
        self.dataset = medmnist_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.transform:
            img = self.transform(img)

        # Multi-label -> take first positive or class 0
        if isinstance(label, np.ndarray):
            nonzero = np.nonzero(label)[0]
            label = int(nonzero[0]) if len(nonzero) > 0 else 0
        elif isinstance(label, torch.Tensor):
            if label.dim() > 0:
                nonzero = label.nonzero(as_tuple=False)
                label = int(nonzero[0, 0]) if len(nonzero) > 0 else 0
            else:
                label = int(label.item())
        else:
            label = int(label)

        return img, label


_TRANSFORM = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def load_chestmnist(
    data_dir: str = "./data",
) -> Tuple[Dataset, Dataset]:
    """Load ChestMNIST train/test sets wrapped for single-label classification."""
    import medmnist
    from medmnist import ChestMNIST

    train_raw = ChestMNIST(split="train", download=True, root=data_dir)
    test_raw = ChestMNIST(split="test", download=True, root=data_dir)

    train_dataset = _ChestMNISTWrapper(train_raw, transform=_TRANSFORM)
    test_dataset = _ChestMNISTWrapper(test_raw, transform=_TRANSFORM)

    return train_dataset, test_dataset


def make_federated_chestmnist(
    num_clients: int = 10,
    alpha: float = 0.5,
    batch_size: int = 32,
    data_dir: str = "./data",
    seed: int = 42,
) -> Tuple[List[DataLoader], DataLoader, dict]:
    """
    Create federated ChestMNIST loaders with Dirichlet partitioning.

    Returns:
        (train_loaders, test_loader, partition_info)
    """
    train_dataset, test_dataset = load_chestmnist(data_dir)

    labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    client_indices = dirichlet_partition(labels, num_clients, alpha, seed)

    train_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        train_loaders.append(loader)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    partition_info = {
        "client_indices": client_indices,
        "labels": labels,
    }

    return train_loaders, test_loader, partition_info
