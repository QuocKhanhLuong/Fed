"""
OrganAMNIST federated dataset loader.

OrganAMNIST: 28x28 grayscale abdominal CT slices, 11 organ classes.
Images are resized to 32x32 and replicated to 3 channels for ConvNeXt.

Usage:
    train_loaders, test_loader = make_federated_organa(
        num_clients=10, alpha=0.5
    )
"""

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from .partition import dirichlet_partition

NUM_CLASSES = 11


class _OrganAWrapper(Dataset):
    """Wraps medmnist OrganAMNIST to return (image_3ch_32x32, label_int)."""

    def __init__(self, medmnist_dataset, transform=None):
        self.dataset = medmnist_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if self.transform:
            img = self.transform(img)

        if isinstance(label, np.ndarray):
            label = int(label.flatten()[0])
        elif isinstance(label, torch.Tensor):
            label = int(label.item()) if label.dim() == 0 else int(label[0].item())
        else:
            label = int(label)

        return img, label


_TRANSFORM = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def load_organa(
    data_dir: str = "./data",
) -> Tuple[Dataset, Dataset]:
    """Load OrganAMNIST train/test sets."""
    from medmnist import OrganAMNIST

    train_raw = OrganAMNIST(split="train", download=True, root=data_dir)
    test_raw = OrganAMNIST(split="test", download=True, root=data_dir)

    train_dataset = _OrganAWrapper(train_raw, transform=_TRANSFORM)
    test_dataset = _OrganAWrapper(test_raw, transform=_TRANSFORM)

    return train_dataset, test_dataset


def make_federated_organa(
    num_clients: int = 10,
    alpha: float = 0.5,
    batch_size: int = 32,
    data_dir: str = "./data",
    seed: int = 42,
) -> Tuple[List[DataLoader], DataLoader, dict]:
    """
    Create federated OrganAMNIST loaders with Dirichlet partitioning.

    Returns:
        (train_loaders, test_loader, partition_info)
    """
    train_dataset, test_dataset = load_organa(data_dir)

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
