"""
CIFAR-100 federated dataset loader with Dirichlet partitioning.

Usage:
    train_loaders, test_loader = make_federated_cifar100(
        num_clients=10, alpha=0.5, batch_size=32
    )
"""

from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .partition import dirichlet_partition

NUM_CLASSES = 100

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
])

_TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
])


def load_cifar100(
    data_dir: str = "./data",
) -> Tuple[datasets.CIFAR100, datasets.CIFAR100]:
    """
    Download and return raw CIFAR-100 train/test datasets.

    Returns:
        (train_dataset, test_dataset) with appropriate transforms.
    """
    train_dataset = datasets.CIFAR100(
        data_dir, train=True, download=True, transform=_TRAIN_TRANSFORM
    )
    test_dataset = datasets.CIFAR100(
        data_dir, train=False, download=True, transform=_TEST_TRANSFORM
    )
    return train_dataset, test_dataset


def make_federated_cifar100(
    num_clients: int = 10,
    alpha: float = 0.5,
    batch_size: int = 32,
    data_dir: str = "./data",
    seed: int = 42,
) -> Tuple[List[DataLoader], DataLoader, dict]:
    """
    Create federated CIFAR-100 data loaders with Dirichlet partitioning.

    Args:
        num_clients: Number of FL clients.
        alpha:       Dirichlet concentration (lower = more non-IID).
        batch_size:  Batch size for all loaders.
        data_dir:    Dataset cache directory.
        seed:        RNG seed for partition reproducibility.

    Returns:
        (train_loaders, test_loader, partition_info):
            train_loaders:  list of N DataLoaders (one per client)
            test_loader:    single shared test DataLoader
            partition_info: dict with "client_indices" and "labels"
    """
    train_dataset, test_dataset = load_cifar100(data_dir)

    labels = np.array(train_dataset.targets)
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
