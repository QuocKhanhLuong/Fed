"""
Data Manager for FL-QUIC-LoRA
Handles dataset loading with Non-IID partitioning for Federated Learning

Supports:
- CIFAR-10/100
- MedMNIST (PathMNIST, DermaMNIST, etc.)
- Dirichlet Non-IID partitioning for realistic FL scenarios

Author: Research Team - FL-QUIC-LoRA Project
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from typing import Tuple, Dict, Optional, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import MedMNIST
try:
    import medmnist
    from medmnist import INFO
    HAS_MEDMNIST = True
except ImportError:
    HAS_MEDMNIST = False
    medmnist = None  # type: ignore
    INFO = None  # type: ignore
    logger.warning("medmnist not installed - MedMNIST datasets unavailable")


def get_transforms(image_size: int = 224, is_grayscale: bool = False) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get standard transforms for training and testing.
    
    CRITICAL: MobileViT requires 224x224 RGB images.
    - Resize all images to 224x224
    - Convert grayscale to RGB (3 channels)
    
    Args:
        image_size: Target image size (default: 224 for MobileViT)
        is_grayscale: Whether the source images are grayscale
        
    Returns:
        Tuple of (train_transform, test_transform)
    """
    # Base transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with augmentation
    train_transform_list: list = [
        transforms.Resize((image_size, image_size)),
    ]
    
    # Convert grayscale to RGB if needed
    if is_grayscale:
        train_transform_list.append(transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x))
    
    train_transform_list.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Test transforms (no augmentation)
    test_transform_list: list = [
        transforms.Resize((image_size, image_size)),
    ]
    
    if is_grayscale:
        test_transform_list.append(transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x))
    
    test_transform_list.extend([
        transforms.ToTensor(),
        normalize,
    ])
    
    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose(test_transform_list)
    
    return train_transform, test_transform


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42
) -> Dict[int, np.ndarray]:
    """
    Partition dataset using Dirichlet distribution for Non-IID split.
    
    Lower alpha = more skewed (heterogeneous) data distribution
    Higher alpha = more uniform (IID-like) distribution
    
    Typical values:
    - alpha = 0.1: Highly skewed (realistic edge FL)
    - alpha = 0.5: Moderately skewed
    - alpha = 1.0: Mildly skewed
    - alpha = 100: Nearly IID
    
    Args:
        labels: Array of labels for all samples
        num_clients: Number of clients to partition data across
        alpha: Dirichlet concentration parameter
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping client_id to array of sample indices
    """
    np.random.seed(seed)
    
    num_classes = len(np.unique(labels))
    label_distribution = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples to clients using Dirichlet
    for k in range(num_classes):
        # Get indices of samples with label k
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Balance proportions (ensure each client gets at least 1 sample per class if possible)
        proportions = np.array([p * (len(idx_k) < num_clients) + 
                               (1 / num_clients) * (len(idx_k) >= num_clients) 
                               for p in proportions])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # Split indices according to proportions
        split_idx = np.split(idx_k, proportions)
        
        # Assign to clients
        for i in range(num_clients):
            label_distribution[i].extend(split_idx[i].tolist())
    
    # Convert to dictionary
    client_dict = {i: np.array(label_distribution[i]) for i in range(num_clients)}
    
    # Log distribution statistics
    for client_id, indices in client_dict.items():
        client_labels = labels[indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        logger.info(f"Client {client_id}: {len(indices)} samples, "
                   f"{len(unique)} classes, distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    return client_dict


def load_cifar_dataset(
    dataset_name: str,
    data_dir: str,
    client_id: int,
    num_clients: int,
    partition_type: str = "dirichlet",
    alpha: float = 0.5,
    batch_size: int = 32,
    num_workers: int = 2,
    train_split: float = 0.8,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Load CIFAR-10 or CIFAR-100 dataset with Non-IID partitioning.
    
    Args:
        dataset_name: "cifar10" or "cifar100"
        data_dir: Directory to store/load dataset
        client_id: ID of this client (0 to num_clients-1)
        num_clients: Total number of clients
        partition_type: "dirichlet" or "iid"
        alpha: Dirichlet concentration parameter
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        train_split: Fraction of local data for training (rest for validation)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset_stats)
    """
    logger.info(f"Loading {dataset_name.upper()} dataset for client {client_id}/{num_clients}")
    
    # Select dataset
    if dataset_name.lower() == "cifar10":
        dataset_class = torchvision.datasets.CIFAR10
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
        dataset_class = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get transforms
    train_transform, test_transform = get_transforms(image_size=224, is_grayscale=False)
    
    # Load full datasets
    train_dataset = dataset_class(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = dataset_class(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Get labels for partitioning
    if hasattr(train_dataset, 'targets'):
        all_labels = np.array(train_dataset.targets)
    else:
        all_labels = np.array([label for _, label in train_dataset])
    
    # Partition data among clients
    if partition_type == "dirichlet":
        client_indices = dirichlet_partition(all_labels, num_clients, alpha)
    elif partition_type == "iid":
        # Simple IID split
        indices = np.arange(len(train_dataset))
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        client_indices = {i: splits[i] for i in range(num_clients)}
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")
    
    # Get this client's indices
    local_indices = client_indices[client_id]
    
    # Split local data into train/val
    np.random.shuffle(local_indices)
    split_point = int(len(local_indices) * train_split)
    train_indices = local_indices[:split_point]
    val_indices = local_indices[split_point:]
    
    # Create subset datasets (convert numpy arrays to lists for type safety)
    train_subset = Subset(train_dataset, train_indices.tolist())
    val_subset = Subset(train_dataset, val_indices.tolist())
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Dataset statistics
    stats = {
        'num_classes': num_classes,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'test_samples': len(test_dataset),
        'image_size': 224,
        'num_channels': 3,
    }
    
    logger.info(f"✓ Dataset loaded: {stats['train_samples']} train, "
               f"{stats['val_samples']} val, {stats['test_samples']} test")
    
    return train_loader, val_loader, test_loader, stats


def load_medmnist_dataset(
    dataset_name: str,
    data_dir: str,
    client_id: int,
    num_clients: int,
    partition_type: str = "dirichlet",
    alpha: float = 0.5,
    batch_size: int = 32,
    num_workers: int = 2,
    train_split: float = 0.8,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Load MedMNIST dataset with Non-IID partitioning.
    
    Supported datasets: pathmnist, dermamnist, octmnist, pneumoniamnist, etc.
    See: https://medmnist.com/
    
    Args:
        dataset_name: MedMNIST dataset name (e.g., "pathmnist")
        data_dir: Directory to store/load dataset
        client_id: ID of this client
        num_clients: Total number of clients
        partition_type: "dirichlet" or "iid"
        alpha: Dirichlet concentration parameter
        batch_size: Batch size
        num_workers: Number of workers
        train_split: Train/val split ratio
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset_stats)
    """
    if not HAS_MEDMNIST:
        raise ImportError("medmnist not installed. Install with: pip install medmnist>=3.0.1")
    
    logger.info(f"Loading {dataset_name.upper()} dataset for client {client_id}/{num_clients}")
    
    # Get dataset info
    if INFO is None:
        raise RuntimeError("MedMNIST INFO not available")
    info = INFO[dataset_name]
    num_classes = len(info['label'])
    is_grayscale = (info['n_channels'] == 1)
    
    # Get transforms
    train_transform, test_transform = get_transforms(image_size=224, is_grayscale=is_grayscale)
    
    # Get dataset class
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load datasets
    train_dataset = DataClass(
        split='train',
        transform=train_transform,
        download=True,
        root=data_dir
    )
    
    test_dataset = DataClass(
        split='test',
        transform=test_transform,
        download=True,
        root=data_dir
    )
    
    # Get labels (MedMNIST labels are 2D arrays)
    all_labels = train_dataset.labels.squeeze()
    
    # Partition data
    if partition_type == "dirichlet":
        client_indices = dirichlet_partition(all_labels, num_clients, alpha)
    else:
        indices = np.arange(len(train_dataset))
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        client_indices = {i: splits[i] for i in range(num_clients)}
    
    # Get local indices and split
    local_indices = client_indices[client_id]
    np.random.shuffle(local_indices)
    split_point = int(len(local_indices) * train_split)
    train_indices = local_indices[:split_point]
    val_indices = local_indices[split_point:]
    
    # Create subsets (convert numpy arrays to lists for type safety)
    train_subset = Subset(train_dataset, train_indices.tolist())
    val_subset = Subset(train_dataset, val_indices.tolist())
    
    # Create loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    stats = {
        'num_classes': num_classes,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'test_samples': len(test_dataset),
        'image_size': 224,
        'num_channels': 3,  # Always 3 after conversion
    }
    
    logger.info(f"✓ MedMNIST loaded: {stats['train_samples']} train, "
               f"{stats['val_samples']} val, {stats['test_samples']} test")
    
    return train_loader, val_loader, test_loader, stats


def load_dataset(
    dataset_name: str,
    data_dir: str,
    client_id: int,
    num_clients: int,
    partition_type: str = "dirichlet",
    alpha: float = 0.5,
    batch_size: int = 32,
    num_workers: int = 2,
    train_split: float = 0.8,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Universal dataset loader with automatic dataset detection.
    
    Supports:
    - CIFAR-10/100
    - MedMNIST family (PathMNIST, DermaMNIST, etc.)
    
    Args:
        dataset_name: Dataset name (cifar10, cifar100, pathmnist, etc.)
        data_dir: Data directory
        client_id: Client ID (0 to num_clients-1)
        num_clients: Total number of clients
        partition_type: "dirichlet" or "iid"
        alpha: Dirichlet concentration parameter (lower = more skew)
        batch_size: Batch size
        num_workers: DataLoader workers
        train_split: Train/val split ratio
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset_stats)
    """
    dataset_name = dataset_name.lower()
    
    # Create data directory
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Route to appropriate loader
    if dataset_name in ['cifar10', 'cifar100']:
        return load_cifar_dataset(
            dataset_name=dataset_name,
            data_dir=data_dir,
            client_id=client_id,
            num_clients=num_clients,
            partition_type=partition_type,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=train_split,
        )
    elif HAS_MEDMNIST and INFO is not None and dataset_name in INFO:
        return load_medmnist_dataset(
            dataset_name=dataset_name,
            data_dir=data_dir,
            client_id=client_id,
            num_clients=num_clients,
            partition_type=partition_type,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=train_split,
        )
    else:
        # Fallback to CIFAR-100
        logger.warning(f"Unknown dataset '{dataset_name}', falling back to CIFAR-100")
        return load_cifar_dataset(
            dataset_name='cifar100',
            data_dir=data_dir,
            client_id=client_id,
            num_clients=num_clients,
            partition_type=partition_type,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=train_split,
        )


# Example usage
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Data Manager Demo")
    logger.info("="*60)
    
    # Test CIFAR-100 with Dirichlet partitioning
    train_loader, val_loader, test_loader, stats = load_dataset(
        dataset_name="cifar100",
        data_dir="./data",
        client_id=0,
        num_clients=5,
        partition_type="dirichlet",
        alpha=0.5,
        batch_size=32,
    )
    
    logger.info(f"\nDataset stats: {stats}")
    
    # Check batch
    for images, labels in train_loader:
        logger.info(f"Batch shape: {images.shape}, Labels: {labels.shape}")
        logger.info(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    logger.info("\n" + "="*60)
    logger.info("✅ Data Manager demo completed!")
    logger.info("="*60)
