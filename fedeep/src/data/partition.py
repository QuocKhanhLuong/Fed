"""
Dataset partitioning strategies for federated learning.

Supports:
  - Dirichlet partition (non-IID, controlled by alpha)
  - Pathological partition (each client gets only a few classes)
  - Partition statistics logging
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("fedeep.partition")


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    """
    Distribute samples across clients using Dirichlet(alpha) per class.

    Lower alpha = more heterogeneous (non-IID).
    alpha -> inf = IID.

    Args:
        labels:      1D array of integer class labels.
        num_clients: Number of FL clients (N).
        alpha:       Dirichlet concentration parameter.
        seed:        RNG seed for reproducibility.

    Returns:
        List of N index lists, one per client.
    """
    rng = np.random.RandomState(seed)
    num_classes = int(labels.max()) + 1
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        proportions = rng.dirichlet([alpha] * num_clients)
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        splits = np.split(class_idx, split_points)

        for k, split in enumerate(splits):
            client_indices[k].extend(split.tolist())

    return client_indices


def pathological_partition(
    labels: np.ndarray,
    num_clients: int,
    classes_per_client: int = 2,
    seed: int = 42,
) -> List[List[int]]:
    """
    Each client receives samples from only a few classes.

    Classes are assigned round-robin across clients (each client gets
    `classes_per_client` classes), then samples are split evenly among
    clients sharing each class.

    Args:
        labels:             1D array of integer class labels.
        num_clients:        Number of FL clients.
        classes_per_client: How many classes each client sees.
        seed:               RNG seed.

    Returns:
        List of N index lists, one per client.
    """
    rng = np.random.RandomState(seed)
    num_classes = int(labels.max()) + 1
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    class_order = rng.permutation(num_classes)

    # Assign classes to clients round-robin
    client_classes: List[List[int]] = [[] for _ in range(num_clients)]
    total_slots = num_clients * classes_per_client
    for slot in range(total_slots):
        client_id = slot % num_clients
        class_id = class_order[slot % num_classes]
        if class_id not in client_classes[client_id]:
            client_classes[client_id].append(class_id)

    # For each class, split its samples among assigned clients
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        owners = [k for k in range(num_clients) if c in client_classes[k]]
        if not owners:
            continue

        splits = np.array_split(class_idx, len(owners))
        for owner, split in zip(owners, splits):
            client_indices[owner].extend(split.tolist())

    return client_indices


def log_partition_stats(
    client_indices: List[List[int]],
    labels: np.ndarray,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Log and optionally save partition statistics.

    Reports:
      - Samples per client (min, max, mean, std)
      - Class distribution per client (num unique classes, histogram)

    Args:
        client_indices: List of index lists, one per client.
        labels:         Full label array (to compute class distributions).
        save_path:      If provided, save stats to this JSON file.

    Returns:
        Dict with partition statistics.
    """
    num_clients = len(client_indices)
    num_classes = int(labels.max()) + 1

    sizes = [len(idx) for idx in client_indices]
    stats = {
        "num_clients": num_clients,
        "num_classes": num_classes,
        "total_samples": sum(sizes),
        "samples_per_client": {
            "min": int(min(sizes)),
            "max": int(max(sizes)),
            "mean": float(np.mean(sizes)),
            "std": float(np.std(sizes)),
        },
        "per_client": [],
    }

    for i, idx in enumerate(client_indices):
        client_labels = labels[idx]
        unique, counts = np.unique(client_labels, return_counts=True)
        class_hist = {int(c): int(n) for c, n in zip(unique, counts)}
        stats["per_client"].append({
            "client_id": i,
            "num_samples": len(idx),
            "num_classes": len(unique),
            "class_histogram": class_hist,
        })

    logger.info(
        f"Partition: {num_clients} clients, "
        f"{stats['total_samples']} total samples, "
        f"per-client: {stats['samples_per_client']['min']}-"
        f"{stats['samples_per_client']['max']} "
        f"(mean={stats['samples_per_client']['mean']:.0f})"
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Partition stats saved to {save_path}")

    return stats
