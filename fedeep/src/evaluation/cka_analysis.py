"""
Centered Kernel Alignment (CKA) analysis.

Computes linear CKA similarity between feature representations from:
  - Different exits of the same model
  - Same exit across different clients (representation divergence)

Used for paper figures showing how features diverge/converge during FL.

Reference: Kornblith et al., "Similarity of Neural Network Representations
Revisited" (ICML 2019).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("fedeep.cka")


def _centering_matrix(n: int) -> np.ndarray:
    """H = I - (1/n) * 11^T"""
    return np.eye(n) - np.ones((n, n)) / n


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two feature matrices.

    Args:
        X: (n_samples, d1) feature matrix.
        Y: (n_samples, d2) feature matrix.

    Returns:
        CKA similarity in [0, 1].
    """
    n = X.shape[0]
    assert Y.shape[0] == n, "X and Y must have the same number of samples"

    H = _centering_matrix(n)

    K = X @ X.T
    L = Y @ Y.T

    HKH = H @ K @ H
    HLH = H @ L @ H

    hsic_xy = np.trace(HKH @ HLH) / ((n - 1) ** 2)
    hsic_xx = np.trace(HKH @ HKH) / ((n - 1) ** 2)
    hsic_yy = np.trace(HLH @ HLH) / ((n - 1) ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


@torch.no_grad()
def extract_exit_features(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000,
) -> List[np.ndarray]:
    """
    Extract intermediate feature representations from all 4 stages.

    Returns list of 4 arrays, each (n_samples, channels).
    Features are taken after GAP (global average pooling), before the
    classifier head.
    """
    model.eval()
    model.to(device)

    features_per_exit: List[List[np.ndarray]] = [[] for _ in range(4)]
    total = 0

    for images, _ in dataloader:
        if total >= max_samples:
            break
        images = images.to(device, non_blocking=True)
        batch_features = model.backbone(images)

        for k, feat in enumerate(batch_features):
            # GAP: (B, C, H, W) -> (B, C)
            pooled = feat.mean(dim=[2, 3]).cpu().numpy()
            features_per_exit[k].append(pooled)

        total += images.size(0)

    return [np.concatenate(f, axis=0)[:max_samples] for f in features_per_exit]


def cka_between_exits(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000,
) -> np.ndarray:
    """
    Compute CKA similarity matrix between all pairs of exits.

    Returns (4, 4) CKA matrix.
    """
    features = extract_exit_features(model, dataloader, device, max_samples)

    cka_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            cka_matrix[i, j] = linear_cka(features[i], features[j])

    return cka_matrix


def cka_between_clients(
    models: List[nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    exit_idx: int = 3,
    max_samples: int = 1000,
) -> np.ndarray:
    """
    Compute CKA similarity between client models at a specific exit.

    Args:
        models:     List of client models.
        dataloader: Shared test DataLoader.
        device:     Compute device.
        exit_idx:   Which exit to compare (0-3).
        max_samples: Max samples for feature extraction.

    Returns:
        (num_clients, num_clients) CKA matrix.
    """
    client_features = []
    for model in models:
        features = extract_exit_features(model, dataloader, device, max_samples)
        client_features.append(features[exit_idx])

    n = len(models)
    cka_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cka_matrix[i, j] = linear_cka(client_features[i], client_features[j])

    return cka_matrix
