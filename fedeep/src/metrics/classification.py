"""
Classification metrics: accuracy, per-exit accuracy, confusion matrix.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def topk_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
) -> float:
    """Top-k accuracy."""
    _, topk_preds = logits.topk(k, dim=-1)
    correct = topk_preds.eq(labels.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


@torch.no_grad()
def per_exit_accuracy(
    model,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute accuracy for each exit head independently.

    Returns dict: {"exit1_acc": ..., "exit2_acc": ..., ...}
    """
    model.eval()
    exit_correct = [0, 0, 0, 0]
    total = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        y1, y2, y3, y4 = model.forward_all_exits(images)
        for k, logits_k in enumerate([y1, y2, y3, y4]):
            exit_correct[k] += (logits_k.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return {
        f"exit{k+1}_acc": c / max(total, 1)
        for k, c in enumerate(exit_correct)
    }


@torch.no_grad()
def confusion_matrix(
    model,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix using the final exit.

    Returns (num_classes, num_classes) ndarray.
    """
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _ = model(images, threshold=1.0)  # force final exit
        preds = logits.argmax(1).cpu().numpy()
        targets = labels.cpu().numpy()
        for t, p in zip(targets, preds):
            cm[t, p] += 1

    return cm
