"""
Global model evaluation for the FL server.

Provides:
  - evaluate_global:   Overall accuracy + loss using early-exit inference
  - evaluate_per_exit: Per-exit accuracy (all 4 exits independently)
"""

import logging
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("fedeep.evaluator")


@torch.no_grad()
def evaluate_global(
    model: nn.Module,
    state_dict: OrderedDict,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.8,
) -> Dict[str, float]:
    """
    Evaluate global model on the test set with early-exit inference.

    Args:
        model:       ConvNeXtEarlyExit instance (weights will be loaded).
        state_dict:  Global state_dict to load.
        test_loader: Shared test DataLoader.
        device:      Evaluation device.
        threshold:   Confidence threshold for early exit.

    Returns:
        Dict with loss, accuracy, exit_distribution, avg_exit.
    """
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    exit_counts = [0, 0, 0, 0]

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, exits = model(images, threshold=threshold)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)
        for e in exits.cpu().tolist():
            exit_counts[e] += 1

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "exit_distribution": exit_counts,
        "avg_exit": sum(
            i * c for i, c in enumerate(exit_counts)
        ) / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate_per_exit(
    model: nn.Module,
    state_dict: OrderedDict,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate each exit head independently (no early stopping).

    Returns dict: {"exit1_acc": ..., ..., "exit4_acc": ..., "exit1_loss": ..., ...}
    """
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    exit_correct = [0, 0, 0, 0]
    exit_loss = [0.0, 0.0, 0.0, 0.0]
    total = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits_all = model.forward_all_exits(images)
        for k, logits_k in enumerate(logits_all):
            exit_correct[k] += (logits_k.argmax(1) == labels).sum().item()
            exit_loss[k] += F.cross_entropy(logits_k, labels).item() * labels.size(0)
        total += labels.size(0)

    metrics = {}
    for k in range(4):
        metrics[f"exit{k+1}_acc"] = exit_correct[k] / max(total, 1)
        metrics[f"exit{k+1}_loss"] = exit_loss[k] / max(total, 1)

    return metrics
