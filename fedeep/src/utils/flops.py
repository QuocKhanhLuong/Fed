"""
Approximate FLOPs estimation per exit for ConvNeXt-Tiny early-exit model.

ConvNeXt-Tiny stages and approximate GFLOPs (224x224 input):
  Stage 1 (96ch,  3 blocks):  ~0.55 GFLOPs
  Stage 2 (192ch, 3 blocks):  ~0.75 GFLOPs  (cumulative: ~1.30)
  Stage 3 (384ch, 9 blocks):  ~2.25 GFLOPs  (cumulative: ~3.55)
  Stage 4 (768ch, 3 blocks):  ~0.95 GFLOPs  (cumulative: ~4.50)

For CIFAR (32x32, stride=1 stem), spatial dims are larger so actual FLOPs
differ, but the relative ratios between exits remain approximately the same.
"""

from typing import Dict, List


# Approximate cumulative GFLOPs at each exit (ConvNeXt-Tiny, 224x224)
CONVNEXT_TINY_CUMULATIVE_GFLOPS = [0.55, 1.30, 3.55, 4.50]

# Relative FLOPs ratio (normalized to Exit4 = 1.0)
CONVNEXT_TINY_FLOPS_RATIO = [
    g / CONVNEXT_TINY_CUMULATIVE_GFLOPS[-1]
    for g in CONVNEXT_TINY_CUMULATIVE_GFLOPS
]


def get_exit_flops_ratios() -> List[float]:
    """
    Return relative FLOPs ratio for each exit (normalized to Exit4 = 1.0).

    Exit 1: ~12%  (cheapest)
    Exit 2: ~29%
    Exit 3: ~79%
    Exit 4: 100%  (full model)
    """
    return CONVNEXT_TINY_FLOPS_RATIO


def estimate_avg_flops(
    exit_distribution: List[int],
    full_model_gflops: float = 4.50,
) -> float:
    """
    Estimate average GFLOPs per sample given an exit distribution.

    Args:
        exit_distribution: [count_exit1, count_exit2, count_exit3, count_exit4]
        full_model_gflops: Total GFLOPs for a full forward pass.

    Returns:
        Average GFLOPs per sample.
    """
    total_samples = sum(exit_distribution)
    if total_samples == 0:
        return full_model_gflops

    ratios = get_exit_flops_ratios()
    weighted_sum = sum(
        count * ratio * full_model_gflops
        for count, ratio in zip(exit_distribution, ratios)
    )
    return weighted_sum / total_samples


def flops_savings_percent(
    exit_distribution: List[int],
    full_model_gflops: float = 4.50,
) -> float:
    """
    Compute percentage of FLOPs saved compared to always using Exit4.

    Returns value in [0, 100].
    """
    avg = estimate_avg_flops(exit_distribution, full_model_gflops)
    return (1.0 - avg / full_model_gflops) * 100.0
