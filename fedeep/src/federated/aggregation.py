"""
Aggregation strategies for federated learning.

Implements:
  - fedavg:          Standard FedAvg (sample-weighted averaging)
  - edpa_aggregate:  Exit-Depth-aware Personalized Aggregation (proposed)

FedProx uses the same aggregation as FedAvg; the proximal term is applied
client-side in the trainer, so no separate aggregation function is needed.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("fedeep.aggregation")


def fedavg(
    client_weights: List[OrderedDict],
    client_sizes: List[int],
) -> OrderedDict:
    """
    Standard Federated Averaging: sample-weighted mean of client parameters.

    w_global = sum_i (n_i / N) * w_i

    Args:
        client_weights: List of state_dicts from participating clients.
        client_sizes:   List of dataset sizes n_i for each client.

    Returns:
        Aggregated global state_dict.
    """
    total = sum(client_sizes)
    weights = [n / total for n in client_sizes]

    keys = client_weights[0].keys()
    aggregated = OrderedDict()

    for key in keys:
        aggregated[key] = sum(
            w * client_weights[i][key].float()
            for i, w in enumerate(weights)
        )

    return aggregated


def build_param_depth_map(model) -> Dict[int, int]:
    """
    Build {param_index: depth_k} mapping for EDPA.

    depth 0 = backbone parameters
    depth 1..4 = exit1..exit4 parameters

    Args:
        model: ConvNeXtEarlyExit instance.

    Returns:
        Dict mapping parameter index to depth level.
    """
    exit_param_ids = {}
    for k, head in enumerate(
        [model.exit1, model.exit2, model.exit3, model.exit4], start=1
    ):
        for p in head.parameters():
            exit_param_ids[id(p)] = k

    depth_map = {}
    for i, (name, p) in enumerate(model.named_parameters()):
        depth_map[i] = exit_param_ids.get(id(p), 0)

    return depth_map


def edpa_aggregate(
    client_weights: List[OrderedDict],
    client_sizes: List[int],
    client_ids: List[int],
    param_depth_map: Dict[int, int],
    gamma: float = 0.5,
    num_exits: int = 4,
    client_states: Optional[Dict[int, OrderedDict]] = None,
) -> Tuple[OrderedDict, Dict[int, OrderedDict]]:
    """
    Exit-Depth-aware Personalized Aggregation.

    Per parameter group k:
        w_k_global = lambda_k * FedAvg(w_k) + (1 - lambda_k) * w_k_local_prev
    where:
        lambda_k = 1 / (1 + gamma * k)
        k=0 (backbone), k=1..4 (exit heads)

    lambda values (default gamma=0.5):
        k=0 backbone: lambda=1.00  (full global)
        k=1 exit1:    lambda=0.67
        k=2 exit2:    lambda=0.50
        k=3 exit3:    lambda=0.40
        k=4 exit4:    lambda=0.33  (most personalized)

    Args:
        client_weights:   List of state_dicts from participating clients.
        client_sizes:     List of dataset sizes for weighting.
        client_ids:       List of client integer IDs (for state tracking).
        param_depth_map:  {param_index: depth_k} from build_param_depth_map.
        gamma:            Personalization curve parameter.
        num_exits:        Number of exit heads.
        client_states:    Previous per-client weights. None on first round.

    Returns:
        (global_weights, updated_client_states)
    """
    if client_states is None:
        client_states = {}

    lambdas = {k: 1.0 / (1.0 + gamma * k) for k in range(num_exits + 1)}

    # Step 1: compute FedAvg as the base
    fedavg_weights = fedavg(client_weights, client_sizes)

    keys = list(fedavg_weights.keys())

    # Step 2: per-client EDPA mixing
    updated_states: Dict[int, OrderedDict] = {}
    all_mixed: List[OrderedDict] = []

    for idx, cid in enumerate(client_ids):
        prev = client_states.get(cid, client_weights[idx])
        mixed = OrderedDict()

        for i, key in enumerate(keys):
            depth_k = param_depth_map.get(i, 0)
            lam = lambdas.get(depth_k, lambdas[num_exits])
            mixed[key] = (
                lam * fedavg_weights[key].float()
                + (1.0 - lam) * prev[key].float()
            )

        updated_states[cid] = mixed
        all_mixed.append(mixed)

    # Step 3: global model = average of all personalized weights
    global_weights = OrderedDict()
    for key in keys:
        global_weights[key] = torch.stack(
            [m[key] for m in all_mixed]
        ).mean(dim=0)

    lam_str = " | ".join(f"k={k} λ={lam:.2f}" for k, lam in sorted(lambdas.items()))
    logger.info(f"[EDPA] {lam_str}")

    return global_weights, updated_states
