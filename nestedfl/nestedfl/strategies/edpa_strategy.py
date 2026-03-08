"""
EDPA Strategy — Exit-Depth-aware Personalized Aggregation

The key contribution of FedEEP server-side.

Core idea:
  Different parameter groups warrant different levels of global aggregation.
  Shallow features (backbone) are universal → aggregate strongly (λ ≈ 1).
  Deep exit heads are client-specific → personalize more (λ ≈ 0.33).

Aggregation per group k:
    w_k_global = λ_k * FedAvg(w_k) + (1 - λ_k) * w_k_local_prev

where:
    λ_k = 1 / (1 + γ * k)      k=0 (backbone), k=1..4 (exits)
    w_k_local_prev = previous round's weights for this client (stored server-side)

λ values (default γ=0.5):
    k=0 backbone: λ=1.00   (full global — feature representation is universal)
    k=1 exit1:    λ=0.67   (mostly global)
    k=2 exit2:    λ=0.50   (balanced)
    k=3 exit3:    λ=0.40   (more personalized)
    k=4 exit4:    λ=0.33   (most personalized — final classifier)

Parameter depth mapping:
  Clients send `param_depth_map` in their metrics dict (only round 1).
  Server caches it. Format: {param_index: depth_k}
  
  If no metadata → fallback: first 70% params = backbone (k=0),
  last 30% split evenly across 4 exit heads (k=1..4).
"""

import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from .fedavg_strategy import FedAvgStrategy

logger = logging.getLogger(__name__)


class EDPAStrategy(FedAvgStrategy):
    """
    Exit-Depth-aware Personalized Aggregation strategy.

    Args:
        gamma:          Controls personalization curve (default: 0.5)
        num_exit_heads: Number of exit heads in the model (default: 4)
        **kwargs:       Forwarded to FedAvgStrategy
    """

    def __init__(
        self,
        gamma: float = 0.5,
        num_exit_heads: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.num_exit_heads = num_exit_heads

        # Precompute λ_k for k = 0 (backbone) to k = num_exit_heads
        self.lambdas = {
            k: 1.0 / (1.0 + gamma * k)
            for k in range(num_exit_heads + 1)
        }

        # Per-client last-round weights {client_id: List[np.ndarray]}
        self._client_states: Dict[str, List[np.ndarray]] = {}

        # Parameter depth map: {param_index: depth_k}
        # Cached from client metadata (round 1)
        self._param_depth_map: Optional[Dict[int, int]] = None

        logger.info(
            f"EDPAStrategy: γ={gamma}, λ_k={self.lambdas}"
        )

    # ── λ utilities ───────────────────────────────────────────────────────────

    def _get_lambda(self, depth_k: int) -> float:
        """Return λ_k; clamp to valid range."""
        return self.lambdas.get(depth_k, self.lambdas[self.num_exit_heads])

    def _build_fallback_depth_map(self, num_params: int) -> Dict[int, int]:
        """
        Fallback depth map when no metadata from client.
        Assigns first 70% params to backbone (k=0),
        remaining 30% split across exit heads (k=1..4).
        """
        cutoff = int(0.70 * num_params)
        depth_map: Dict[int, int] = {}
        for i in range(num_params):
            if i < cutoff:
                depth_map[i] = 0
            else:
                # Evenly distribute remaining among exit heads
                relative = i - cutoff
                per_head = max(1, (num_params - cutoff) // self.num_exit_heads)
                depth_map[i] = min(
                    1 + relative // per_head,
                    self.num_exit_heads
                )
        return depth_map

    # ── Main aggregation ──────────────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        EDPA aggregation:
        1. Compute FedAvg(w_k) for each parameter group k
        2. Mix with client's previous local weights using λ_k
        3. Store updated weights back as client state
        4. Return average of all client global weights as new global model
        """
        if not results:
            return None, {}
        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} failures")

        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        # ── Extract client weights ────────────────────────────────────────────
        client_data: List[Tuple[str, List[np.ndarray], int]] = []
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            arrays = parameters_to_ndarrays(fit_res.parameters)
            n = fit_res.num_examples
            client_data.append((client_id, arrays, n))

            # Cache depth map from first client's metrics (round 1 only)
            if self._param_depth_map is None and fit_res.metrics:
                raw = fit_res.metrics.get("param_depth_map", None)
                if raw:
                    try:
                        parsed = json.loads(raw) if isinstance(raw, str) else raw
                        # Keys are strings from JSON — convert to int
                        self._param_depth_map = {int(k): v for k, v in parsed.items()}
                        logger.info(
                            f"Cached param_depth_map from client {client_id}: "
                            f"{len(self._param_depth_map)} params"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to parse param_depth_map: {e}")

        num_params = len(client_data[0][1])

        # Build fallback depth map if still missing
        if self._param_depth_map is None:
            self._param_depth_map = self._build_fallback_depth_map(num_params)
            logger.info(
                f"Using fallback depth map: {num_params} params, "
                f"backbone cutoff at ~70%"
            )

        # ── Step 1: FedAvg for each parameter ────────────────────────────────
        fedavg_weights: List[np.ndarray] = []
        for i in range(num_params):
            weighted_sum = sum(
                (n / total_examples) * arrays[i]
                for _, arrays, n in client_data
            )
            fedavg_weights.append(weighted_sum)

        # ── Step 2: Per-client EDPA mix ───────────────────────────────────────
        new_client_states: Dict[str, List[np.ndarray]] = {}
        all_client_new_weights: List[List[np.ndarray]] = []

        for client_id, current_arrays, _ in client_data:
            prev_arrays = self._client_states.get(client_id, current_arrays)
            mixed: List[np.ndarray] = []

            for i, (fedavg_w, local_prev_w) in enumerate(
                zip(fedavg_weights, prev_arrays)
            ):
                depth_k = self._param_depth_map.get(i, 0)
                lam = self._get_lambda(depth_k)
                blended = lam * fedavg_w + (1.0 - lam) * local_prev_w
                mixed.append(blended)

            new_client_states[client_id] = mixed
            all_client_new_weights.append(mixed)

        # Update stored client states
        self._client_states.update(new_client_states)

        # ── Step 3: Global model = average of all personalized weights ────────
        global_weights: List[np.ndarray] = [
            np.mean([cw[i] for cw in all_client_new_weights], axis=0)
            for i in range(num_params)
        ]

        # Log λ values per depth
        lam_str = " | ".join(
            f"k={k} λ={lam:.2f}" for k, lam in sorted(self.lambdas.items())
        )
        logger.info(f"[EDPA] Round {server_round}: {lam_str}")

        metrics = {
            "strategy": "edpa",
            "num_clients": len(results),
            "edpa_gamma": self.gamma,
        }
        return ndarrays_to_parameters(global_weights), metrics
