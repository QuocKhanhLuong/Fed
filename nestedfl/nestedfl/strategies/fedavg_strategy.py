"""
FedAvg Strategy — Standard Federated Averaging

Baseline: all parameter groups aggregated with equal global weight (λ=1).
No personalization, no depth-aware weighting.

Used as Row 1 in ablation table: "FedAvg (baseline)".
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


class FedAvgStrategy(FedAvg):
    """
    Standard FedAvg with sample-weighted averaging.

    Inherits directly from Flower's built-in FedAvg.
    We subclass it (rather than using FedAvg directly) so all strategies
    share the same interface and can be swapped via get_strategy().

    Aggregation:
        w_global = Σ_i (n_i / N) * w_i
        where n_i = client i's dataset size, N = total samples.
    """

    def __init__(
        self,
        fraction_train: float = 0.8,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(
            fraction_fit=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
        logger.info("FedAvgStrategy initialized")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Standard sample-weighted FedAvg aggregation."""
        if not results:
            return None, {}
        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} client failures")

        # Sample-weighted average
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        aggregated = [
            np.sum([w * arrays[i] for arrays, w in
                    [(a, n / total_examples) for a, n in weights_results]], axis=0)
            for i in range(len(weights_results[0][0]))
        ]

        logger.info(
            f"[FedAvg] Round {server_round}: aggregated {len(results)} clients "
            f"({total_examples:,} total samples)"
        )

        metrics = {"strategy": "fedavg", "num_clients": len(results)}
        return ndarrays_to_parameters(aggregated), metrics
