"""
FedProx Strategy — Federated Learning with Proximal Term

Server-side: aggregation is same as FedAvg (sample-weighted average).
Client-side: includes a proximal regularization term that penalizes deviation
from the global model, reducing client drift in non-IID settings.

The proximal term is communicated to clients via train_config:
    config["proximal_mu"] = μ

The client trainer adds:  L_total += (μ/2) * ||θ_local - θ_global||^2

Used as Row 2 in ablation table: "FedProx (stronger baseline)".

Reference: Li et al., "Federated Optimization in Heterogeneous Networks",
           MLSys 2020.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from .fedavg_strategy import FedAvgStrategy

logger = logging.getLogger(__name__)


class FedProxStrategy(FedAvgStrategy):
    """
    FedProx: FedAvg aggregation + proximal term signal to clients.

    Server-side aggregation is identical to FedAvg.
    The difference is that clients receive proximal_mu in their config
    and add the proximal loss term during local training.

    Args:
        proximal_mu: Regularization strength μ (default: 0.01)
        **kwargs: Forwarded to FedAvgStrategy
    """

    def __init__(self, proximal_mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu
        logger.info(f"FedProxStrategy: μ={proximal_mu}")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """FedProx aggregation = FedAvg aggregation (proximal term is client-side)."""
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if metrics:
            metrics["strategy"] = "fedprox"
            metrics["proximal_mu"] = self.proximal_mu
        logger.info(f"[FedProx] Round {server_round}: μ={self.proximal_mu}")
        return aggregated, metrics
