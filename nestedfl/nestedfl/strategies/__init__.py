"""FedEEP Server Strategies Package"""

from .fedavg_strategy import FedAvgStrategy
from .fedprox_strategy import FedProxStrategy
from .edpa_strategy import EDPAStrategy

__all__ = ["FedAvgStrategy", "FedProxStrategy", "EDPAStrategy"]


def get_strategy(name: str, **kwargs):
    """
    Factory: instantiate strategy by name string.

    Args:
        name: "fedavg" | "fedprox" | "edpa"
        **kwargs: forwarded to strategy constructor

    Usage:
        strategy = get_strategy("edpa", gamma=0.5, fraction_train=0.8)
    """
    strategies = {
        "fedavg":  FedAvgStrategy,
        "fedprox": FedProxStrategy,
        "edpa":    EDPAStrategy,
    }
    key = name.lower().strip()
    if key not in strategies:
        raise ValueError(
            f"Unknown strategy '{name}'. Choose from: {list(strategies.keys())}"
        )
    return strategies[key](**kwargs)
