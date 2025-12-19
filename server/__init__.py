"""Server components for FL-QUIC system"""

# Lazy imports to avoid aioquic dependency when not needed
from .aggregators import (
    FedAvgAggregator,
    FedProxAggregator, 
    FedDynAggregator,
    NestedFedDynAggregator,
    create_aggregator,
)

__all__ = [
    'FedAvgAggregator',
    'FedProxAggregator',
    'FedDynAggregator', 
    'NestedFedDynAggregator',
    'create_aggregator',
]

from .quic_server import FLQuicServer
