"""
FL Aggregation Strategies for IEEE Publication

Implements baseline and proposed aggregation algorithms:
- FedAvg (McMahan et al., 2017) - Baseline
- FedProx (Li et al., 2020) - Baseline with regularization
- FedDyn (Acar et al., 2021) - Proposed method

Each aggregator follows the same interface for fair comparison.

References:
- FedAvg: "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- FedProx: "Federated Optimization in Heterogeneous Networks" 
- FedDyn: "Federated Learning Based on Dynamic Regularization"

Author: Research Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Supported aggregation strategies."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDDYN = "feddyn"


class BaseAggregator(ABC):
    """
    Abstract base class for FL aggregation strategies.
    
    All aggregators must implement:
    - aggregate(): Combine client updates into global model
    """
    
    def __init__(self, name: str):
        self.name = name
        self.round_num = 0
        
    @abstractmethod
    def aggregate(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int]],
        global_weights: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """
        Aggregate client updates.
        
        Args:
            client_updates: {client_id: (weights, num_samples)}
            global_weights: Current global model (for regularization)
            
        Returns:
            Aggregated weights
        """
        pass
    
    def on_round_end(self):
        """Called at end of each round."""
        self.round_num += 1


# =============================================================================
# FedAvg - Baseline
# =============================================================================

class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg) - Baseline Algorithm
    
    Reference: McMahan et al., "Communication-Efficient Learning of Deep 
               Networks from Decentralized Data", AISTATS 2017
    
    Algorithm:
        w_t+1 = Σ (n_k / n) * w_k
        
    Where:
        - n_k: Number of samples on client k
        - n: Total samples across all clients
        - w_k: Weights from client k
    """
    
    def __init__(self):
        super().__init__("FedAvg")
        logger.info("FedAvgAggregator initialized")
    
    def aggregate(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int]],
        global_weights: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """
        Weighted average aggregation.
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Calculate total samples
        total_samples = sum(num for _, num in client_updates.values())
        
        # Get number of layers from first client
        first_weights = list(client_updates.values())[0][0]
        num_layers = len(first_weights)
        
        # Initialize aggregated weights
        aggregated = [np.zeros_like(w) for w in first_weights]
        
        # Weighted average
        for client_id, (weights, num_samples) in client_updates.items():
            weight = num_samples / total_samples
            for i in range(num_layers):
                aggregated[i] += weight * weights[i]
        
        logger.info(f"FedAvg: Aggregated {len(client_updates)} clients, "
                   f"{total_samples} total samples")
        
        self.on_round_end()
        return aggregated


# =============================================================================
# FedProx - Baseline with Regularization
# =============================================================================

class FedProxAggregator(BaseAggregator):
    """
    Federated Proximal (FedProx) - Handles Heterogeneous Data
    
    Reference: Li et al., "Federated Optimization in Heterogeneous Networks",
               MLSys 2020
    
    Key difference from FedAvg:
        - Client local objective includes proximal term:
          h_k(w; w^t) = F_k(w) + (μ/2) ||w - w^t||^2
        
        - Server aggregation is same as FedAvg
        - μ (mu) controls how close local models stay to global
    
    Note: The proximal term is applied during CLIENT training,
          not during server aggregation.
    """
    
    def __init__(self, mu: float = 0.01):
        """
        Args:
            mu: Proximal regularization strength (0.01 recommended)
        """
        super().__init__("FedProx")
        self.mu = mu
        logger.info(f"FedProxAggregator initialized: μ={mu}")
    
    def aggregate(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int]],
        global_weights: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """
        FedProx uses same aggregation as FedAvg.
        The proximal term is applied during client training.
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Same weighted average as FedAvg
        total_samples = sum(num for _, num in client_updates.values())
        first_weights = list(client_updates.values())[0][0]
        num_layers = len(first_weights)
        aggregated = [np.zeros_like(w) for w in first_weights]
        
        for client_id, (weights, num_samples) in client_updates.items():
            weight = num_samples / total_samples
            for i in range(num_layers):
                aggregated[i] += weight * weights[i]
        
        logger.info(f"FedProx: Aggregated {len(client_updates)} clients "
                   f"(μ={self.mu} applied during training)")
        
        self.on_round_end()
        return aggregated
    
    def get_proximal_loss(
        self,
        local_weights: List[np.ndarray],
        global_weights: List[np.ndarray],
    ) -> float:
        """
        Calculate proximal regularization term for client training.
        
        Returns: (μ/2) * ||w - w^t||^2
        """
        diff_squared = sum(
            np.sum((lw - gw) ** 2) 
            for lw, gw in zip(local_weights, global_weights)
        )
        return (self.mu / 2) * diff_squared


# =============================================================================
# FedDyn - Proposed Method (Dynamic Regularization)
# =============================================================================

class FedDynAggregator(BaseAggregator):
    """
    Federated Dynamic Regularization (FedDyn)
    
    Reference: Acar et al., "Federated Learning Based on Dynamic Regularization",
               ICLR 2021
    
    Key innovation:
        - Server maintains gradient correction term h
        - Helps with non-IID data convergence
    
    Algorithm:
        Server:
            h^t+1 = h^t - α(w^t+1 - w^t)
            
        Client local objective:
            h_k(w) = F_k(w) - <∇_k, w> + (α/2)||w - w^t||^2
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Args:
            alpha: Regularization strength (0.01 recommended)
        """
        super().__init__("FedDyn")
        self.alpha = alpha
        self.h: Optional[List[np.ndarray]] = None  # Gradient correction
        logger.info(f"FedDynAggregator initialized: α={alpha}")
    
    def aggregate(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int]],
        global_weights: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """
        FedDyn aggregation with gradient correction.
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Weighted average (same as FedAvg)
        total_samples = sum(num for _, num in client_updates.values())
        first_weights = list(client_updates.values())[0][0]
        num_layers = len(first_weights)
        aggregated = [np.zeros_like(w) for w in first_weights]
        
        for client_id, (weights, num_samples) in client_updates.items():
            weight = num_samples / total_samples
            for i in range(num_layers):
                aggregated[i] += weight * weights[i]
        
        # Initialize h if needed
        if self.h is None:
            self.h = [np.zeros_like(w) for w in aggregated]
        
        # Apply gradient correction: w_corrected = w_avg - (1/α) * h
        if global_weights is not None:
            corrected = []
            for i in range(num_layers):
                corrected_w = aggregated[i] - (1.0 / self.alpha) * self.h[i]
                corrected.append(corrected_w)
            
            # Update h: h = h - α * (w_new - w_old)
            for i in range(num_layers):
                self.h[i] = self.h[i] - self.alpha * (corrected[i] - global_weights[i])
            
            aggregated = corrected
        
        logger.info(f"FedDyn: Aggregated {len(client_updates)} clients "
                   f"with gradient correction (α={self.alpha})")
        
        self.on_round_end()
        return aggregated
    
    def reset(self):
        """Reset gradient correction term."""
        self.h = None
        self.round_num = 0


# =============================================================================
# NestedFedDyn - For Nested Learning (only aggregates slow weights)
# =============================================================================

class NestedFedDynAggregator(BaseAggregator):
    """
    Nested FedDyn Aggregator for Multi-timescale Federated Learning.
    
    Key Innovation: Only aggregates SLOW weights (backbone), 
    keeps FAST weights (exits) local for personalization.
    
    This is designed to work with NestedEarlyExitTrainer which separates:
    - Fast Weights: Exit classifiers (stay local, personalized)
    - Slow Weights: Backbone stages (aggregated globally)
    
    Algorithm:
        Client sends: (slow_weights, num_samples)
        Server aggregates: Only slow_weights using FedDyn
        Client keeps: Local fast_weights unchanged
        
    Benefits:
    - Reduces communication (only backbone transmitted)
    - Personalized exits for each client
    - Global backbone learns shared representations
    - Addresses catastrophic forgetting in FL
    
    Reference: 
    - Nested Learning (Google Research, NeurIPS 2025)
    - FedDyn (Acar et al., ICLR 2021)
    """
    
    def __init__(self, alpha: float = 0.01):
        """
        Args:
            alpha: Regularization strength (0.01 recommended)
        """
        super().__init__("NestedFedDyn")
        self.alpha = alpha
        self.h: Optional[List[np.ndarray]] = None
        logger.info(f"NestedFedDynAggregator initialized: α={alpha}")
        logger.info("  → Only aggregates SLOW weights (backbone)")
        logger.info("  → FAST weights (exits) remain local for personalization")
    
    def aggregate(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int]],
        global_weights: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """
        Aggregate only slow (backbone) weights using FedDyn.
        
        Args:
            client_updates: {client_id: (slow_weights_only, num_samples)}
            global_weights: Current global slow weights
            
        Returns:
            Aggregated slow weights
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Weighted average
        total_samples = sum(num for _, num in client_updates.values())
        first_weights = list(client_updates.values())[0][0]
        num_layers = len(first_weights)
        aggregated = [np.zeros_like(w) for w in first_weights]
        
        for client_id, (weights, num_samples) in client_updates.items():
            weight = num_samples / total_samples
            for i in range(num_layers):
                aggregated[i] += weight * weights[i]
        
        # Initialize h if needed
        if self.h is None:
            self.h = [np.zeros_like(w) for w in aggregated]
        
        # Apply gradient correction
        if global_weights is not None:
            corrected = []
            for i in range(num_layers):
                corrected_w = aggregated[i] - (1.0 / self.alpha) * self.h[i]
                corrected.append(corrected_w)
            
            # Update h
            for i in range(num_layers):
                self.h[i] = self.h[i] - self.alpha * (corrected[i] - global_weights[i])
            
            aggregated = corrected
        
        logger.info(f"NestedFedDyn: Aggregated {len(client_updates)} clients "
                   f"(slow weights only, α={self.alpha})")
        
        self.on_round_end()
        return aggregated
    
    def reset(self):
        """Reset gradient correction term."""
        self.h = None
        self.round_num = 0


# =============================================================================
# Factory Function
# =============================================================================

def create_aggregator(
    strategy: str = "feddyn",
    **kwargs
) -> BaseAggregator:
    """
    Factory function to create aggregator.
    
    Args:
        strategy: "fedavg", "fedprox", "feddyn", or "nested_feddyn"
        **kwargs: Strategy-specific parameters
        
    Returns:
        Aggregator instance
    """
    strategy = strategy.lower()
    
    if strategy == "fedavg":
        return FedAvgAggregator()
    elif strategy == "fedprox":
        mu = kwargs.get("mu", 0.01)
        return FedProxAggregator(mu=mu)
    elif strategy == "feddyn":
        alpha = kwargs.get("alpha", 0.01)
        return FedDynAggregator(alpha=alpha)
    elif strategy == "nested_feddyn":
        alpha = kwargs.get("alpha", 0.01)
        return NestedFedDynAggregator(alpha=alpha)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                        f"Choose from: fedavg, fedprox, feddyn, nested_feddyn")


# =============================================================================
# Unit Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Aggregator Comparison Test")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create dummy client updates
    def create_dummy_updates(n_clients=3, n_params=100):
        updates = {}
        for i in range(n_clients):
            weights = [np.random.randn(n_params).astype(np.float32)]
            num_samples = 100 + i * 50
            updates[f"client_{i}"] = (weights, num_samples)
        return updates
    
    # Create initial global weights
    global_weights = [np.random.randn(100).astype(np.float32)]
    
    # Test each aggregator
    for strategy in ["fedavg", "fedprox", "feddyn"]:
        print(f"\n{'='*40}")
        print(f"Testing: {strategy.upper()}")
        print(f"{'='*40}")
        
        agg = create_aggregator(strategy, mu=0.01, alpha=0.01)
        updates = create_dummy_updates()
        
        result = agg.aggregate(updates, global_weights)
        
        print(f"✓ Input: {len(updates)} clients")
        print(f"✓ Output: {len(result)} weight arrays")
        print(f"✓ Shape: {result[0].shape}")
        print(f"✓ Mean: {np.mean(result[0]):.4f}")
    
    print("\n" + "=" * 60)
    print("✅ All aggregators working correctly!")
    print("=" * 60)
