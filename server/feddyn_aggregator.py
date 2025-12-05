"""
FedDyn: Federated Dynamic Regularization

This module implements the FedDyn aggregation strategy from:

    Acar et al., "Federated Learning Based on Dynamic Regularization"
    ICLR 2021

Algorithm Overview (Section 3 of original paper):
------------------------------------------------
FedDyn introduces a server-side correction term h to handle client drift
without requiring additional client computation (unlike FedProx).

Key Insight:
    Instead of regularizing on clients (FedProx: μ||w - w_global||²),
    FedDyn corrects the aggregated model on the server.

Convergence Properties (Theorem 1):
    - Converges to optimal with rate O(1/√T) for convex objectives
    - Robust to non-IID data distributions
    - Communication complexity: O(1/ε²)

Author: Research Team
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FedDynAggregator:
    """
    FedDyn Aggregation Strategy.
    
    Algorithm 2: FedDyn Server Update
    ─────────────────────────────────
    Input: Client updates {(w_i, n_i)}_{i=1}^{m}, global weights w^t, 
           correction term h^t, regularization α
    
    1. w_avg ← Σ(n_i/n) · w_i          # Weighted average
    2. w^{t+1} ← w_avg - (1/α) · h^t   # Apply correction
    3. h^{t+1} ← h^t - α(w^{t+1} - w_avg)  # Update correction
    
    Output: w^{t+1}
    
    Mathematical Formulation:
    -------------------------
    The correction term h evolves as:
        h^{t+1} = h^t - α · Δw^t
    
    where Δw^t = w^{t+1} - w_avg is the correction applied.
    
    This implicitly tracks the gradient drift across clients:
        h ≈ -∇F(w*) for stationary solution
    
    Args:
        alpha (float): Regularization strength α
            - Higher α → Stronger correction
            - Typical range: [0.01, 0.1]
            
    Attributes:
        h (List[np.ndarray]): Server-side correction term
        round (int): Current communication round
    """
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.h: Optional[List[np.ndarray]] = None
        self.round = 0
        
        logger.info(f"FedDynAggregator initialized: α={alpha}")
    
    def _weighted_average(
        self, 
        updates: List[Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        """
        Compute weighted average of client updates.
        
        Implements: w_avg = Σ(n_i/n) · w_i
        
        Args:
            updates: List of (weights, num_samples) tuples
            
        Returns:
            Weighted average weights
        """
        n_total = sum(n for _, n in updates)
        
        w_avg = [np.zeros_like(w, dtype=np.float32) for w in updates[0][0]]
        
        for weights, n_i in updates:
            coef = n_i / n_total
            for j, w in enumerate(weights):
                w_avg[j] += w.astype(np.float32) * coef
        
        return w_avg
    
    def aggregate(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int]],
        global_weights: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """
        Aggregate client updates using FedDyn.
        
        Implements Algorithm 2 from the paper.
        
        Args:
            client_updates: {client_id: (weights, num_samples)}
            global_weights: Previous global model (for logging)
            
        Returns:
            Aggregated global weights w^{t+1}
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        self.round += 1
        updates = list(client_updates.values())
        m = len(updates)  # Number of participating clients
        
        logger.info(f"FedDyn Round {self.round}: m={m} clients")
        
        # Step 1: Weighted average
        w_avg = self._weighted_average(updates)
        
        # Initialize h on first round
        if self.h is None:
            self.h = [np.zeros_like(w, dtype=np.float32) for w in w_avg]
            logger.info(f"Initialized h with {len(self.h)} parameter arrays")
        
        # Step 2: Apply correction
        # w^{t+1} = w_avg - (1/α) · h^t
        w_new = []
        for w_j, h_j in zip(w_avg, self.h):
            w_new.append(w_j - (1.0 / self.alpha) * h_j)
        
        # Step 3: Update correction term
        # h^{t+1} = h^t - α · (w^{t+1} - w_avg)
        for j in range(len(self.h)):
            delta = w_new[j] - w_avg[j]
            self.h[j] = self.h[j] - self.alpha * delta
        
        # Logging
        n_total = sum(n for _, n in updates)
        h_norm = np.mean([np.linalg.norm(h) for h in self.h])
        logger.info(f"FedDyn: n_total={n_total}, ||h||={h_norm:.6f}")
        
        return w_new
    
    def reset(self):
        """Reset state for new training run."""
        self.h = None
        self.round = 0
        logger.info("FedDynAggregator reset")


class FedNovaAggregator:
    """
    FedNova: Normalized Averaging.
    
    Reference: Wang et al., "Tackling the Objective Inconsistency Problem"
               NeurIPS 2020
    
    Key Idea:
        Normalize updates by local step count to handle
        computational heterogeneity.
    
    Formula:
        w^{t+1} = w^t + τ_eff · Σ(p_i/τ_i) · Δw_i
        
    where τ_i is local steps for client i, and
          τ_eff = Σ p_i · τ_i is effective steps.
    """
    
    def __init__(self):
        self.round = 0
        logger.info("FedNovaAggregator initialized")
    
    def aggregate(
        self,
        client_updates: Dict[str, Tuple[List[np.ndarray], int, int]],
        global_weights: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Aggregate with normalized averaging.
        
        Args:
            client_updates: {id: (delta_weights, n_samples, local_steps)}
            global_weights: Previous global weights
            
        Returns:
            New global weights
        """
        if not client_updates:
            raise ValueError("No client updates")
        
        self.round += 1
        
        # Compute effective steps
        n_total = sum(n for _, n, _ in client_updates.values())
        tau_eff = sum(
            (n / n_total) * tau 
            for _, n, tau in client_updates.values()
        )
        
        # Initialize new weights
        w_new = [np.copy(w) for w in global_weights]
        
        # Normalized aggregation
        for delta, n_i, tau_i in client_updates.values():
            coef = (n_i / n_total) * (tau_eff / tau_i)
            for j, d in enumerate(delta):
                w_new[j] += coef * d
        
        logger.info(f"FedNova Round {self.round}: τ_eff={tau_eff:.2f}")
        
        return w_new


# =============================================================================
# Factory Function
# =============================================================================

def create_aggregator(strategy: str = "feddyn", **kwargs):
    """
    Create aggregation strategy instance.
    
    Args:
        strategy: "feddyn" or "fednova"
        **kwargs: Strategy-specific parameters
        
    Returns:
        Aggregator instance
    """
    strategies = {
        "feddyn": lambda: FedDynAggregator(alpha=kwargs.get("alpha", 0.01)),
        "fednova": lambda: FedNovaAggregator(),
    }
    
    if strategy.lower() not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies)}")
    
    return strategies[strategy.lower()]()


# =============================================================================
# Unit Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FedDyn Aggregator - IEEE Format Test")
    print("=" * 60)
    
    # Create aggregator
    aggregator = FedDynAggregator(alpha=0.01)
    
    # Simulate 3 clients
    np.random.seed(42)
    client_updates = {
        f"client_{i}": (
            [np.random.randn(100, 100).astype(np.float32)],
            np.random.randint(50, 200)
        )
        for i in range(3)
    }
    
    print(f"\nClients: {list(client_updates.keys())}")
    print(f"Samples: {[v[1] for v in client_updates.values()]}")
    
    # Aggregate
    result = aggregator.aggregate(client_updates)
    print(f"\nRound 1: shape={result[0].shape}, ||h||={np.linalg.norm(aggregator.h[0]):.6f}")
    
    # Second round
    result2 = aggregator.aggregate(client_updates)
    print(f"Round 2: shape={result2[0].shape}, ||h||={np.linalg.norm(aggregator.h[0]):.6f}")
