"""
FL Client abstraction -- wraps a LocalTrainer with its local data.

Each Client holds:
  - A cached LocalTrainer (so CMS buffers persist across rounds)
  - Its own train/test DataLoaders
  - Convenience methods for the FL loop in server.py
"""

import logging
from collections import OrderedDict
from typing import Dict

import torch.utils.data

from src.trainer.local_trainer import LocalTrainer

logger = logging.getLogger("fedeep.client")


class Client:
    """
    Federated learning client.

    Args:
        client_id:    Unique integer identifier.
        trainer:      LocalTrainer instance (cached, CMS persists).
        train_loader: Client's local training DataLoader.
        test_loader:  Shared test DataLoader (same for all clients).
        num_samples:  Number of local training samples (for aggregation weighting).
    """

    def __init__(
        self,
        client_id: int,
        trainer: LocalTrainer,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        num_samples: int,
    ):
        self.id = client_id
        self.trainer = trainer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_samples = num_samples

    def set_weights(self, state_dict: OrderedDict) -> None:
        """Load global model weights (also snapshots for FedProx)."""
        self.trainer.set_weights(state_dict)

    def local_train(
        self,
        epochs: int,
        phase: int,
        learning_rate: float,
        server_round: int = 1,
        num_rounds: int = 100,
    ) -> Dict[str, float]:
        """
        Run local training for the current FL round.

        Args:
            epochs:        Number of local epochs.
            phase:         Training phase (0-4).
            learning_rate: Base learning rate.
            server_round:  Current FL round (for LR scheduler).
            num_rounds:    Total FL rounds (for LR scheduler).

        Returns:
            Training metrics dict.
        """
        self.trainer.set_phase(phase)
        metrics = self.trainer.train(
            train_loader=self.train_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            server_round=server_round,
            num_rounds=num_rounds,
        )
        return metrics

    def get_weights(self) -> OrderedDict:
        """Return model state_dict (CMS excluded)."""
        return self.trainer.get_weights()

    def evaluate(self, threshold: float = 0.8) -> Dict[str, float]:
        """Evaluate on test set with early-exit inference."""
        return self.trainer.evaluate(
            test_loader=self.test_loader,
            threshold=threshold,
        )
