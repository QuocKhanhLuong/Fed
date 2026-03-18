"""
Federated learning metrics tracker.

Accumulates per-round metrics and writes them to CSV.
Optional TensorBoard integration.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("fedeep.metrics")


class FLMetricsTracker:
    """
    Accumulate and persist per-round FL metrics.

    Writes a CSV with columns:
        round, phase, strategy, loss, accuracy, exit_distribution, ...

    Optionally writes to TensorBoard.

    Args:
        log_dir:        Directory for CSV output.
        use_tensorboard: Whether to write TensorBoard scalars.
    """

    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict] = []

        self._csv_path = self.log_dir / "metrics.csv"
        self._csv_writer = None
        self._csv_file = None
        self._header_written = False

        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))
                logger.info(f"TensorBoard logging to {self.log_dir / 'tb'}")
            except ImportError:
                logger.warning("tensorboard not installed, skipping TB logging")

    def log_round(self, metrics: Dict) -> None:
        """
        Log one round of metrics.

        Args:
            metrics: Dict with at least 'round', 'phase'. Other keys
                     (loss, accuracy, exit_distribution, ...) are logged as-is.
        """
        self.history.append(metrics)

        # CSV
        if not self._header_written:
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(metrics.keys())
            )
            self._csv_writer.writeheader()
            self._header_written = True

        self._csv_writer.writerow(
            {k: str(v) for k, v in metrics.items()}
        )
        self._csv_file.flush()

        # TensorBoard
        if self._tb_writer is not None:
            step = metrics.get("round", len(self.history))
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self._tb_writer.add_scalar(key, val, step)

    def close(self) -> None:
        """Flush and close all writers."""
        if self._csv_file is not None:
            self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()

    def get_best_round(self, metric: str = "accuracy") -> Dict:
        """Return the round with the highest value of the given metric."""
        if not self.history:
            return {}
        return max(
            self.history,
            key=lambda r: r.get(metric, float("-inf")),
        )
