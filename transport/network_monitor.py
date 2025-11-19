"""
Network Monitor Module
Tracks network statistics (RTT, Packet Loss) and calculates a quality score.
"""

import time
import logging
from typing import Dict, Optional
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class NetworkMonitor:
    """
    Monitors network conditions and provides a quality score (0.0 - 1.0).
    Designed to work with QUIC connection statistics.
    """
    
    def __init__(self, window_size: int = 20):
        """
        Initialize NetworkMonitor.
        
        Args:
            window_size: Number of samples to keep for moving average
        """
        self.rtt_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.window_size = window_size
        
        # Thresholds for scoring
        self.min_rtt = 0.020  # 20ms (Excellent)
        self.max_rtt = 0.500  # 500ms (Poor)
        self.max_loss = 0.10  # 10% packet loss (Poor)
        
        self.current_score = 1.0
        self.last_update = time.time()
        
    def update_stats(self, rtt: float, packet_loss: float = 0.0):
        """
        Update network statistics.
        
        Args:
            rtt: Round Trip Time in seconds
            packet_loss: Packet loss rate (0.0 - 1.0)
        """
        self.rtt_history.append(rtt)
        self.loss_history.append(packet_loss)
        self.last_update = time.time()
        self._recalculate_score()
        
    def _recalculate_score(self):
        """Calculate network score based on recent history."""
        if not self.rtt_history:
            self.current_score = 1.0
            return

        # Calculate averages
        avg_rtt = np.mean(self.rtt_history)
        avg_loss = np.mean(self.loss_history) if self.loss_history else 0.0
        
        # Score based on RTT (Linear interpolation)
        # 0.02s -> 1.0, 0.5s -> 0.0
        rtt_score = 1.0 - ((avg_rtt - self.min_rtt) / (self.max_rtt - self.min_rtt))
        rtt_score = max(0.0, min(1.0, rtt_score))
        
        # Score based on Packet Loss
        # 0% -> 1.0, 10% -> 0.0
        loss_score = 1.0 - (avg_loss / self.max_loss)
        loss_score = max(0.0, min(1.0, loss_score))
        
        # Weighted combination (RTT is usually more critical for FL synchronization)
        self.current_score = 0.7 * rtt_score + 0.3 * loss_score
        
        logger.debug(f"Network Score Updated: {self.current_score:.4f} "
                     f"(RTT: {avg_rtt*1000:.1f}ms, Loss: {avg_loss*100:.1f}%)")

    def get_network_score(self) -> float:
        """
        Get current network quality score.
        
        Returns:
            Float between 0.0 (Unusable) and 1.0 (Perfect)
        """
        # Decay score if no updates for a while (stale connection)
        time_since_update = time.time() - self.last_update
        if time_since_update > 30.0:  # 30 seconds without update
            decay_factor = max(0.5, 1.0 - (time_since_update - 30.0) * 0.01)
            return self.current_score * decay_factor
            
        return self.current_score

    def get_stats(self) -> Dict[str, float]:
        """Get current raw statistics."""
        return {
            'rtt': self.rtt_history[-1] if self.rtt_history else 0.0,
            'packet_loss': self.loss_history[-1] if self.loss_history else 0.0,
            'score': self.current_score
        }
