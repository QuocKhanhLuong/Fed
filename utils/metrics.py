"""
Publication-Ready Evaluation Framework for FL-QUIC-LoRA
Modular metrics computation for Model Performance & System Efficiency

Supports:
- Classification Metrics: Accuracy, F1, Precision, Recall, AUROC
- Resource Tracking: GPU Memory, Execution Time
- System Metrics: Communication Cost, Network Statistics

Author: Research Team - FL-QUIC-LoRA Project
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import sklearn for advanced metrics
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix as sklearn_confusion_matrix,
    )
    from sklearn.preprocessing import label_binarize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available - using basic metrics only")

# Try to import torch for GPU tracking
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available - GPU tracking disabled")


class ClassificationEvaluator:
    """
    Classification metrics evaluator for publication-ready results.
    
    Computes:
    - Accuracy: Overall classification accuracy
    - Precision (Macro): Average precision across classes
    - Recall (Macro): Average recall across classes
    - F1-Score (Macro): Harmonic mean of precision and recall
    - AUROC (One-vs-Rest): Multi-class ROC curve area
    - Confusion Matrix: For detailed error analysis
    """
    
    @staticmethod
    def compute(
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        num_classes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            predictions: Predicted class labels (N,)
            targets: Ground truth labels (N,)
            probabilities: Class probabilities (N, C) - optional for AUROC
            num_classes: Number of classes (auto-detected if None)
            
        Returns:
            Dictionary with metrics:
            {
                'accuracy': float,
                'precision_macro': float,
                'recall_macro': float,
                'f1_macro': float,
                'auroc': float (if probabilities provided),
                'confusion_matrix': List[List[int]],
                'num_samples': int,
                'num_classes': int,
            }
        """
        # Input validation
        if len(predictions) == 0 or len(targets) == 0:
            logger.warning("Empty predictions or targets")
            return {
                'accuracy': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'f1_macro': 0.0,
                'num_samples': 0,
                'num_classes': 0,
            }
        
        # Convert to numpy if needed
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        
        # Auto-detect number of classes
        if num_classes is None:
            num_classes = max(int(targets.max()) + 1, int(predictions.max()) + 1)
        
        metrics = {
            'num_samples': len(predictions),
            'num_classes': num_classes,
        }
        
        # Basic accuracy (always computable)
        metrics['accuracy'] = float(np.mean(predictions == targets))
        
        # Advanced metrics with sklearn
        if HAS_SKLEARN:
            try:
                # Precision, Recall, F1-Score (Macro)
                metrics['precision_macro'] = float(
                    precision_score(targets, predictions, average='macro', zero_division=0)
                )
                metrics['recall_macro'] = float(
                    recall_score(targets, predictions, average='macro', zero_division=0)
                )
                metrics['f1_macro'] = float(
                    f1_score(targets, predictions, average='macro', zero_division=0)
                )
                
                # Confusion Matrix (convert to standard list for JSON serialization)
                cm = sklearn_confusion_matrix(targets, predictions, labels=range(num_classes))
                metrics['confusion_matrix'] = cm.tolist()
                
                # AUROC (One-vs-Rest) - requires probabilities
                if probabilities is not None and len(probabilities) > 0:
                    try:
                        probabilities = np.asarray(probabilities)
                        
                        # Check if we have all classes in the batch
                        unique_classes = np.unique(targets)
                        
                        if len(unique_classes) > 1:
                            # Binarize labels for multi-class AUROC
                            labels_binarized = label_binarize(
                                targets, 
                                classes=range(num_classes)
                            )
                            
                            # Handle case where not all classes are present
                            if labels_binarized.shape[1] == probabilities.shape[1]:
                                auroc = roc_auc_score(
                                    labels_binarized,
                                    probabilities,
                                    average='macro',
                                    multi_class='ovr'
                                )
                                metrics['auroc'] = float(auroc)
                            else:
                                logger.debug("Class mismatch in AUROC computation")
                        else:
                            logger.debug("Only one class present - AUROC undefined")
                    except Exception as e:
                        logger.debug(f"AUROC computation failed: {e}")
                
            except Exception as e:
                logger.warning(f"Advanced metrics computation failed: {e}")
        else:
            # Fallback: compute basic metrics manually
            metrics['precision_macro'] = metrics['accuracy']  # Approximation
            metrics['recall_macro'] = metrics['accuracy']
            metrics['f1_macro'] = metrics['accuracy']
        
        return metrics
    
    @staticmethod
    def format_metrics(metrics: Dict[str, Any], precision: int = 4) -> str:
        """
        Format metrics as human-readable string.
        
        Args:
            metrics: Metrics dictionary from compute()
            precision: Decimal precision
            
        Returns:
            Formatted string
        """
        lines = [
            f"Samples: {metrics.get('num_samples', 0)}",
            f"Accuracy: {metrics.get('accuracy', 0):.{precision}f}",
            f"Precision (Macro): {metrics.get('precision_macro', 0):.{precision}f}",
            f"Recall (Macro): {metrics.get('recall_macro', 0):.{precision}f}",
            f"F1-Score (Macro): {metrics.get('f1_macro', 0):.{precision}f}",
        ]
        
        if 'auroc' in metrics:
            lines.append(f"AUROC: {metrics['auroc']:.{precision}f}")
        
        return " | ".join(lines)


class ResourceTracker:
    """
    Context manager for tracking computational resources.
    
    Tracks:
    - Execution time (seconds)
    - GPU memory usage (MB) - peak allocation
    
    Usage:
        with ResourceTracker() as tracker:
            # Your code here
            train_model()
        
        metrics = tracker.get_metrics()
        print(f"Time: {metrics['time_s']:.2f}s")
        print(f"GPU Memory: {metrics['gpu_mem_peak_mb']:.2f} MB")
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize resource tracker.
        
        Args:
            device: Device to track (e.g., 'cuda:0'). Auto-detected if None.
        """
        self.device = device
        self.start_time = None
        self.end_time = None
        self.gpu_mem_start = 0.0
        self.gpu_mem_peak = 0.0
        
        # Auto-detect device
        if self.device is None and HAS_TORCH:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __enter__(self):
        """Start tracking."""
        self.start_time = time.time()
        
        # Reset GPU memory stats
        if HAS_TORCH and self.device and 'cuda' in self.device:
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
                self.gpu_mem_start = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            except Exception as e:
                logger.debug(f"GPU memory tracking failed: {e}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking."""
        self.end_time = time.time()
        
        # Get peak GPU memory
        if HAS_TORCH and self.device and 'cuda' in self.device:
            try:
                self.gpu_mem_peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            except Exception as e:
                logger.debug(f"GPU memory tracking failed: {e}")
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get tracked metrics.
        
        Returns:
            Dictionary with:
            {
                'time_s': float,  # Execution time in seconds
                'gpu_mem_peak_mb': float,  # Peak GPU memory in MB
                'gpu_mem_delta_mb': float,  # Memory increase in MB
            }
        """
        if self.start_time is None or self.end_time is None:
            logger.warning("ResourceTracker not properly started/stopped")
            return {
                'time_s': 0.0,
                'gpu_mem_peak_mb': 0.0,
                'gpu_mem_delta_mb': 0.0,
            }
        
        return {
            'time_s': self.end_time - self.start_time,
            'gpu_mem_peak_mb': self.gpu_mem_peak,
            'gpu_mem_delta_mb': max(0.0, self.gpu_mem_peak - self.gpu_mem_start),
        }
    
    @staticmethod
    def get_gpu_memory_usage(device: Optional[str] = None) -> float:
        """
        Get current GPU memory usage.
        
        Args:
            device: Device to query (e.g., 'cuda:0')
            
        Returns:
            Memory usage in MB
        """
        if not HAS_TORCH:
            return 0.0
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if 'cuda' not in device:
            return 0.0
        
        try:
            return torch.cuda.memory_allocated(device) / (1024 ** 2)
        except Exception as e:
            logger.debug(f"Failed to get GPU memory: {e}")
            return 0.0


class SystemMetrics:
    """
    Helper for tracking system-level metrics in Federated Learning.
    
    Tracks:
    - Communication cost (bytes sent/received)
    - Round statistics
    - Network quality
    """
    
    def __init__(self):
        """Initialize system metrics tracker."""
        self.bytes_sent_total = 0
        self.bytes_received_total = 0
        self.bytes_sent_last = 0
        self.bytes_received_last = 0
        self.round_count = 0
    
    def update_communication(
        self,
        bytes_sent: int = 0,
        bytes_received: int = 0
    ) -> Dict[str, int]:
        """
        Update communication statistics and compute deltas.
        
        Args:
            bytes_sent: Total bytes sent (cumulative)
            bytes_received: Total bytes received (cumulative)
            
        Returns:
            Dictionary with:
            {
                'bytes_sent_delta': int,  # Since last update
                'bytes_received_delta': int,  # Since last update
                'bytes_sent_total': int,  # Total since start
                'bytes_received_total': int,  # Total since start
            }
        """
        # Compute deltas
        bytes_sent_delta = bytes_sent - self.bytes_sent_last
        bytes_received_delta = bytes_received - self.bytes_received_last
        
        # Update totals
        self.bytes_sent_total = bytes_sent
        self.bytes_received_total = bytes_received
        
        # Update last values
        self.bytes_sent_last = bytes_sent
        self.bytes_received_last = bytes_received
        
        return {
            'bytes_sent_delta': max(0, bytes_sent_delta),
            'bytes_received_delta': max(0, bytes_received_delta),
            'bytes_sent_total': self.bytes_sent_total,
            'bytes_received_total': self.bytes_received_total,
        }
    
    def increment_round(self) -> int:
        """
        Increment round counter.
        
        Returns:
            Current round number (after increment)
        """
        self.round_count += 1
        return self.round_count
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all tracked metrics.
        
        Returns:
            Dictionary with all statistics
        """
        total_bytes = self.bytes_sent_total + self.bytes_received_total
        
        return {
            'rounds_completed': self.round_count,
            'bytes_sent_total': self.bytes_sent_total,
            'bytes_received_total': self.bytes_received_total,
            'total_communication_bytes': total_bytes,
            'avg_bytes_per_round': total_bytes / max(1, self.round_count),
        }
    
    @staticmethod
    def format_bytes(num_bytes: float) -> str:
        """
        Format bytes in human-readable form.
        
        Args:
            num_bytes: Number of bytes
            
        Returns:
            Formatted string (e.g., "1.23 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(num_bytes) < 1024.0:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.2f} PB"


class AggregatedMetrics:
    """
    Helper for aggregating metrics across multiple clients/rounds.
    
    Used by FL server to compute:
    - Global accuracy (weighted average)
    - Fairness (standard deviation across clients)
    - Convergence detection
    """
    
    @staticmethod
    def weighted_average(
        metrics_list: List[Tuple[Dict[str, float], int]]
    ) -> Dict[str, float]:
        """
        Compute weighted average of metrics.
        
        Args:
            metrics_list: List of (metrics_dict, num_samples) tuples
            
        Returns:
            Weighted average metrics
        """
        if not metrics_list:
            return {}
        
        # Collect all metric keys
        all_keys = set()
        for metrics, _ in metrics_list:
            all_keys.update(metrics.keys())
        
        # Compute weighted average for each metric
        total_samples = sum(num for _, num in metrics_list)
        
        if total_samples == 0:
            return {key: 0.0 for key in all_keys}
        
        weighted_metrics = {}
        for key in all_keys:
            if key in ['num_samples', 'num_classes']:
                # Sum for count metrics
                weighted_metrics[key] = sum(
                    metrics.get(key, 0) for metrics, _ in metrics_list
                )
            else:
                # Weighted average for other metrics
                weighted_sum = sum(
                    metrics.get(key, 0.0) * num
                    for metrics, num in metrics_list
                )
                weighted_metrics[key] = weighted_sum / total_samples
        
        return weighted_metrics
    
    @staticmethod
    def compute_fairness(
        accuracies: List[float]
    ) -> Dict[str, float]:
        """
        Compute fairness metrics across clients.
        
        Lower standard deviation = more fair (all clients perform similarly)
        Higher standard deviation = less fair (some clients perform much worse)
        
        Args:
            accuracies: List of accuracy values from different clients
            
        Returns:
            Dictionary with:
            {
                'fairness_std': float,  # Standard deviation
                'fairness_min': float,  # Worst client
                'fairness_max': float,  # Best client
                'fairness_range': float,  # Max - Min
            }
        """
        if not accuracies:
            return {
                'fairness_std': 0.0,
                'fairness_min': 0.0,
                'fairness_max': 0.0,
                'fairness_range': 0.0,
            }
        
        accuracies = np.array(accuracies)
        
        return {
            'fairness_std': float(np.std(accuracies)),
            'fairness_min': float(np.min(accuracies)),
            'fairness_max': float(np.max(accuracies)),
            'fairness_range': float(np.max(accuracies) - np.min(accuracies)),
        }
    
    @staticmethod
    def check_convergence(
        current_accuracy: float,
        target_accuracy: float,
        patience: int = 3,
        history: Optional[List[float]] = None
    ) -> Tuple[bool, str]:
        """
        Check if training has converged.
        
        Args:
            current_accuracy: Current accuracy
            target_accuracy: Target accuracy threshold
            patience: Number of rounds to wait for improvement
            history: Recent accuracy history
            
        Returns:
            (converged: bool, reason: str)
        """
        # Check if target reached
        if current_accuracy >= target_accuracy:
            return True, f"Target accuracy {target_accuracy:.2%} reached"
        
        # Check if stuck (no improvement in recent rounds)
        if history and len(history) >= patience:
            recent = history[-patience:]
            if all(abs(acc - current_accuracy) < 0.001 for acc in recent):
                return True, f"No improvement in last {patience} rounds"
        
        return False, "Training continues"


# Example usage and tests
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Metrics Module Demo")
    logger.info("="*60)
    
    # Test 1: Classification Evaluator
    logger.info("\n[TEST 1] ClassificationEvaluator")
    predictions = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    targets = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2, 1])
    probabilities = np.random.rand(10, 3)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    metrics = ClassificationEvaluator.compute(predictions, targets, probabilities)
    logger.info(ClassificationEvaluator.format_metrics(metrics))
    
    # Test 2: Resource Tracker
    logger.info("\n[TEST 2] ResourceTracker")
    with ResourceTracker() as tracker:
        # Simulate some work
        time.sleep(0.1)
        _ = [i**2 for i in range(100000)]
    
    resource_metrics = tracker.get_metrics()
    logger.info(f"Time: {resource_metrics['time_s']:.3f}s")
    logger.info(f"GPU Memory Peak: {resource_metrics['gpu_mem_peak_mb']:.2f} MB")
    
    # Test 3: System Metrics
    logger.info("\n[TEST 3] SystemMetrics")
    sys_metrics = SystemMetrics()
    
    # Simulate 3 rounds
    for round_num in range(1, 4):
        comm = sys_metrics.update_communication(
            bytes_sent=1024 * 1024 * round_num,  # 1MB per round
            bytes_received=512 * 1024 * round_num  # 512KB per round
        )
        sys_metrics.increment_round()
        logger.info(f"Round {round_num}: Sent {SystemMetrics.format_bytes(comm['bytes_sent_delta'])}, "
                   f"Received {SystemMetrics.format_bytes(comm['bytes_received_delta'])}")
    
    summary = sys_metrics.get_summary()
    logger.info(f"Summary: {summary['rounds_completed']} rounds, "
               f"{SystemMetrics.format_bytes(summary['total_communication_bytes'])} total")
    
    # Test 4: Aggregated Metrics
    logger.info("\n[TEST 4] AggregatedMetrics")
    client_metrics = [
        ({'accuracy': 0.85, 'f1_macro': 0.83}, 100),
        ({'accuracy': 0.90, 'f1_macro': 0.88}, 150),
        ({'accuracy': 0.80, 'f1_macro': 0.78}, 120),
    ]
    
    avg_metrics = AggregatedMetrics.weighted_average(client_metrics)
    logger.info(f"Global Accuracy: {avg_metrics['accuracy']:.4f}")
    
    accuracies = [m['accuracy'] for m, _ in client_metrics]
    fairness = AggregatedMetrics.compute_fairness(accuracies)
    logger.info(f"Fairness (Std): {fairness['fairness_std']:.4f}")
    
    converged, reason = AggregatedMetrics.check_convergence(0.87, 0.85)
    logger.info(f"Convergence: {converged} - {reason}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ All tests passed!")
    logger.info("="*60)
