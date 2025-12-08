"""
FL Evaluation Framework - IEEE Paper Metrics

Comprehensive evaluation for Federated Learning with Early-Exit Networks.
Designed for IEEE Transaction Journal publication standards.

Metrics Categories:
-------------------
1. Model Performance (Table I)
   - Accuracy, F1-Score, Precision, Recall, AUROC
   
2. Communication Efficiency (Table II)
   - Total bytes exchanged
   - Compression ratio
   - Rounds to convergence

3. Fairness (Table III)
   - Client accuracy variance (σ)
   - Min/Max accuracy gap

4. Early-Exit Statistics (Table IV)
   - Exit distribution per stage
   - Average exit depth
   - Computational savings

5. System Efficiency (Table V-VI)
   - Training time per round
   - Inference latency
   - GPU memory usage

Author: Research Team
Reference: IEEE Transactions on Mobile Computing, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for FL experiment."""
    experiment_name: str = "fl_experiment"
    num_rounds: int = 50
    num_clients: int = 3
    dataset: str = "cifar100"
    model: str = "EarlyExitMobileViTv2"
    strategy: str = "FedDyn"
    

@dataclass 
class RoundMetrics:
    """Metrics for a single FL round."""
    round_num: int
    global_accuracy: float
    global_loss: float
    client_accuracies: List[float] = field(default_factory=list)
    bytes_sent: int = 0
    bytes_received: int = 0
    round_time_s: float = 0.0
    exit_distribution: Optional[List[float]] = None


class FLEvaluator:
    """
    Federated Learning Evaluator for IEEE Publication.
    
    Usage:
        evaluator = FLEvaluator("experiment_1")
        
        for round in range(num_rounds):
            # ... training ...
            evaluator.log_round(
                round_num=round,
                global_accuracy=acc,
                client_accuracies=[c1_acc, c2_acc, c3_acc],
                bytes_sent=bytes_s,
                exit_distribution=[0.3, 0.3, 0.4]
            )
        
        # Generate publication tables
        evaluator.generate_tables()
    """
    
    def __init__(
        self, 
        experiment_name: str = "fl_experiment",
        save_dir: str = "./results",
    ):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.rounds: List[RoundMetrics] = []
        self.start_time = time.time()
        self.config: Optional[ExperimentConfig] = None
        
        logger.info(f"FLEvaluator initialized: {experiment_name}")
    
    def set_config(self, config: ExperimentConfig):
        """Set experiment configuration."""
        self.config = config
    
    def log_round(
        self,
        round_num: int,
        global_accuracy: float,
        global_loss: float = 0.0,
        client_accuracies: Optional[List[float]] = None,
        bytes_sent: int = 0,
        bytes_received: int = 0,
        round_time_s: float = 0.0,
        exit_distribution: Optional[List[float]] = None,
    ):
        """
        Log metrics for a single round.
        
        Args:
            round_num: Current round number
            global_accuracy: Global test accuracy
            global_loss: Global test loss
            client_accuracies: List of per-client accuracies
            bytes_sent: Bytes sent this round
            bytes_received: Bytes received this round
            round_time_s: Round duration in seconds
            exit_distribution: [exit1_ratio, exit2_ratio, exit3_ratio]
        """
        metrics = RoundMetrics(
            round_num=round_num,
            global_accuracy=global_accuracy,
            global_loss=global_loss,
            client_accuracies=client_accuracies or [],
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            round_time_s=round_time_s,
            exit_distribution=exit_distribution,
        )
        self.rounds.append(metrics)
        
        # Enhanced per-round logging
        comm_total = self._format_bytes(bytes_sent + bytes_received)
        log_parts = [f"Round {round_num}: acc={global_accuracy:.4f}"]
        
        if global_loss > 0:
            log_parts.append(f"loss={global_loss:.4f}")
        
        # Fairness: client accuracy variance
        if client_accuracies and len(client_accuracies) > 1:
            acc_std = np.std(client_accuracies)
            acc_min = min(client_accuracies)
            acc_max = max(client_accuracies)
            log_parts.append(f"σ={acc_std:.4f}")
            log_parts.append(f"range=[{acc_min:.2f},{acc_max:.2f}]")
        
        # Early exit distribution
        if exit_distribution:
            exit_str = "/".join([f"{e*100:.0f}%" for e in exit_distribution])
            log_parts.append(f"exits=[{exit_str}]")
        
        log_parts.append(f"comm={comm_total}")
        log_parts.append(f"time={round_time_s:.1f}s")
        
        logger.info(", ".join(log_parts))
    
    # =========================================================================
    # Table I: Model Performance
    # =========================================================================
    
    def get_model_performance(self) -> Dict[str, float]:
        """
        Get final model performance metrics.
        
        Returns:
            {
                'final_accuracy': float,
                'best_accuracy': float,
                'final_loss': float,
                'convergence_round': int,
            }
        """
        if not self.rounds:
            return {}
        
        accuracies = [r.global_accuracy for r in self.rounds]
        
        # Find convergence round (95% of final accuracy)
        final_acc = accuracies[-1]
        convergence_round = len(accuracies)
        for i, acc in enumerate(accuracies):
            if acc >= 0.95 * final_acc:
                convergence_round = i + 1
                break
        
        return {
            'final_accuracy': final_acc,
            'best_accuracy': max(accuracies),
            'final_loss': self.rounds[-1].global_loss,
            'convergence_round': convergence_round,
        }
    
    # =========================================================================
    # Table II: Communication Efficiency
    # =========================================================================
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """
        Get communication efficiency metrics.
        
        Returns:
            {
                'total_bytes_sent': int,
                'total_bytes_received': int,
                'total_communication': int,
                'avg_bytes_per_round': float,
                'communication_formatted': str,
            }
        """
        total_sent = sum(r.bytes_sent for r in self.rounds)
        total_received = sum(r.bytes_received for r in self.rounds)
        total_comm = total_sent + total_received
        
        return {
            'total_bytes_sent': total_sent,
            'total_bytes_received': total_received,
            'total_communication': total_comm,
            'avg_bytes_per_round': total_comm / max(1, len(self.rounds)),
            'communication_formatted': self._format_bytes(total_comm),
        }
    
    # =========================================================================
    # Table III: Fairness
    # =========================================================================
    
    def get_fairness_metrics(self) -> Dict[str, float]:
        """
        Get fairness metrics across clients.
        
        Returns:
            {
                'accuracy_std': float,  # Lower = more fair
                'accuracy_min': float,
                'accuracy_max': float,
                'accuracy_gap': float,  # Max - Min
            }
        """
        # Get final round client accuracies
        if not self.rounds or not self.rounds[-1].client_accuracies:
            return {
                'accuracy_std': 0.0,
                'accuracy_min': 0.0,
                'accuracy_max': 0.0,
                'accuracy_gap': 0.0,
            }
        
        accs = np.array(self.rounds[-1].client_accuracies)
        
        return {
            'accuracy_std': float(np.std(accs)),
            'accuracy_min': float(np.min(accs)),
            'accuracy_max': float(np.max(accs)),
            'accuracy_gap': float(np.max(accs) - np.min(accs)),
        }
    
    # =========================================================================
    # Table IV: Early-Exit Statistics
    # =========================================================================
    
    def get_exit_statistics(self) -> Dict[str, Any]:
        """
        Get early-exit distribution statistics.
        
        Returns:
            {
                'exit_1_ratio': float,
                'exit_2_ratio': float,
                'exit_3_ratio': float,
                'avg_exit_depth': float,  # 1-3, lower = faster
                'computation_savings': float,  # % saved vs full network
            }
        """
        # Aggregate exit distributions
        all_exits = [r.exit_distribution for r in self.rounds 
                     if r.exit_distribution is not None]
        
        if not all_exits:
            return {
                'exit_1_ratio': 0.0,
                'exit_2_ratio': 0.0,
                'exit_3_ratio': 1.0,
                'avg_exit_depth': 3.0,
                'computation_savings': 0.0,
            }
        
        # Average across rounds
        avg_dist = np.mean(all_exits, axis=0)
        
        # Compute average exit depth (1-indexed)
        avg_depth = sum((i+1) * avg_dist[i] for i in range(len(avg_dist)))
        
        # Computation savings (exit 1 = 33% compute, exit 2 = 66%, exit 3 = 100%)
        compute_ratios = [0.33, 0.66, 1.0]
        avg_compute = sum(compute_ratios[i] * avg_dist[i] for i in range(len(avg_dist)))
        savings = (1.0 - avg_compute) * 100
        
        return {
            'exit_1_ratio': float(avg_dist[0]) if len(avg_dist) > 0 else 0.0,
            'exit_2_ratio': float(avg_dist[1]) if len(avg_dist) > 1 else 0.0,
            'exit_3_ratio': float(avg_dist[2]) if len(avg_dist) > 2 else 1.0,
            'avg_exit_depth': float(avg_depth),
            'computation_savings': float(savings),
        }
    
    # =========================================================================
    # Table V-VI: System Efficiency
    # =========================================================================
    
    def get_system_efficiency(self) -> Dict[str, float]:
        """
        Get system efficiency metrics.
        
        Returns:
            {
                'total_time_s': float,
                'avg_round_time_s': float,
                'total_rounds': int,
            }
        """
        total_time = time.time() - self.start_time
        round_times = [r.round_time_s for r in self.rounds if r.round_time_s > 0]
        
        return {
            'total_time_s': total_time,
            'avg_round_time_s': np.mean(round_times) if round_times else 0.0,
            'total_rounds': len(self.rounds),
        }
    
    # =========================================================================
    # Summary & Export
    # =========================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete experiment summary."""
        return {
            'experiment': self.experiment_name,
            'config': self.config.__dict__ if self.config else {},
            'model_performance': self.get_model_performance(),
            'communication': self.get_communication_stats(),
            'fairness': self.get_fairness_metrics(),
            'early_exit': self.get_exit_statistics(),
            'system': self.get_system_efficiency(),
        }
    
    def generate_tables(self) -> str:
        """
        Generate IEEE-format tables as markdown.
        
        Returns:
            Markdown string with all tables
        """
        summary = self.get_summary()
        perf = summary['model_performance']
        comm = summary['communication']
        fair = summary['fairness']
        exits = summary['early_exit']
        sys_eff = summary['system']
        
        tables = f"""
# Experiment Results: {self.experiment_name}

## Table I: Model Performance

| Metric | Value |
|--------|-------|
| Final Accuracy | {perf.get('final_accuracy', 0):.4f} |
| Best Accuracy | {perf.get('best_accuracy', 0):.4f} |
| Convergence Round | {perf.get('convergence_round', 0)} |

## Table II: Communication Efficiency

| Metric | Value |
|--------|-------|
| Total Communication | {comm.get('communication_formatted', 'N/A')} |
| Avg per Round | {self._format_bytes(comm.get('avg_bytes_per_round', 0))} |
| Total Rounds | {sys_eff.get('total_rounds', 0)} |

## Table III: Fairness

| Metric | Value |
|--------|-------|
| Accuracy Std (σ) | {fair.get('accuracy_std', 0):.4f} |
| Min Accuracy | {fair.get('accuracy_min', 0):.4f} |
| Max Accuracy | {fair.get('accuracy_max', 0):.4f} |
| Accuracy Gap | {fair.get('accuracy_gap', 0):.4f} |

## Table IV: Early-Exit Statistics

| Exit | Ratio | Compute |
|------|-------|---------|
| Exit 1 (Early) | {exits.get('exit_1_ratio', 0):.2%} | 33% |
| Exit 2 (Mid) | {exits.get('exit_2_ratio', 0):.2%} | 66% |
| Exit 3 (Full) | {exits.get('exit_3_ratio', 0):.2%} | 100% |
| **Avg Depth** | {exits.get('avg_exit_depth', 0):.2f} | - |
| **Savings** | {exits.get('computation_savings', 0):.1f}% | - |

## Table V: System Efficiency

| Metric | Value |
|--------|-------|
| Total Time | {sys_eff.get('total_time_s', 0):.1f}s |
| Avg Round Time | {sys_eff.get('avg_round_time_s', 0):.2f}s |
"""
        return tables
    
    def save_results(self):
        """Save results to JSON and Markdown files."""
        # Save JSON
        json_path = self.save_dir / f"{self.experiment_name}.json"
        with open(json_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        
        # Save Markdown tables
        md_path = self.save_dir / f"{self.experiment_name}_tables.md"
        with open(md_path, 'w') as f:
            f.write(self.generate_tables())
        
        logger.info(f"Results saved to {self.save_dir}")
        return json_path, md_path
    
    @staticmethod
    def _format_bytes(num_bytes: float) -> str:
        """Format bytes in human-readable form."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if abs(num_bytes) < 1024.0:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.2f} TB"


# =============================================================================
# Comparison Helper
# =============================================================================

class ExperimentComparison:
    """Compare multiple FL experiments for ablation studies."""
    
    def __init__(self):
        self.experiments: Dict[str, FLEvaluator] = {}
    
    def add_experiment(self, name: str, evaluator: FLEvaluator):
        """Add experiment for comparison."""
        self.experiments[name] = evaluator
    
    def generate_comparison_table(self) -> str:
        """Generate comparison table across experiments."""
        if not self.experiments:
            return "No experiments to compare."
        
        # Header
        names = list(self.experiments.keys())
        header = "| Metric | " + " | ".join(names) + " |"
        separator = "|--------|" + "|".join(["-------"] * len(names)) + "|"
        
        rows = [header, separator]
        
        # Metrics
        metrics = [
            ('Final Accuracy', lambda e: e.get_model_performance().get('final_accuracy', 0)),
            ('Convergence', lambda e: e.get_model_performance().get('convergence_round', 0)),
            ('Communication', lambda e: e.get_communication_stats().get('communication_formatted', 'N/A')),
            ('Fairness (σ)', lambda e: e.get_fairness_metrics().get('accuracy_std', 0)),
            ('Compute Savings', lambda e: f"{e.get_exit_statistics().get('computation_savings', 0):.1f}%"),
        ]
        
        for metric_name, getter in metrics:
            values = []
            for name in names:
                val = getter(self.experiments[name])
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                else:
                    values.append(str(val))
            rows.append(f"| {metric_name} | " + " | ".join(values) + " |")
        
        return "\n".join(rows)


# =============================================================================
# Unit Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FL Evaluator - IEEE Format Test")
    print("=" * 60)
    
    # Create evaluator
    evaluator = FLEvaluator("test_experiment")
    evaluator.set_config(ExperimentConfig(
        num_rounds=10,
        num_clients=3,
        dataset="cifar100",
        strategy="FedDyn"
    ))
    
    # Simulate 10 rounds
    np.random.seed(42)
    for r in range(1, 11):
        acc = 0.5 + 0.04 * r + np.random.rand() * 0.02
        client_accs = [acc + np.random.randn() * 0.02 for _ in range(3)]
        exit_dist = [0.2 + r * 0.02, 0.3, 0.5 - r * 0.02]
        exit_dist = [max(0, min(1, x)) for x in exit_dist]
        total = sum(exit_dist)
        exit_dist = [x / total for x in exit_dist]
        
        evaluator.log_round(
            round_num=r,
            global_accuracy=acc,
            global_loss=1.0 - acc,
            client_accuracies=client_accs,
            bytes_sent=1024 * 1024,
            bytes_received=512 * 1024,
            round_time_s=5.0 + np.random.rand(),
            exit_distribution=exit_dist,
        )
    
    # Generate tables
    print(evaluator.generate_tables())
    
    # Save results
    json_path, md_path = evaluator.save_results()
    print(f"\n✓ Results saved to:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
