"""
Statistical Analysis for IEEE Publication

Provides rigorous statistical tools for FL experiments:
- Multiple runs with confidence intervals
- Paired t-tests for method comparison
- LaTeX table generation

IEEE Standard Requirements:
- Report mean ± std from 3-5 independent runs
- Include 95% confidence intervals
- Perform statistical significance tests (p < 0.05)

Author: Research Team
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    run_id: int
    strategy: str
    final_accuracy: float
    best_accuracy: float
    convergence_round: int
    total_communication_mb: float
    total_time_s: float
    fairness_std: float = 0.0
    
    # Per-round data
    accuracy_history: List[float] = field(default_factory=list)
    

@dataclass
class StatisticalSummary:
    """Statistical summary across multiple runs."""
    strategy: str
    n_runs: int
    
    # Accuracy
    accuracy_mean: float
    accuracy_std: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    
    # Communication
    comm_mean_mb: float
    comm_std_mb: float
    
    # Convergence
    convergence_mean: float
    convergence_std: float
    
    # Time
    time_mean_s: float
    time_std_s: float


class StatisticalAnalyzer:
    """
    Statistical analysis for FL experiments.
    
    Usage:
        analyzer = StatisticalAnalyzer()
        
        # Add results from multiple runs
        for run in range(5):
            result = run_experiment(strategy="feddyn")
            analyzer.add_result(result)
        
        # Get statistical summary
        summary = analyzer.get_summary("feddyn")
        
        # Compare methods
        p_value = analyzer.paired_t_test("fedavg", "feddyn", metric="accuracy")
        
        # Generate IEEE table
        print(analyzer.generate_latex_table())
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Args:
            confidence_level: Confidence level for CI (default 95%)
        """
        self.confidence_level = confidence_level
        self.results: Dict[str, List[ExperimentResult]] = {}
        
        logger.info(f"StatisticalAnalyzer initialized (CI={confidence_level:.0%})")
    
    def add_result(self, result: ExperimentResult):
        """Add experiment result."""
        if result.strategy not in self.results:
            self.results[result.strategy] = []
        self.results[result.strategy].append(result)
        
        logger.debug(f"Added result: {result.strategy} run {result.run_id}")
    
    def get_summary(self, strategy: str) -> Optional[StatisticalSummary]:
        """
        Get statistical summary for a strategy.
        
        Returns:
            StatisticalSummary with mean, std, CI
        """
        if strategy not in self.results or len(self.results[strategy]) < 2:
            logger.warning(f"Insufficient runs for {strategy}")
            return None
        
        runs = self.results[strategy]
        n = len(runs)
        
        # Extract metrics
        accuracies = [r.final_accuracy for r in runs]
        comms = [r.total_communication_mb for r in runs]
        convergences = [r.convergence_round for r in runs]
        times = [r.total_time_s for r in runs]
        
        # Calculate statistics
        acc_mean, acc_std = np.mean(accuracies), np.std(accuracies, ddof=1)
        comm_mean, comm_std = np.mean(comms), np.std(comms, ddof=1)
        conv_mean, conv_std = np.mean(convergences), np.std(convergences, ddof=1)
        time_mean, time_std = np.mean(times), np.std(times, ddof=1)
        
        # Confidence interval
        ci_low, ci_high = self._confidence_interval(accuracies)
        
        return StatisticalSummary(
            strategy=strategy,
            n_runs=n,
            accuracy_mean=acc_mean,
            accuracy_std=acc_std,
            accuracy_ci_low=ci_low,
            accuracy_ci_high=ci_high,
            comm_mean_mb=comm_mean,
            comm_std_mb=comm_std,
            convergence_mean=conv_mean,
            convergence_std=conv_std,
            time_mean_s=time_mean,
            time_std_s=time_std,
        )
    
    def _confidence_interval(self, data: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval."""
        n = len(data)
        if n < 2:
            return (data[0], data[0]) if data else (0.0, 0.0)
        
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of mean
        
        # t-distribution for small samples
        t_value = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
        margin = t_value * sem
        
        return (mean - margin, mean + margin)
    
    # =========================================================================
    # Statistical Tests
    # =========================================================================
    
    def paired_t_test(
        self,
        strategy_a: str,
        strategy_b: str,
        metric: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Perform paired t-test between two strategies.
        
        Args:
            strategy_a: First strategy (baseline)
            strategy_b: Second strategy (proposed)
            metric: "accuracy", "communication", "convergence", "time"
            
        Returns:
            {
                't_statistic': float,
                'p_value': float,
                'significant': bool,  # p < 0.05
                'effect_size': float,  # Cohen's d
            }
        """
        if strategy_a not in self.results or strategy_b not in self.results:
            raise ValueError(f"Missing results for {strategy_a} or {strategy_b}")
        
        runs_a = self.results[strategy_a]
        runs_b = self.results[strategy_b]
        
        # Must have same number of runs (paired)
        n = min(len(runs_a), len(runs_b))
        
        # Extract metric
        if metric == "accuracy":
            data_a = [r.final_accuracy for r in runs_a[:n]]
            data_b = [r.final_accuracy for r in runs_b[:n]]
        elif metric == "communication":
            data_a = [r.total_communication_mb for r in runs_a[:n]]
            data_b = [r.total_communication_mb for r in runs_b[:n]]
        elif metric == "convergence":
            data_a = [r.convergence_round for r in runs_a[:n]]
            data_b = [r.convergence_round for r in runs_b[:n]]
        elif metric == "time":
            data_a = [r.total_time_s for r in runs_a[:n]]
            data_b = [r.total_time_s for r in runs_b[:n]]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(data_a, data_b)
        
        # Cohen's d (effect size)
        diff = np.array(data_b) - np.array(data_a)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0.0
        
        result = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'effect_size': float(cohens_d),
            'interpretation': self._interpret_effect_size(cohens_d),
        }
        
        logger.info(f"T-test {strategy_a} vs {strategy_b} ({metric}): "
                   f"p={p_value:.4f}, d={cohens_d:.2f}")
        
        return result
    
    @staticmethod
    def _interpret_effect_size(d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    # =========================================================================
    # Table Generation
    # =========================================================================
    
    def generate_markdown_table(self) -> str:
        """Generate IEEE-format markdown table."""
        if not self.results:
            return "No results to display."
        
        lines = [
            "## Table: Method Comparison (mean ± std)",
            "",
            "| Method | Accuracy | Comm. (MB) | Rounds | Time (s) |",
            "|--------|----------|------------|--------|----------|",
        ]
        
        for strategy in sorted(self.results.keys()):
            summary = self.get_summary(strategy)
            if summary:
                lines.append(
                    f"| {strategy.upper()} | "
                    f"{summary.accuracy_mean:.2%}±{summary.accuracy_std:.2%} | "
                    f"{summary.comm_mean_mb:.1f}±{summary.comm_std_mb:.1f} | "
                    f"{summary.convergence_mean:.0f}±{summary.convergence_std:.0f} | "
                    f"{summary.time_mean_s:.0f}±{summary.time_std_s:.0f} |"
                )
        
        return "\n".join(lines)
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for IEEE paper."""
        if not self.results:
            return "% No results"
        
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Comparison of FL Aggregation Strategies}",
            "\\label{tab:comparison}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Method & Accuracy & Comm. (MB) & Rounds & Time (s) \\\\",
            "\\midrule",
        ]
        
        for strategy in sorted(self.results.keys()):
            summary = self.get_summary(strategy)
            if summary:
                lines.append(
                    f"{strategy.upper()} & "
                    f"${summary.accuracy_mean*100:.1f} \\pm {summary.accuracy_std*100:.1f}$ & "
                    f"${summary.comm_mean_mb:.1f} \\pm {summary.comm_std_mb:.1f}$ & "
                    f"${summary.convergence_mean:.0f} \\pm {summary.convergence_std:.0f}$ & "
                    f"${summary.time_mean_s:.0f} \\pm {summary.time_std_s:.0f}$ \\\\"
                )
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        
        return "\n".join(lines)
    
    def generate_significance_table(self, baseline: str = "fedavg") -> str:
        """Generate statistical significance table."""
        lines = [
            f"## Statistical Significance vs {baseline.upper()} (p-values)",
            "",
            "| Method | Accuracy | Comm. | Convergence |",
            "|--------|----------|-------|-------------|",
        ]
        
        for strategy in sorted(self.results.keys()):
            if strategy == baseline:
                continue
            
            try:
                acc_test = self.paired_t_test(baseline, strategy, "accuracy")
                comm_test = self.paired_t_test(baseline, strategy, "communication")
                conv_test = self.paired_t_test(baseline, strategy, "convergence")
                
                def format_p(test):
                    p = test['p_value']
                    sig = "**" if test['significant'] else ""
                    return f"{sig}{p:.4f}{sig}"
                
                lines.append(
                    f"| {strategy.upper()} | "
                    f"{format_p(acc_test)} | "
                    f"{format_p(comm_test)} | "
                    f"{format_p(conv_test)} |"
                )
            except Exception as e:
                logger.warning(f"Could not compute test for {strategy}: {e}")
        
        lines.append("")
        lines.append("*Note: ** indicates p < 0.05 (statistically significant)*")
        
        return "\n".join(lines)
    
    # =========================================================================
    # Save/Load
    # =========================================================================
    
    def save_results(self, path: str):
        """Save results to JSON."""
        data = {
            strategy: [
                {
                    'run_id': r.run_id,
                    'strategy': r.strategy,
                    'final_accuracy': r.final_accuracy,
                    'best_accuracy': r.best_accuracy,
                    'convergence_round': r.convergence_round,
                    'total_communication_mb': r.total_communication_mb,
                    'total_time_s': r.total_time_s,
                    'fairness_std': r.fairness_std,
                }
                for r in runs
            ]
            for strategy, runs in self.results.items()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {path}")
    
    def load_results(self, path: str):
        """Load results from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for strategy, runs in data.items():
            for r in runs:
                result = ExperimentResult(
                    run_id=r['run_id'],
                    strategy=r['strategy'],
                    final_accuracy=r['final_accuracy'],
                    best_accuracy=r['best_accuracy'],
                    convergence_round=r['convergence_round'],
                    total_communication_mb=r['total_communication_mb'],
                    total_time_s=r['total_time_s'],
                    fairness_std=r.get('fairness_std', 0.0),
                )
                self.add_result(result)
        
        logger.info(f"Loaded {sum(len(r) for r in self.results.values())} results")


# =============================================================================
# Unit Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Statistical Analyzer - Unit Test")
    print("=" * 60)
    
    np.random.seed(42)
    analyzer = StatisticalAnalyzer()
    
    # Simulate 5 runs for each strategy
    strategies = {
        'fedavg': {'acc_base': 0.78, 'comm_base': 120, 'conv': 50},
        'fedprox': {'acc_base': 0.80, 'comm_base': 115, 'conv': 45},
        'feddyn': {'acc_base': 0.84, 'comm_base': 32, 'conv': 35},
    }
    
    for strategy, params in strategies.items():
        for run_id in range(5):
            result = ExperimentResult(
                run_id=run_id,
                strategy=strategy,
                final_accuracy=params['acc_base'] + np.random.randn() * 0.02,
                best_accuracy=params['acc_base'] + 0.02 + np.random.randn() * 0.01,
                convergence_round=int(params['conv'] + np.random.randn() * 3),
                total_communication_mb=params['comm_base'] + np.random.randn() * 10,
                total_time_s=300 + np.random.randn() * 30,
                fairness_std=0.03 + np.random.rand() * 0.02,
            )
            analyzer.add_result(result)
    
    # Print summaries
    print("\n📊 Statistical Summaries:")
    for strategy in strategies:
        summary = analyzer.get_summary(strategy)
        if summary:
            print(f"\n{strategy.upper()}:")
            print(f"  Accuracy: {summary.accuracy_mean:.2%} ± {summary.accuracy_std:.2%}")
            print(f"  95% CI: [{summary.accuracy_ci_low:.2%}, {summary.accuracy_ci_high:.2%}]")
            print(f"  Comm: {summary.comm_mean_mb:.1f} ± {summary.comm_std_mb:.1f} MB")
    
    # T-tests
    print("\n📈 Statistical Tests (vs FedAvg):")
    for metric in ['accuracy', 'communication']:
        test = analyzer.paired_t_test('fedavg', 'feddyn', metric)
        sig = "✓ SIG" if test['significant'] else "✗ N.S."
        print(f"  {metric}: p={test['p_value']:.4f} ({sig}), d={test['effect_size']:.2f} ({test['interpretation']})")
    
    # Tables
    print("\n" + "=" * 60)
    print(analyzer.generate_markdown_table())
    print("\n" + "=" * 60)
    print(analyzer.generate_significance_table())
    
    print("\n" + "=" * 60)
    print("✅ Statistical analysis complete!")
    print("=" * 60)
