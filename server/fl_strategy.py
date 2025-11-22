import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import AggregatedMetrics

try:
    import flwr as fl
    from flwr.common import FitRes, EvaluateRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
    from flwr.server.client_proxy import ClientProxy
    HAS_FLOWER = True
except ImportError:
    HAS_FLOWER = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FLStrategy(fl.server.strategy.FedProx if HAS_FLOWER else object):
    def __init__(
        self,
        fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
        evaluate_fn=None, on_fit_config_fn=None, on_evaluate_config_fn=None, accept_failures=True, initial_parameters=None,
        proximal_mu=0.1,  # FedProx proximal term
    ):
        if not HAS_FLOWER: return
        super().__init__(
            fraction_fit=fraction_fit, 
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients, 
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients, 
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn, 
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures, 
            initial_parameters=initial_parameters,
            proximal_mu=proximal_mu,  # FedProx parameter
        )
        self.current_round = 0
        logger.info(f"FLStrategy (FedProx) init: min_fit={min_fit_clients}, min_eval={min_evaluate_clients}, mu={proximal_mu}")
    
    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round
        logger.info(f"Round {server_round}: Aggregating {len(results)} updates")
        
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            total_samples = sum(r.num_examples for _, r in results)
            avg_loss = np.mean([r.metrics.get('loss', 0.0) for _, r in results])
            avg_accuracy = np.mean([r.metrics.get('accuracy', 0.0) for _, r in results])
            
            metrics.update({'total_samples': total_samples, 'avg_loss': avg_loss, 'avg_accuracy': avg_accuracy})
            logger.info(f"Aggregated: loss={avg_loss:.4f}, acc={avg_accuracy:.4f}")
        
        return aggregated_parameters, metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggregated, metrics = super().aggregate_evaluate(server_round, results, failures)
        if results:
            avg_accuracy = np.mean([r.metrics.get('accuracy', 0.0) for _, r in results])
            metrics['avg_accuracy'] = avg_accuracy
            logger.info(f"Eval aggregated: loss={loss_aggregated:.4f}, acc={avg_accuracy:.4f}")
        return loss_aggregated, metrics

    def aggregate_time_window(self, server_round, results, time_window=300.0):
        return self.aggregate_fit(server_round, results, failures=[])


class ContributionAwareStrategy(FLStrategy):
    def aggregate_fit(self, server_round, results, failures):
        if not results: return None, {}
            
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
        
        total_examples = sum([num for _, num in weights_results])
        simple_avg_weights = [np.zeros_like(w) for w in weights_results[0][0]]
        for weights, num in weights_results:
            for i, layer in enumerate(weights):
                simple_avg_weights[i] += layer * (num / total_examples)
                
        scores = []
        for weights, num in weights_results:
            client_vec = np.concatenate([w.flatten() for w in weights])
            avg_vec = np.concatenate([w.flatten() for w in simple_avg_weights])
            
            norm_c, norm_a = np.linalg.norm(client_vec), np.linalg.norm(avg_vec)
            sim = np.dot(client_vec, avg_vec) / (norm_c * norm_a) if norm_c > 0 and norm_a > 0 else 0
            scores.append(max(0, sim) * np.log1p(num))
            
        total_score = sum(scores)
        normalized_weights = [s / total_score for s in scores] if total_score > 0 else [1.0 / len(scores)] * len(scores)
        
        aggregated_weights = [np.zeros_like(w) for w in weights_results[0][0]]
        for idx, (weights, _) in enumerate(weights_results):
            score = normalized_weights[idx]
            for i, layer in enumerate(weights):
                aggregated_weights[i] += layer * score
                
        return ndarrays_to_parameters(aggregated_weights), {}


def create_fit_config(server_round: int, local_epochs: int = 3) -> Dict[str, Scalar]:
    config = {'round': server_round, 'local_epochs': local_epochs, 'learning_rate': 1e-3, 'fedprox_mu': 0.01}
    if server_round > 5: config['learning_rate'] = 5e-4
    if server_round > 10: config['learning_rate'] = 1e-4
    return config

def create_evaluate_config(server_round: int) -> Dict[str, Scalar]:
    return {'round': server_round}

def create_strategy(min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2, local_epochs=3) -> FLStrategy:
    if not HAS_FLOWER: raise RuntimeError("Flower not installed")
    return ContributionAwareStrategy(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=lambda r: create_fit_config(r, local_epochs),
        on_evaluate_config_fn=create_evaluate_config,
        accept_failures=True,
    )

if __name__ == "__main__":
    if not HAS_FLOWER: exit(1)
    strategy = create_strategy(min_fit_clients=2, min_evaluate_clients=2, local_epochs=3)
    logger.info("Strategy demo completed")
