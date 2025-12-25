"""
nestedfl: Nested Early-Exit Federated Learning ServerApp

Flower 1.18+ Message API implementation for centralized FL coordination
with FedAvg/FedProx aggregation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

# Setup logging (creates timestamped log file)
from nestedfl.logging_config import setup_logging, get_log_file
setup_logging()

from nestedfl.task import get_model, test, load_data

logger = logging.getLogger(__name__)

# Create ServerApp
app = ServerApp()


def get_evaluate_fn(num_classes: int, dataset: str):
    """
    Create centralized evaluation function.
    
    This evaluates the global model on the full test set after each round.
    """
    def evaluate_fn(server_round: int, arrays: ArrayRecord):
        """Evaluate global model on centralized test set."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load model with global weights
        model = get_model(num_classes=num_classes, use_pretrained=True)
        
        # Filter out thop keys if present
        state_dict = {k: v for k, v in arrays.to_torch_state_dict().items()
                      if 'total_ops' not in k and 'total_params' not in k}
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        
        # Load full test set (partition_id=0 just to get testloader)
        _, testloader = load_data(partition_id=0, num_partitions=1, dataset=dataset)
        
        # Evaluate with exit stats
        result = test(model, testloader, device, return_exit_stats=True)
        
        if len(result) == 3:
            loss, accuracy, exit_stats = result
            print(f"[Server] Round {server_round}: loss={loss:.4f}, acc={accuracy:.4f}")
            print(f"  Exit Accuracies: Exit1={exit_stats['exit1_acc']:.4f}, "
                  f"Exit2={exit_stats['exit2_acc']:.4f}, Exit3={exit_stats['exit3_acc']:.4f}")
            
            # Return metrics with exit stats
            return {
                "loss": float(loss), 
                "accuracy": float(accuracy),
                "exit1_acc": float(exit_stats['exit1_acc']),
                "exit2_acc": float(exit_stats['exit2_acc']),
                "exit3_acc": float(exit_stats['exit3_acc']),
            }
        else:
            loss, accuracy = result
            print(f"[Server] Round {server_round}: loss={loss:.4f}, acc={accuracy:.4f}")
            return {"loss": float(loss), "accuracy": float(accuracy)}
    
    return evaluate_fn


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main entry point for the ServerApp.
    
    Orchestrates federated learning with:
    - FedAvg or FedProx aggregation (configurable)
    - Centralized evaluation
    - Result saving
    """
    # Read run config
    fraction_train: float = context.run_config.get("fraction-train", 1.0)
    fraction_eval: float = context.run_config.get("fraction-evaluate", 0.5)
    num_rounds: int = context.run_config.get("num-server-rounds", 10)
    lr: float = context.run_config.get("lr", 0.001)
    dataset: str = context.run_config.get("dataset", "cifar100")
    num_classes = 100 if dataset == "cifar100" else 10
    
    # Strategy selection
    strategy_name: str = context.run_config.get("strategy", "fedprox")
    proximal_mu: float = context.run_config.get("proximal-mu", 0.1)  # FedProx regularization
    
    # Nested Learning config
    fast_lr_mult: float = context.run_config.get("fast-lr-mult", 3.0)
    slow_update_freq: int = context.run_config.get("slow-update-freq", 5)
    use_distillation: bool = context.run_config.get("use-distillation", True)
    cms_enabled: bool = context.run_config.get("cms-enabled", True)
    use_lss: bool = context.run_config.get("use-lss", True)
    
    print(f"\n{'='*60}")
    print(f"Nested Early-Exit Federated Learning")
    print(f"{'='*60}")
    print(f"Dataset: {dataset} ({num_classes} classes)")
    print(f"Strategy: {strategy_name.upper()}" + (f" (μ={proximal_mu})" if strategy_name == "fedprox" else ""))
    print(f"Rounds: {num_rounds}")
    print(f"Learning rate: {lr}")
    print(f"")
    print(f"Nested Learning Features:")
    print(f"  - Fast LR Multiplier: {fast_lr_mult}")
    print(f"  - Slow Update Freq (K): {slow_update_freq}")
    print(f"  - Self-Distillation: {use_distillation}")
    print(f"  - CMS (Memory): {cms_enabled}")
    print(f"  - LSS (Surprise): {use_lss}")
    print(f"{'='*60}\n")
    
    # Load global model
    global_model = get_model(num_classes=num_classes, use_pretrained=True)
    arrays = ArrayRecord(global_model.state_dict())
    
    print(f"Global model initialized: {sum(p.numel() for p in global_model.parameters())} parameters")
    
    # Initialize strategy based on config
    if strategy_name.lower() == "fedprox":
        strategy = FedProx(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_eval,
            proximal_mu=proximal_mu,
        )
        print(f"Using FedProx strategy with μ={proximal_mu}")
    else:
        strategy = FedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_eval,
        )
        print(f"Using FedAvg strategy")
    
    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr, "proximal_mu": proximal_mu}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(num_classes, dataset),
    )
    
    # Save final model
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = results_dir / f"model_{dataset}_{strategy_name}_{timestamp}.pt"
    
    print(f"\nSaving final model to {model_path}...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, model_path)
    
    # Save metrics
    metrics_path = results_dir / f"metrics_{dataset}_{strategy_name}_{timestamp}.json"
    
    # Extract metrics from result (train and eval)
    train_history = []
    eval_history = []
    
    # Get metrics from result if available
    if hasattr(result, 'metrics') and result.metrics:
        # Direct metrics from result
        for key, value in result.metrics.items():
            if 'train' in key.lower():
                train_history.append({key: value})
            elif 'eval' in key.lower() or 'acc' in key.lower() or 'loss' in key.lower():
                eval_history.append({key: value})
    
    # Try to get history from strategy if available  
    if hasattr(result, 'history'):
        if hasattr(result.history, 'metrics_distributed_fit'):
            for round_num, metrics in enumerate(result.history.metrics_distributed_fit, 1):
                train_history.append({"round": round_num, **metrics})
        if hasattr(result.history, 'metrics_distributed'):
            for round_num, metrics in enumerate(result.history.metrics_distributed, 1):
                eval_history.append({"round": round_num, **metrics})
    
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": dataset,
            "strategy": strategy_name,
            "proximal_mu": proximal_mu,
            "num_rounds": num_rounds,
            "lr": lr,
            "nested_learning": {
                "fast_lr_mult": fast_lr_mult,
                "slow_update_freq": slow_update_freq,
                "use_distillation": use_distillation,
                "cms_enabled": cms_enabled,
                "use_lss": use_lss,
            },
        },
        "history": {
            "train_metrics": train_history,
            "eval_metrics": eval_history,
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    print(f"\n{'='*60}")
    print(f"FL Training Complete!")
    print(f"{'='*60}")
