"""
nestedfl: Nested Early-Exit Federated Learning ServerApp

Flower 1.18+ Message API implementation for centralized FL coordination
with FedAvg/FedDyn aggregation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

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
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(device)
        
        # Load full test set (partition_id=0 just to get testloader)
        _, testloader = load_data(partition_id=0, num_partitions=1, dataset=dataset)
        
        # Evaluate
        loss, accuracy = test(model, testloader, device)
        
        print(f"[Server] Round {server_round}: loss={loss:.4f}, acc={accuracy:.4f}")
        
        return loss, {"accuracy": accuracy}
    
    return evaluate_fn


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main entry point for the ServerApp.
    
    Orchestrates federated learning with:
    - FedAvg aggregation
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
    
    print(f"\n{'='*60}")
    print(f"Nested Early-Exit Federated Learning")
    print(f"{'='*60}")
    print(f"Dataset: {dataset} ({num_classes} classes)")
    print(f"Rounds: {num_rounds}")
    print(f"Learning rate: {lr}")
    print(f"{'='*60}\n")
    
    # Load global model
    global_model = get_model(num_classes=num_classes, use_pretrained=True)
    arrays = ArrayRecord(global_model.state_dict())
    
    print(f"Global model initialized: {sum(p.numel() for p in global_model.parameters())} parameters")
    
    # Initialize FedAvg strategy with centralized evaluation
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_eval,
    )
    
    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(num_classes, dataset),
    )
    
    # Save final model
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = results_dir / f"model_{dataset}_{timestamp}.pt"
    
    print(f"\nSaving final model to {model_path}...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, model_path)
    
    # Save metrics
    metrics_path = results_dir / f"metrics_{dataset}_{timestamp}.json"
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": dataset,
            "num_rounds": num_rounds,
            "lr": lr,
        },
        "history": {
            "train_metrics": [],  # Would be populated from result
            "eval_metrics": [],
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    print(f"\n{'='*60}")
    print(f"FL Training Complete!")
    print(f"{'='*60}")
