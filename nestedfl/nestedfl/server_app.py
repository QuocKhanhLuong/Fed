"""
FedEEP ServerApp — Phase-controlled FL orchestration

Phase schedule:
  Phase 0 (rounds  1-20): backbone warmup (only Exit4)
  Phase 1 (rounds 21-40): multi-exit CE (all 4 exits)
  Phase 2 (rounds 41-60): fast/slow split + CMS
  Phase 3 (rounds 61-80): + knowledge distillation chain
  Phase 4 (rounds 81-100): full system

Strategy (config key "strategy"):
  "fedavg"  → FedAvgStrategy  (baseline)
  "fedprox" → FedProxStrategy (stronger baseline)
  "edpa"    → EDPAStrategy    (proposed, default)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from nestedfl.strategies import get_strategy

logger = logging.getLogger(__name__)
app = ServerApp()


# ─────────────────────────────────────────────────────────────────────────────
# Phase Controller
# ─────────────────────────────────────────────────────────────────────────────

PHASE_MILESTONES = {
    "phase_1_round": 20,   # Multi-exit activation
    "phase_2_round": 40,   # Fast/Slow + CMS
    "phase_3_round": 60,   # Knowledge Distillation
    "phase_4_round": 80,   # Full system (= EDPA active on server)
}

PHASE_NAMES = {
    0: "Backbone Warmup (Exit4 only)",
    1: "Multi-Exit CE (4 exits)",
    2: "Fast/Slow + CMS",
    3: "Self-Distillation (KD chain)",
    4: "Full FedEEP",
}


def get_current_phase(server_round: int, milestones: dict) -> int:
    """Map server_round → phase index 0..4."""
    if server_round <= milestones["phase_1_round"]:
        return 0
    if server_round <= milestones["phase_2_round"]:
        return 1
    if server_round <= milestones["phase_3_round"]:
        return 2
    if server_round <= milestones["phase_4_round"]:
        return 3
    return 4


# ─────────────────────────────────────────────────────────────────────────────
# Centralized Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def get_evaluate_fn(num_classes: int, dataset: str):
    """Build centralized evaluation function for the server."""

    def evaluate_fn(server_round: int, arrays: ArrayRecord):
        import torch
        from nestedfl.data.cifar100 import CIFAR100FederatedDataset
        from models.convnext_early_exit import ConvNeXtEarlyExit

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ConvNeXtEarlyExit(num_classes=num_classes, pretrained=False)
        state_dict = arrays.to_torch_state_dict()
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        data = CIFAR100FederatedDataset(num_partitions=1, alpha=0.5)
        _, testloader = data.get_partition(0)

        correct = total = total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        exit_counts = [0] * 4

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                logits, exits = model(images, threshold=0.8)
                total_loss += criterion(logits, labels).item()
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
                for e in exits.cpu().tolist():
                    exit_counts[e] += 1

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(len(testloader), 1)

        logger.info(
            f"Server eval round {server_round}: "
            f"loss={avg_loss:.4f} acc={accuracy:.4f} exits={exit_counts}"
        )
        return {
            "loss": float(avg_loss),
            "accuracy": float(accuracy),
            "exit_distribution": str(exit_counts),
        }

    return evaluate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config

    # ── Config ────────────────────────────────────────────────────────────────
    num_rounds: int    = cfg.get("num-server-rounds", 100)
    dataset: str       = cfg.get("dataset", "cifar100")
    num_classes: int   = 100 if dataset == "cifar100" else 10
    lr: float          = cfg.get("lr", 0.001)
    strategy_name: str = cfg.get("strategy", "edpa")

    fraction_train: float = cfg.get("fraction-train", 0.8)
    fraction_eval: float  = cfg.get("fraction-evaluate", 0.5)
    min_fit: int           = cfg.get("min-train-clients", 2)

    # EDPA-specific
    edpa_gamma: float  = cfg.get("edpa-gamma", 0.5)
    proximal_mu: float = cfg.get("proximal-mu", 0.01)

    # Phase milestones (can override in config)
    milestones = {
        "phase_1_round": cfg.get("phase-1-round", PHASE_MILESTONES["phase_1_round"]),
        "phase_2_round": cfg.get("phase-2-round", PHASE_MILESTONES["phase_2_round"]),
        "phase_3_round": cfg.get("phase-3-round", PHASE_MILESTONES["phase_3_round"]),
        "phase_4_round": cfg.get("phase-4-round", PHASE_MILESTONES["phase_4_round"]),
    }

    print(f"\n{'='*60}")
    print(f"FedEEP — Federated Early-Exit with Progressive Phases")
    print(f"{'='*60}")
    print(f"Dataset   : {dataset} ({num_classes} classes)")
    print(f"Strategy  : {strategy_name.upper()}")
    print(f"Rounds    : {num_rounds}")
    print(f"LR        : {lr}")
    print(f"Phase milestones: {milestones}")
    print(f"{'='*60}\n")

    # ── Global model initialization ───────────────────────────────────────────
    from models.convnext_early_exit import ConvNeXtEarlyExit
    global_model = ConvNeXtEarlyExit(num_classes=num_classes, pretrained=True)
    initial_arrays = ArrayRecord(global_model.state_dict())
    print(f"Global model: {sum(p.numel() for p in global_model.parameters()):,} params")

    # ── Strategy ─────────────────────────────────────────────────────────────
    strategy_kwargs = dict(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_eval,
        min_fit_clients=min_fit,
        min_evaluate_clients=max(1, min_fit // 2),
        min_available_clients=min_fit,
    )
    if strategy_name == "edpa":
        strategy_kwargs["gamma"] = edpa_gamma
    elif strategy_name == "fedprox":
        strategy_kwargs["proximal_mu"] = proximal_mu

    strategy = get_strategy(strategy_name, **strategy_kwargs)
    print(f"Strategy initialized: {strategy_name.upper()}")

    # ── FL rounds with phase injection ────────────────────────────────────────
    history = []
    arrays = initial_arrays

    for server_round in range(1, num_rounds + 1):
        phase = get_current_phase(server_round, milestones)

        # Detect phase change
        if server_round == 1 or phase != get_current_phase(server_round - 1, milestones):
            print(f"\n>>> Round {server_round}: entering Phase {phase} — {PHASE_NAMES[phase]}")

        # Build per-round train config (sent to ALL selected clients)
        train_config = ConfigRecord({
            "lr":            lr,
            "phase":         phase,
            "proximal_mu":   proximal_mu if strategy_name == "fedprox" else 0.0,
        })

        # Run one round
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=train_config,
            num_rounds=1,
            evaluate_fn=get_evaluate_fn(num_classes, dataset),
        )

        arrays = result.arrays
        round_metrics = {
            "round": server_round,
            "phase": phase,
            **(result.metrics or {}),
        }
        history.append(round_metrics)
        logger.info(f"Round {server_round} done: {round_metrics}")

    # ── Save results ──────────────────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = results_dir / f"model_{dataset}_{strategy_name}_{timestamp}.pt"
    torch.save(result.arrays.to_torch_state_dict(), model_path)

    metrics_path = results_dir / f"metrics_{dataset}_{strategy_name}_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "config": dict(cfg),
            "history": history,
        }, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Model  : {model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"{'='*60}")
