"""
FedEEP Main — single entrypoint for all experiments.

Usage:
    python -m src.fedeep_main --config configs/cifar100_fedeep.yaml
    python -m src.fedeep_main --config configs/cifar100_fedeep.yaml --strategy fedavg
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.data.cifar100 import make_federated_cifar100
from src.evaluation.evaluator import evaluate_global
from src.federated.client import Client
from src.federated.server import run_fl
from src.models.convnext_early_exit import ConvNeXtEarlyExit
from src.trainer.local_trainer import LocalTrainer
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_dataset(config: dict):
    """Load dataset and create federated partitions based on config."""
    dataset = config.get("dataset", "cifar100")

    if dataset == "cifar100":
        return make_federated_cifar100(
            num_clients=config["num_clients"],
            alpha=config.get("dirichlet_alpha", 0.5),
            batch_size=config.get("batch_size", 32),
            data_dir=config.get("data_dir", "./data"),
            seed=config.get("seed", 42),
        )
    elif dataset == "medmnist_chest":
        from src.data.medmnist_chest import make_federated_chestmnist
        return make_federated_chestmnist(
            num_clients=config["num_clients"],
            alpha=config.get("dirichlet_alpha", 0.5),
            batch_size=config.get("batch_size", 32),
            data_dir=config.get("data_dir", "./data"),
            seed=config.get("seed", 42),
        )
    elif dataset == "medmnist_organ":
        from src.data.medmnist_organ import make_federated_organa
        return make_federated_organa(
            num_clients=config["num_clients"],
            alpha=config.get("dirichlet_alpha", 0.5),
            batch_size=config.get("batch_size", 32),
            data_dir=config.get("data_dir", "./data"),
            seed=config.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def build_clients(config, train_loaders, test_loader, device):
    """Create Client instances with cached LocalTrainers."""
    num_classes = config.get("num_classes", 100)
    nested = config.get("nested", {})
    kd = config.get("kd", {})
    cms = config.get("cms", {})

    clients = []
    for i, train_loader in enumerate(train_loaders):
        model = ConvNeXtEarlyExit(
            num_classes=num_classes,
            pretrained=config.get("pretrained", True),
        )
        trainer = LocalTrainer(
            model=model,
            device=device,
            fast_lr_multiplier=nested.get("fast_lr_mult", 3.0),
            slow_update_freq=nested.get("slow_update_freq", 5),
            kd_weight=kd.get("weight", 0.3),
            kd_temperature=kd.get("temperature", 4.0),
            cms_decay_rates=cms.get("decay_rates"),
            cms_weight=cms.get("weight", 0.1),
            use_mixed_precision=config.get("use_amp", True),
        )
        client = Client(
            client_id=i,
            trainer=trainer,
            train_loader=train_loader,
            test_loader=test_loader,
            num_samples=len(train_loader.dataset),
        )
        clients.append(client)

    return clients


def main():
    parser = argparse.ArgumentParser(description="FedEEP Experiment Runner")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--edpa-gamma", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Dirichlet alpha for data partition")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    config = load_config(args.config)

    # CLI overrides
    if args.strategy is not None:
        config["strategy"] = args.strategy
    if args.num_rounds is not None:
        config["num_rounds"] = args.num_rounds
    if args.num_clients is not None:
        config["num_clients"] = args.num_clients
    if args.edpa_gamma is not None:
        config["edpa_gamma"] = args.edpa_gamma
    if args.alpha is not None:
        config["dirichlet_alpha"] = args.alpha
    if args.seed is not None:
        config["seed"] = args.seed

    seed = config.get("seed", 42)
    set_seed(seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ── Logging ───────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config.get("dataset", "cifar100")
    strategy_name = config.get("strategy", "edpa")
    run_name = f"{dataset_name}_{strategy_name}_{timestamp}"

    log_dir = Path("src/experiments/logs") / run_name
    logger = setup_logging(log_dir=str(log_dir))
    logger.info(f"Config: {config}")
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading dataset and creating federated partitions...")
    train_loaders, test_loader = make_dataset(config)
    logger.info(
        f"Created {len(train_loaders)} client partitions, "
        f"test set: {len(test_loader.dataset)} samples"
    )

    # ── Global model (for initial weights + param_depth_map) ──────────────────
    num_classes = config.get("num_classes", 100)
    global_model = ConvNeXtEarlyExit(
        num_classes=num_classes,
        pretrained=config.get("pretrained", True),
    )
    logger.info(
        f"Global model: {sum(p.numel() for p in global_model.parameters()):,} params"
    )

    # ── Clients ───────────────────────────────────────────────────────────────
    clients = build_clients(config, train_loaders, test_loader, device)
    logger.info(f"Created {len(clients)} clients")

    # ── Evaluate function ─────────────────────────────────────────────────────
    def eval_fn(global_weights, server_round):
        return evaluate_global(
            model=global_model,
            state_dict=global_weights,
            test_loader=test_loader,
            device=device,
        )

    # ── FL config ─────────────────────────────────────────────────────────────
    fl_config = {
        "num_rounds": config.get("num_rounds", 100),
        "local_epochs": config.get("local_epochs", 5),
        "lr": config.get("lr", 0.001),
        "strategy": strategy_name,
        "fraction_train": config.get("fraction_train", 1.0),
        "milestones": config.get("phases", {
            "phase_1_round": 20,
            "phase_2_round": 40,
            "phase_3_round": 60,
            "phase_4_round": 80,
        }),
        "edpa_gamma": config.get("edpa_gamma", 0.5),
    }

    # ── Run FL ────────────────────────────────────────────────────────────────
    history = run_fl(
        clients=clients,
        global_model=global_model,
        config=fl_config,
        evaluate_fn=eval_fn,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    ckpt_dir = Path("src/experiments/checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # global_model has final weights (loaded by eval_fn in last round)
    save_checkpoint(
        state_dict=global_model.state_dict(),
        config=config,
        round_num=config.get("num_rounds", 100),
        path=str(ckpt_dir / "final_model.pt"),
    )

    metrics_path = log_dir / "history.json"
    with open(metrics_path, "w") as f:
        json.dump({"config": config, "history": history}, f, indent=2, default=str)

    logger.info(f"Results saved to {log_dir}")
    logger.info(f"Checkpoint saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
