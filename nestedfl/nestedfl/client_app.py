"""
FedEEP ClientApp

Reads `phase` from server's train_config each round and sets it on the trainer.
Also sends `param_depth_map` in round 1's metrics so EDPA knows which
parameters belong to backbone (k=0) vs exit heads (k=1..4).
"""

import json
import logging

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from nestedfl.data.cifar100 import CIFAR100FederatedDataset

logger = logging.getLogger(__name__)
app = ClientApp()

# Cached trainer and depth map per client process (persists across rounds)
_trainer_cache = {}


def _get_trainer(partition_id: int, num_partitions: int, cfg: dict, device: torch.device):
    """
    Build (or retrieve from cache) a NestedEarlyExitTrainer for this client.
    Trainer is cached so CMS buffers persist across FL rounds.
    """
    global _trainer_cache
    cache_key = f"client_{partition_id}"

    if cache_key not in _trainer_cache:
        from models.convnext_early_exit import ConvNeXtEarlyExit
        from nestedfl.nested_trainer import NestedEarlyExitTrainer

        num_classes = 100 if cfg.get("dataset", "cifar100") == "cifar100" else 10

        model = ConvNeXtEarlyExit(num_classes=num_classes, pretrained=True)
        trainer = NestedEarlyExitTrainer(
            model=model,
            device=device,
            fast_lr_multiplier=cfg.get("fast-lr-mult", 3.0),
            slow_update_freq=cfg.get("slow-update-freq", 5),
            kd_weight=cfg.get("kd-weight", 0.3),
            kd_temperature=cfg.get("kd-temp", 4.0),
            cms_weight=cfg.get("cms-weight", 0.1),
            use_mixed_precision=True,
        )
        _trainer_cache[cache_key] = trainer
        logger.info(f"[Client {partition_id}] Trainer created and cached (CMS will persist)")

    return _trainer_cache[cache_key]


def _build_param_depth_map(model) -> dict:
    """
    Build {param_index: depth_k} map for the EDPA strategy.

    depth 0 = backbone parameters
    depth 1..4 = exit1..exit4 parameters

    This is sent to the server in round 1's metrics and cached there.
    """
    exit_param_ids = {}
    for k, head in enumerate(
        [model.exit1, model.exit2, model.exit3, model.exit4], start=1
    ):
        for p in head.parameters():
            exit_param_ids[id(p)] = k  # depth k

    depth_map = {}
    for i, (name, p) in enumerate(model.named_parameters()):
        depth_map[i] = exit_param_ids.get(id(p), 0)  # 0 = backbone

    return depth_map


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

@app.train()
def train_fn(msg: Message, context: Context):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    partition_id  = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    cfg = context.run_config

    # ── Config from server (round-specific) ───────────────────────────────────
    train_config = msg.content["config"]
    lr           = float(train_config.get("lr", cfg.get("lr", 0.001)))
    phase        = int(train_config.get("phase", 0))
    proximal_mu  = float(train_config.get("proximal_mu", 0.0))
    local_epochs = int(cfg.get("local-epochs", 5))

    # ── Data ──────────────────────────────────────────────────────────────────
    data = CIFAR100FederatedDataset(
        num_partitions=num_partitions,
        alpha=float(cfg.get("dirichlet-alpha", 0.5)),
    )
    trainloader, _ = data.get_partition(partition_id)

    # ── Trainer (cached — CMS persists) ───────────────────────────────────────
    trainer = _get_trainer(partition_id, num_partitions, dict(cfg), device)

    # Load global weights from server
    state_dict = msg.content["arrays"].to_torch_state_dict()
    trainer.load_model_state_dict(state_dict)

    # Set phase (toggling components)
    trainer.set_phase(phase)

    # ── Local training ────────────────────────────────────────────────────────
    metrics = trainer.train(
        train_loader=trainloader,
        epochs=local_epochs,
        learning_rate=lr,
    )

    # ── Build reply ───────────────────────────────────────────────────────────
    updated_state = trainer.get_model_state_dict()
    model_record = ArrayRecord(updated_state)

    reply_metrics = {
        "train_loss":    float(metrics.get("loss", 0.0)),
        "train_acc":     float(metrics.get("accuracy", 0.0)),
        "num-examples":  len(trainloader.dataset),
        "phase":         phase,
    }

    # Send depth map in round 1 so EDPA can cache it
    server_round = int(msg.metadata.get("group_id", "1"))
    if server_round == 1:
        depth_map = _build_param_depth_map(trainer.model)
        reply_metrics["param_depth_map"] = json.dumps(depth_map)

    print(
        f"[Client {partition_id}] Phase={phase} "
        f"loss={reply_metrics['train_loss']:.4f} "
        f"acc={reply_metrics['train_acc']:.4f}"
    )

    content = RecordDict({
        "arrays":  model_record,
        "metrics": MetricRecord(reply_metrics),
    })
    return Message(content=content, reply_to=msg)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

@app.evaluate()
def evaluate_fn(msg: Message, context: Context):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    partition_id  = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    cfg = context.run_config

    num_classes = 100 if cfg.get("dataset", "cifar100") == "cifar100" else 10

    from models.convnext_early_exit import ConvNeXtEarlyExit
    model = ConvNeXtEarlyExit(num_classes=num_classes, pretrained=False)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
    model.to(device)
    model.eval()

    data = CIFAR100FederatedDataset(
        num_partitions=num_partitions,
        alpha=float(cfg.get("dirichlet-alpha", 0.5)),
    )
    _, testloader = data.get_partition(partition_id)

    import torch.nn.functional as F
    correct = total = total_loss = 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images, threshold=0.8)
            total_loss += F.cross_entropy(logits, labels).item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    eval_loss = total_loss / max(total, 1)
    eval_acc  = correct / max(total, 1)

    print(f"[Client {partition_id}] Eval: loss={eval_loss:.4f} acc={eval_acc:.4f}")

    content = RecordDict({
        "metrics": MetricRecord({
            "eval_loss":    eval_loss,
            "eval_acc":     eval_acc,
            "num-examples": int(total),
        })
    })
    return Message(content=content, reply_to=msg)
