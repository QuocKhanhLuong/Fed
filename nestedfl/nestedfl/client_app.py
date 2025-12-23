"""
nestedfl: Nested Early-Exit Federated Learning ClientApp

Flower 1.18+ Message API implementation for local training with
Nested Learning features (Fast/Slow separation, LSS, CMS, DMGD).
"""

import torch
import logging
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from nestedfl.task import get_model, load_data, test

# Try to import NestedEarlyExitTrainer for full features
try:
    from nestedfl.nested_trainer import NestedEarlyExitTrainer
    HAS_NESTED_TRAINER = True
except ImportError as e:
    HAS_NESTED_TRAINER = False
    print(f"WARNING: NestedEarlyExitTrainer import failed: {e}")
except Exception as e:
    HAS_NESTED_TRAINER = False
    print(f"WARNING: NestedEarlyExitTrainer import error: {e}")

logger = logging.getLogger(__name__)

# Flower ClientApp
app = ClientApp()


@app.train()
def train_fn(msg: Message, context: Context):
    """
    Train the model on local data using Nested Learning.
    
    Features enabled:
    - Fast/Slow parameter separation (DMGD)
    - Local Surprise Signal (LSS)
    - Continuum Memory System (CMS)
    - Self-distillation for early exits
    """
    # Get configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Get run config with defaults
    local_epochs = context.run_config.get("local-epochs", 3)
    lr = msg.content["config"].get("lr", 0.001)
    dataset = context.run_config.get("dataset", "cifar100")
    num_classes = 100 if dataset == "cifar100" else 10
    
    # Nested Learning parameters
    fast_lr_mult = context.run_config.get("fast-lr-mult", 3.0)
    slow_update_freq = context.run_config.get("slow-update-freq", 5)
    use_distillation = context.run_config.get("use-distillation", True)
    cms_enabled = context.run_config.get("cms-enabled", True)
    use_lss = context.run_config.get("use-lss", True)
    
    # Load local data partition
    trainloader, testloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        dataset=dataset,
    )
    
    if HAS_NESTED_TRAINER:
        # Use full NestedEarlyExitTrainer with all features
        trainer = NestedEarlyExitTrainer(
            num_classes=num_classes,
            device=device,
            use_mixed_precision=True,
            use_timm_pretrained=True,
            use_self_distillation=use_distillation,
            cms_enabled=cms_enabled,
            use_lss=use_lss,
            fast_lr_multiplier=fast_lr_mult,
            slow_update_freq=slow_update_freq,
        )
        
        # Load weights from server using state_dict (proper key-based matching)
        state_dict_server = msg.content["arrays"].to_torch_state_dict()
        trainer.load_model_state_dict(state_dict_server)
        
        # Train with Nested Learning (DMGD + CMS + LSS)
        metrics = trainer.train(
            train_loader=trainloader,
            epochs=local_epochs,
            learning_rate=lr,
        )
        
        # Get updated state_dict for aggregation
        state_dict = trainer.get_model_state_dict()
        
        train_loss = float(metrics.get('loss', 0.0))
        train_acc = float(metrics.get('accuracy', 0.0))
        fast_steps = int(metrics.get('fast_steps', 0))
        slow_steps = int(metrics.get('slow_steps', 0))
        
        logger.info(f"[Client {partition_id}] Nested Train: loss={train_loss:.4f}, acc={train_acc:.4f}, "
                    f"fast={fast_steps}, slow={slow_steps}")
    else:
        # Fallback to basic training
        model = get_model(num_classes=num_classes, use_pretrained=True)
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        model.to(device)
        
        # Basic training loop
        from nestedfl.task import train as basic_train
        train_loss = basic_train(model, trainloader, local_epochs, lr, device)
        _, train_acc = test(model, testloader, device)
        state_dict = model.state_dict()
        fast_steps, slow_steps = 0, 0
        
        logger.info(f"[Client {partition_id}] Basic Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
    
    # Construct reply Message
    model_record = ArrayRecord(state_dict)
    metrics = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "num-examples": len(trainloader.dataset),
        "fast_steps": fast_steps,
        "slow_steps": slow_steps,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
    print(f"[Client {partition_id}] Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
    
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context):
    """
    Evaluate the model on local validation data.
    """
    # Get configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    dataset = context.run_config.get("dataset", "cifar100")
    num_classes = 100 if dataset == "cifar100" else 10
    
    # Load model and weights
    model = get_model(num_classes=num_classes, use_pretrained=True)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)
    
    # Load local test data
    _, testloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        dataset=dataset,
    )
    
    # Evaluate
    eval_loss, eval_acc = test(model, testloader, device)
    
    # Construct reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(testloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    print(f"[Client {partition_id}] Eval: loss={eval_loss:.4f}, acc={eval_acc:.4f}")
    
    return Message(content=content, reply_to=msg)
