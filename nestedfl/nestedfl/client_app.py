"""
nestedfl: Nested Early-Exit Federated Learning ClientApp

Flower 1.18+ Message API implementation for local training with
Nested Learning features (Fast/Slow separation, LSS, CMS).
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from nestedfl.task import get_model, get_trainer, load_data, train, test

# Flower ClientApp
app = ClientApp()


@app.train()
def train_fn(msg: Message, context: Context):
    """
    Train the model on local data using Nested Learning.
    
    This method:
    1. Loads global model weights from message
    2. Trains locally with fast/slow parameter separation
    3. Returns updated weights and metrics
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
    
    # Load model and weights
    model = get_model(num_classes=num_classes, use_pretrained=True)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)
    
    # Load local data partition
    trainloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        dataset=dataset,
    )
    
    # Train locally
    train_loss = train(
        model=model,
        trainloader=trainloader,
        epochs=local_epochs,
        lr=lr,
        device=device,
    )
    
    # Compute training accuracy
    _, train_acc = test(model, trainloader, device)
    
    # Construct reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "num-examples": len(trainloader.dataset),
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
