# Nested Early-Exit Federated Learning

Flower 1.18+ implementation of Nested Learning for Federated Edge Devices.

## Quick Start

```bash
# Install dependencies
cd nestedfl
pip install -e .

# Run simulation (3 clients on RTX 4070)
flwr run

# Or with custom settings
flwr run --run-config "num-server-rounds=20 dataset=cifar10"
```

## Configuration

Edit `pyproject.toml` to customize:

```toml
[tool.flwr.app.config]
num-server-rounds = 10    # FL rounds
local-epochs = 3          # Local training epochs
dataset = "cifar100"      # Dataset: cifar10 or cifar100
fast-lr-mult = 3.0        # Nested: Fast learning rate multiplier
slow-update-freq = 5      # Nested: K value (slow update frequency)
```

## Structure

```
nestedfl/
├── nestedfl/
│   ├── client_app.py   # @app.train(), @app.evaluate()
│   ├── server_app.py   # @app.main() with FedAvg
│   └── task.py         # Model, data loading, training
├── pyproject.toml      # Configuration
└── README.md
```

## Features

- **Early-Exit Networks**: MobileViTv2 with 3 exit points
- **Nested Learning**: Fast/Slow parameter separation
- **Non-IID Data**: Dirichlet partitioning (α=0.5)
- **GPU Optimized**: 0.33 GPU per client (3 clients on 1 GPU)

## Integration with Main Project

This module imports from the parent project:
- `models/early_exit_mobilevit.py` - TimmPretrainedEarlyExit
- `client/nested_trainer.py` - NestedEarlyExitTrainer
