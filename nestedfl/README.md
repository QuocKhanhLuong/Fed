# FedEEP — Federated Early-Exit with Progressive Phases

Flower 1.18+ implementation of depth-aware personalized aggregation for early-exit networks.

## Architecture

- **Backbone:** ConvNeXt-Tiny (pretrained, 27.8M params)
- **Exits:** 4 exit heads (GAP → LayerNorm → Linear)
- **Aggregation:** EDPA — λ_k = 1/(1+γk), deeper exits get more personalization
- **Local Training:** Fast/Slow split + CMS regularization + Self-Distillation chain

## Quick Start

```bash
# Activate environment
conda activate fl-quic

# Install
cd nestedfl
pip install -e .

# Run full FedEEP (100 rounds, EDPA, progressive phases)
flwr run

# Run with specific strategy (ablation)
flwr run --run-config "strategy=fedavg num-server-rounds=50"
flwr run --run-config "strategy=fedprox num-server-rounds=50"
flwr run --run-config "strategy=edpa num-server-rounds=100"
```

## Progressive Phases

| Phase | Rounds | Components Active |
|-------|--------|-------------------|
| 0     | 1-20   | Backbone warmup (Exit4 only) |
| 1     | 21-40  | Multi-exit CE (all 4 exits) |
| 2     | 41-60  | + Fast/Slow split + CMS regularization |
| 3     | 61-80  | + Self-Distillation KD chain |
| 4     | 81-100 | Full FedEEP system |

## Configuration

Edit `pyproject.toml`:

```toml
strategy = "edpa"           # "fedavg" | "fedprox" | "edpa"
edpa-gamma = 0.5            # EDPA personalization curve
phase-1-round = 20          # Multi-exit activation round
cms-weight = 0.1            # CMS regularization strength
kd-weight = 0.3             # Self-distillation weight
kd-temp = 4.0               # KD temperature
fast-lr-mult = 3.0          # Exit heads LR multiplier
slow-update-freq = 5        # Backbone update every K steps
```

## Project Structure

```
nestedfl/
├── models/
│   └── convnext_early_exit.py      # ConvNeXt-Tiny + 4 ExitHeads
├── nestedfl/
│   ├── client_app.py               # Flower ClientApp (phase-aware)
│   ├── server_app.py               # Flower ServerApp (phase controller)
│   ├── nested_trainer.py           # Local trainer + CMS + KD
│   ├── task.py                     # Model & data utilities
│   ├── checkpoint_manager.py       # Best/periodic model saving
│   ├── logging_config.py           # Experiment logging
│   ├── strategies/
│   │   ├── fedavg_strategy.py      # FedAvg baseline
│   │   ├── fedprox_strategy.py     # FedProx baseline
│   │   └── edpa_strategy.py        # EDPA (proposed)
│   └── data/
│       ├── base.py                 # Abstract FederatedDataset
│       └── cifar100.py             # CIFAR-100 Dirichlet α=0.5
└── pyproject.toml                  # Configuration
```

## Key Design Decisions

- **CMS is NOT in the model** — it's an external loss term in the trainer, never sent to server (zero communication overhead)
- **EDPA λ_k** — backbone (k=0) gets λ=1.0 (full global), exit4 (k=4) gets λ=0.33 (most personalized)
- **Trainer cached across rounds** — CMS memory buffers persist, preventing catastrophic forgetting
- **Phase is server-controlled** — prevents de-sync when clients drop/rejoin

## Ablation Table (Paper)

| Row | Config | Description |
|-----|--------|-------------|
| 1   | `strategy=fedavg` | FedAvg (baseline) |
| 2   | `strategy=fedprox` | FedProx (stronger baseline) |
| 3   | `strategy=edpa` | FedEEP-EDPA (proposed) |
| 4   | `strategy=edpa cms-weight=0` | FedEEP-noCMS |
| 5   | `strategy=edpa kd-weight=0` | FedEEP-noKD |
