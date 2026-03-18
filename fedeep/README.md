# FedEEP — Federated Early-Exit with Progressive Phases

Pure PyTorch implementation of FedEEP for federated learning research.

## Architecture

- **Model**: ConvNeXt-Tiny backbone with 4 early-exit heads (GAP + LayerNorm + Linear)
- **Training**: Progressive phases (backbone warmup -> multi-exit CE -> fast/slow + CMS -> KD chain -> full system)
- **Aggregation**: FedAvg, FedProx, or EDPA (Exit-Depth-aware Personalized Aggregation)
- **Datasets**: CIFAR-100, ChestMNIST, OrganAMNIST

## Setup

```bash
# Create and activate environment
conda create -n fedeep python=3.10
conda activate fedeep

# Install dependencies
pip install -e .

# (Optional) Install visualization tools
pip install -e ".[viz]"
```

## Quick Start

```bash
# CIFAR-100 with EDPA strategy (default config)
python -m src.fedeep_main --config configs/cifar100_fedeep.yaml

# Use FedAvg baseline instead
python -m src.fedeep_main --config configs/cifar100_fedeep.yaml --strategy fedavg

# MedMNIST experiments
python -m src.fedeep_main --config configs/medmnist_chest_fedeep.yaml
python -m src.fedeep_main --config configs/medmnist_organ_fedeep.yaml

# Override settings via CLI
python -m src.fedeep_main --config configs/cifar100_fedeep.yaml \
    --num-rounds 50 --num-clients 20 --seed 123
```

## Shell Scripts

```bash
# Run experiments
bash scripts/run_cifar_fedeep.sh
bash scripts/run_medmnist_fedeep.sh medmnist_chest
bash scripts/run_medmnist_fedeep.sh medmnist_organ

# EDPA gamma sensitivity sweep
bash scripts/sweep_edpa_gamma.sh
```

## Project Structure

```
fedeep/
├── configs/                          # YAML experiment configs
│   ├── cifar100_fedeep.yaml
│   ├── medmnist_chest_fedeep.yaml
│   └── medmnist_organ_fedeep.yaml
├── scripts/                          # Bash run scripts
└── src/
    ├── fedeep_main.py                # Single entrypoint
    ├── models/
    │   ├── convnext_early_exit.py    # ConvNeXt-Tiny + 4 exit heads
    │   └── cms.py                    # Continuum Memory System
    ├── data/
    │   ├── partition.py              # Dirichlet / pathological partition
    │   ├── cifar100.py               # CIFAR-100 loader
    │   ├── medmnist_chest.py         # ChestMNIST loader
    │   └── medmnist_organ.py         # OrganAMNIST loader
    ├── federated/
    │   ├── client.py                 # Client wrapper
    │   ├── server.py                 # FL for-loop orchestration
    │   └── aggregation.py            # FedAvg, EDPA aggregation
    ├── trainer/
    │   └── local_trainer.py          # CE + KD + Fast/Slow + CMS + phases
    ├── metrics/
    │   ├── classification.py         # Accuracy, per-exit acc, confusion matrix
    │   └── fl_metrics.py             # Round-level CSV + TensorBoard tracker
    ├── evaluation/
    │   ├── evaluator.py              # Global + per-exit evaluation
    │   └── cka_analysis.py           # CKA similarity analysis
    └── utils/
        ├── logging.py                # Stdout + file logging
        ├── checkpoint.py             # Save/load checkpoints
        ├── seed.py                   # Reproducible random seeds
        └── flops.py                  # Per-exit FLOPs estimation
```

## Progressive Phase Schedule

| Phase | Rounds  | Components Active                |
|-------|---------|----------------------------------|
| 0     | 1-20    | Backbone warmup (Exit4 CE only)  |
| 1     | 21-40   | Multi-exit CE (all 4 exits)      |
| 2     | 41-60   | + Fast/Slow split + CMS          |
| 3     | 61-80   | + Self-distillation KD chain     |
| 4     | 81-100  | Full FedEEP system               |

## EDPA Aggregation

Exit-Depth-aware Personalized Aggregation assigns different mixing ratios per parameter depth:

```
lambda_k = 1 / (1 + gamma * k)

k=0 backbone: lambda=1.00  (full global average)
k=1 exit1:    lambda=0.67  (mostly global)
k=2 exit2:    lambda=0.50  (balanced)
k=3 exit3:    lambda=0.40  (more personalized)
k=4 exit4:    lambda=0.33  (most personalized)
```

## Output

Results are saved to `src/experiments/`:
- `logs/<run_name>/train.log` — training log
- `logs/<run_name>/history.json` — full metrics history
- `checkpoints/<run_name>/final_model.pt` — final model checkpoint
