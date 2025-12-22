# Scripts Directory

This directory contains experiment scripts for the Nested Early-Exit FL project.

## Files

| File | Description |
|------|-------------|
| `run_experiment.py` | Main experiment runner (IEEE-standard format) |
| `setup.sh` | General setup script for Linux |
| `setup_conda.sh` | Conda setup for macOS |
| `demo_transport.py` | Transport layer demo |

## Quick Start

### 1. Simulation Mode (Single GPU)

Run FL simulation with all clients on one machine:

```bash
# CIFAR-100, 10 clients, 50 rounds
python scripts/run_experiment.py \
    --mode simulation \
    --dataset cifar100 \
    --num_clients 10 \
    --num_rounds 50 \
    --local_epochs 5

# Non-IID with Dirichlet α=0.1 (highly heterogeneous)
python scripts/run_experiment.py \
    --mode simulation \
    --dataset cifar100 \
    --partition dirichlet \
    --alpha 0.1

# IID partition
python scripts/run_experiment.py \
    --mode simulation \
    --partition iid
```

### 2. Distributed Mode (Server + Jetson Clients)

**On Server (RTX 4070):**
```bash
python scripts/run_experiment.py --mode server --port 4433
```

**On Jetson Nano:**
```bash
./jetson/run_client.sh --server <SERVER_IP>
```

## Configuration (Table I)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_clients` | 10 | Number of FL clients K |
| `--num_rounds` | 50 | Communication rounds T |
| `--local_epochs` | 5 | Local epochs E |
| `--client_fraction` | 1.0 | Fraction of clients per round |
| `--dataset` | cifar100 | Dataset (cifar10, cifar100) |
| `--partition` | dirichlet | Data partition (iid, dirichlet) |
| `--alpha` | 0.5 | Dirichlet concentration α |
| `--lr` | 1e-3 | Learning rate η |
| `--fedprox_mu` | 0.01 | FedProx regularization μ |
| `--fast_lr_mult` | 3.0 | Fast weight LR multiplier |
| `--slow_freq` | 5 | Slow weight update frequency |
| `--exit_threshold` | 0.8 | Early exit confidence τ |

## Example Experiments

### Experiment 1: Impact of Data Heterogeneity

```bash
# IID baseline
python scripts/run_experiment.py --partition iid --alpha 1.0 --seed 42

# Mild non-IID
python scripts/run_experiment.py --partition dirichlet --alpha 0.5 --seed 42

# Severe non-IID
python scripts/run_experiment.py --partition dirichlet --alpha 0.1 --seed 42
```

### Experiment 2: Early Exit Threshold Ablation

```bash
for tau in 0.5 0.6 0.7 0.8 0.9; do
    python scripts/run_experiment.py --exit_threshold $tau --seed 42
done
```

### Experiment 3: Nested Learning Parameters

```bash
# Standard training (no nested)
python scripts/run_experiment.py --fast_lr_mult 1.0 --slow_freq 1

# Nested with different frequencies
python scripts/run_experiment.py --fast_lr_mult 3.0 --slow_freq 5
python scripts/run_experiment.py --fast_lr_mult 5.0 --slow_freq 10
```

## Output

Results are saved to `experiments/logs/<experiment_name>/`:
- `config.json`: Experiment configuration
- `history.json`: Training history (loss, accuracy per round)

## Citation

If you use this code, please cite:

```bibtex
@article{yourname2025nestedexit,
  title={Difficulty-Aware Federated Learning with Nested Early-Exit Networks},
  author={Your Name},
  journal={IEEE Transactions on Mobile Computing},
  year={2025}
}
```
