# Nested Early-Exit Federated Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

**Difficulty-Aware Federated Learning with Nested Early-Exit Networks**

Implementation of Nested Learning (NeurIPS 2025) for Federated Learning, featuring:

| Feature | Description |
|---------|-------------|
| **Nested Learning** | Multi-timescale optimization (fast/slow weights) |
| **Early-Exit MobileViTv2** | 3 exit points with difficulty-aware inference |
| **Local Surprise Signal (LSS)** | Sample importance weighting |
| **Continuum Memory System (CMS)** | 4-level memory for catastrophic forgetting |
| **QUIC Transport** | Low-latency communication with 0-RTT |

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    FL Server (RTX 4070)                        │
│  • QUIC Server (Port 4433)                                     │
│  • FedProx/FedDyn Aggregation                                  │
└────────────────────┬───────────────────────────────────────────┘
                     │ QUIC (0-RTT + Multiplexing)
     ┌───────────────┼───────────────┐
     ▼               ▼               ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐
│ Client 1 │  │ Client 2 │  │ Jetson Nano  │
│ RTX GPU  │  │ RTX GPU  │  │ Edge Device  │
└──────────┘  └──────────┘  └──────────────┘

Each client runs:
┌─────────────────────────────────────────────┐
│        NestedEarlyExitTrainer               │
│  ┌─────────────────────────────────────┐    │
│  │ MobileViTv2 + Early Exit (3 exits)  │    │
│  │ Fast weights: Exit classifiers      │    │
│  │ Slow weights: Backbone              │    │
│  └─────────────────────────────────────┘    │
│  + LSS (Local Surprise Signal)              │
│  + CMS (4-level Continuum Memory)           │
└─────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Fed/
├── client/                     # FL Client
│   ├── nested_trainer.py       # ⭐ Main trainer (LSS, CMS, DMGD)
│   ├── early_exit_trainer.py   # Basic early-exit trainer
│   ├── app_client.py           # CLI entry point
│   └── data_manager.py         # Dataset loading
├── server/                     # FL Server
│   ├── quic_server.py          # QUIC connection handler
│   └── feddyn_aggregator.py    # Aggregation strategies
├── models/
│   └── early_exit_mobilevit.py # MobileViTv2 + 3 Early Exits
├── scripts/
│   ├── run_experiment.py       # ⭐ IEEE-style experiment runner
│   ├── setup.sh                # Linux setup
│   └── setup_conda.sh          # macOS setup
├── jetson/
│   ├── run_client.sh           # ⭐ One-click Jetson setup
│   └── README.md
├── tests/
│   ├── test_nested_features.py # ⭐ Test LSS, DMGD, CMS
│   └── test_model.py
└── utils/
    ├── config.py               # Configuration
    └── torch_compat.py         # PyTorch 1.x/2.x compatibility
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/QuocKhanhLuong/Fed.git
cd Fed

# Option 1: Conda (recommended)
./scripts/setup_conda.sh

# Option 2: pip
pip install -r requirements.txt
```

### Run Experiment (Simulation Mode)

```bash
# Quick test - 1 client, 5 rounds
python scripts/run_experiment.py \
    --mode simulation \
    --num_clients 1 \
    --num_rounds 5 \
    --dataset cifar10 \
    --batch_size 64

# Full experiment - 10 clients, non-IID
python scripts/run_experiment.py \
    --mode simulation \
    --num_clients 10 \
    --num_rounds 50 \
    --dataset cifar100 \
    --partition dirichlet \
    --alpha 0.5
```

### Run on Jetson Nano

```bash
# One-click setup and run
./jetson/run_client.sh --server <SERVER_IP>
```

## ⚙️ Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_clients` | 10 | Number of FL clients K |
| `--num_rounds` | 50 | Communication rounds T |
| `--local_epochs` | 5 | Local epochs E |
| `--partition` | dirichlet | Data partition (iid, dirichlet) |
| `--alpha` | 0.5 | Dirichlet α (lower = more non-IID) |
| `--use_lss` | True | Enable Local Surprise Signal |
| `--cms_levels` | 4 | CMS memory levels |
| `--use_dmgd` | False | Enable Deep Momentum GD |

## 🔬 Nested Learning Features (NeurIPS 2025)

### 1. Local Surprise Signal (LSS)

```python
# Weights samples by "surprise" (loss magnitude)
LSS(x) = loss(x) / E[loss]
# Higher loss = more surprising = higher weight
```

### 2. Continuum Memory System (CMS)

```python
# 4-level memory with exponential update frequencies
update_freqs = [1, 5, 25, 125]  # Steps between updates
# Fast layer: adapts immediately
# Anchor layer: preserves long-term knowledge
```

### 3. Deep Momentum GD (Optional)

```python
# MLP-based momentum instead of EMA
DeepMomentum: gradient → MLP → momentum_update
```

## 📊 Running Tests

```bash
# Test Nested Learning features
python tests/test_nested_features.py

# Test model training
python tests/test_model.py
```

## 📝 Citation

```bibtex
@article{luong2025nestedexit,
  title={Difficulty-Aware Federated Learning with Nested Early-Exit Networks},
  author={Luong, Quoc Khanh},
  journal={IEEE Transactions on Mobile Computing},
  year={2025}
}
```

## 📄 License

MIT License - see LICENSE file for details.
