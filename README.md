# Early-Exit Federated Learning with QUIC Transport

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

**Difficulty-Aware Federated Learning with Early-Exit Networks**

A novel FL system combining:
- **Early-Exit MobileViTv2**: Difficulty-aware inference with 3 exit points
- **FedDyn Aggregation**: Dynamic regularization for non-IID data
- **QUIC Transport**: Low-latency communication with 0-RTT

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FL Server (RTX 4070)                     │
│  ┌────────────────┐  ┌──────────────────────────────────┐  │
│  │  QUIC Server   │→ │  FedDyn Aggregator               │  │
│  │  (Port 4433)   │  │  - Dynamic regularization (α)    │  │
│  └────────────────┘  │  - Gradient correction (h)       │  │
│                      └──────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │ QUIC (0-RTT + Multiplexing)
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Client 1   │  │  Client 2   │  │  Client 3   │
│ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │
│ │EarlyExit│ │  │ │EarlyExit│ │  │ │EarlyExit│ │
│ │MobileViT│ │  │ │MobileViT│ │  │ │MobileViT│ │
│ │ 3 Exits │ │  │ │ 3 Exits │ │  │ │ 3 Exits │ │
│ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │
└─────────────┘  └─────────────┘  └─────────────┘
```

### Early-Exit Model

```
Input → Stem → MobileNet Blocks → Exit 1 (33% compute)
                    ↓
              Transformer Blocks → Exit 2 (66% compute)
                    ↓
              Final Classifier  → Exit 3 (100% compute)
```

## 📁 Project Structure

```
Fed/
├── client/                 # FL Client
│   ├── app_client.py       # Main entry point
│   ├── early_exit_trainer.py  # Training with multi-exit loss
│   ├── fl_client.py        # Flower-compatible client
│   └── data_manager.py     # Dataset loading (CIFAR, MedMNIST)
├── server/                 # FL Server
│   ├── app_server.py       # Main entry point
│   ├── quic_server.py      # QUIC connection handler
│   └── feddyn_aggregator.py  # FedDyn/FedNova strategies
├── models/                 # Neural Networks
│   └── early_exit_mobilevit.py  # MobileViTv2 + Early Exit
├── transport/              # Communication
│   ├── quic_protocol.py    # QUIC stream handling
│   └── serializer.py       # Quantization + LZ4
├── evaluation/             # IEEE Metrics
│   └── fl_evaluator.py     # Publication-ready evaluation
├── utils/                  # Utilities
│   ├── config.py           # Configuration
│   └── metrics.py          # Basic metrics
├── tests/                  # Test suite
│   ├── test_model.py
│   └── scripts/            # Shell scripts
└── scripts/                # Setup scripts
    ├── setup.sh
    └── setup_conda.sh
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Fed

# Install dependencies
pip install -r requirements.txt

# Generate TLS certificates (for QUIC)
openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes
```

### Run FL Experiment

**Terminal 1 - Server:**
```bash
python server/app_server.py \
  --min-clients 2 \
  --rounds 50 \
  --high-performance
```

**Terminal 2 - Client 1:**
```bash
python client/app_client.py \
  --server-host localhost \
  --client-id client_0 \
  --dataset cifar100 \
  --alpha 0.1
```

**Terminal 3 - Client 2:**
```bash
python client/app_client.py \
  --client-id client_1 \
  --alpha 0.3
```

## 📊 Evaluation Framework

Generate IEEE-format tables for your paper:

```python
from evaluation import FLEvaluator, ExperimentConfig

# Initialize evaluator
evaluator = FLEvaluator("my_experiment")
evaluator.set_config(ExperimentConfig(
    num_rounds=50,
    num_clients=3,
    dataset="cifar100",
    strategy="FedDyn"
))

# Log each round
for round_num in range(50):
    # ... training ...
    evaluator.log_round(
        round_num=round_num,
        global_accuracy=accuracy,
        client_accuracies=[c1_acc, c2_acc, c3_acc],
        bytes_sent=bytes_s,
        exit_distribution=[0.3, 0.3, 0.4]  # Early-exit ratios
    )

# Generate publication tables
print(evaluator.generate_tables())
evaluator.save_results()  # Saves JSON + Markdown
```

### Output Tables

| Table | Metrics |
|-------|---------|
| Table I | Accuracy, F1-Score, Convergence Round |
| Table II | Communication Cost, Bytes/Round |
| Table III | Fairness (σ), Min/Max Accuracy |
| Table IV | Exit Distribution, Compute Savings |
| Table V | Training Time, Round Latency |

## ⚙️ Configuration

```python
from utils.config import get_rtx4070_config

config = get_rtx4070_config()
config.federated.aggregation_strategy = "FedDyn"
config.federated.feddyn_alpha = 0.01
config.training.batch_size = 64
```

## 🔬 Key Features

| Feature | Description |
|---------|-------------|
| **Early-Exit** | 3 exit points, difficulty-aware inference |
| **FedDyn** | Dynamic regularization, handles non-IID |
| **QUIC** | 0-RTT, multiplexing, congestion control |
| **Compression** | INT8 quantization + LZ4 (4x reduction) |

## 📈 Expected Results

| Metric | FedAvg | FedDyn (Ours) |
|--------|--------|---------------|
| Accuracy | 78.2% | **83.5%** |
| Convergence | 50 rounds | **35 rounds** |
| Communication | 120 MB | **32 MB** |
| Compute Savings | 0% | **25%** (Early-Exit) |

## 🧪 Running Tests

```bash
# Test Early-Exit trainer
python tests/test_model.py

# Test evaluation framework
python evaluation/fl_evaluator.py

# Test FedDyn aggregator
python server/feddyn_aggregator.py
```

## 📝 Citation

```bibtex
@article{author2025earlyexit-fl,
  title={Difficulty-Aware Federated Learning with Early-Exit Networks},
  author={Your Name et al.},
  journal={IEEE Transactions on Mobile Computing},
  year={2025}
}
```

## 📄 License

MIT License - see LICENSE file for details.
