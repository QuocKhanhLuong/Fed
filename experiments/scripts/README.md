# Paper Experiments

## Overview

Full experiment suite following **IEEE standard** practices from top-tier FL papers (FedAvg, FedProx, JMLR 2023).

## Quick Start

```bash
cd ~/Fed
chmod +x experiments/scripts/*.sh
./experiments/scripts/run_all_paper_experiments.sh
```

## Experiments

| # | Name | Purpose | Time |
|---|------|---------|------|
| 1 | Baselines | FedAvg vs FedProx vs Ours | ~15h |
| 2 | Non-IID | α ∈ {0.01, 0.1, 0.5, 1.0, 10.0} | ~5h |
| 3 | Ablation | +Nested, +LSS, +CMS, +DMGD | ~5h |
| 4 | K Sensitivity | K ∈ {1,2,3,5,7,10,15,20} | ~8h |
| 5 | Scalability | 10,20,50,100,200 clients | ~6h |

## Configuration (Industry Standard)

```
Clients: 100 (sample 10% = 10 per round)
Rounds: 200
Local epochs: E=5
Batch size: B=32
Learning rate: η=0.01
Non-IID: Dirichlet(α=0.1)
Seeds: 5 runs (42, 123, 456, 789, 2024)
```

## Results Location

```
experiments/results/paper_experiments/
├── exp1_baselines/
├── exp2_noniid/
├── exp3_ablation/
├── exp4_sensitivity_K/
└── exp5_scalability/
```

## Expected Tables

### Table III: Baseline Comparison
| Method | Acc (%) | Rounds to 60% |
|--------|---------|---------------|
| FedAvg | - | - |
| FedProx | - | - |
| **NL-FL (Ours)** | - | - |

### Table V: Ablation Study
| Method | Acc | Δ vs Baseline |
|--------|-----|---------------|
| Baseline | - | - |
| + Nested (K=5) | - | +X.X |
| + LSS | - | +X.X |
| + CMS | - | +X.X |
