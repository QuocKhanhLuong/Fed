#!/bin/bash
# Run FedEEP on CIFAR-100 with EDPA strategy
set -e

cd "$(dirname "$0")/.."

echo "=== FedEEP CIFAR-100 ==="

python -m src.fedeep_main \
    --config configs/cifar100_fedeep.yaml \
    "$@"
