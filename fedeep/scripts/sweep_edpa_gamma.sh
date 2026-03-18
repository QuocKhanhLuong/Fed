#!/bin/bash
# Sweep EDPA gamma parameter on CIFAR-100
# Produces results for sensitivity analysis figure in the paper
set -e

cd "$(dirname "$0")/.."

echo "=== EDPA Gamma Sweep ==="

for GAMMA in 0.0 0.1 0.25 0.5 0.75 1.0 2.0; do
    echo ""
    echo "--- gamma=$GAMMA ---"
    python -m src.fedeep_main \
        --config configs/cifar100_fedeep.yaml \
        --strategy edpa \
        --edpa-gamma "$GAMMA" \
        "$@"
done

echo ""
echo "=== Sweep complete ==="
