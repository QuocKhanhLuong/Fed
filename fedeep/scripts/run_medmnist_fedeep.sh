#!/bin/bash
# Run FedEEP on MedMNIST datasets
set -e

cd "$(dirname "$0")/.."

DATASET=${1:-medmnist_chest}

if [ "$DATASET" = "medmnist_chest" ]; then
    CONFIG="configs/medmnist_chest_fedeep.yaml"
elif [ "$DATASET" = "medmnist_organ" ]; then
    CONFIG="configs/medmnist_organ_fedeep.yaml"
else
    echo "Unknown dataset: $DATASET"
    echo "Usage: $0 [medmnist_chest|medmnist_organ]"
    exit 1
fi

echo "=== FedEEP $DATASET ==="

python -m src.fedeep_main \
    --config "$CONFIG" \
    "${@:2}"
