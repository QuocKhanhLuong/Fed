#!/bin/bash
# =============================================================================
# Experiment 2: Scalability Test
# Purpose: Answer reviewer question "Only 3-5 clients, does it scale?"
# =============================================================================

set -e

# Configuration
DATASET="cifar10"
NUM_ROUNDS=30
LOCAL_EPOCHS=3
BATCH_SIZE=64
SEED=42

# Output directory
OUTPUT_DIR="experiments/results/exp2_scale_clients"
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Experiment 2: Scalability Test"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Rounds: $NUM_ROUNDS"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Test different number of clients
for N in 5 10 20 50; do
    # Adjust client fraction for large N
    if [ $N -ge 20 ]; then
        FRAC=0.2
    else
        FRAC=1.0
    fi
    
    echo ""
    echo "[N=$N, frac=$FRAC] Starting experiment..."
    
    python scripts/run_experiment.py \
        --mode simulation \
        --dataset $DATASET \
        --num_clients $N \
        --client_fraction $FRAC \
        --num_rounds $NUM_ROUNDS \
        --local_epochs $LOCAL_EPOCHS \
        --batch_size $BATCH_SIZE \
        --slow_freq 5 \
        --seed $SEED \
        2>&1 | tee "$OUTPUT_DIR/clients_${N}.log"
    
    # Move results
    LATEST=$(ls -td experiments/logs/*/ | head -1)
    mv "$LATEST" "$OUTPUT_DIR/clients_${N}/"
    
    echo "[N=$N] Done. Results: $OUTPUT_DIR/clients_${N}/"
done

echo ""
echo "=============================================="
echo "Experiment 2 Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
