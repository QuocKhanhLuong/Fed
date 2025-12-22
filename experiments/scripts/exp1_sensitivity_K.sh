#!/bin/bash
# =============================================================================
# Experiment 1: Sensitivity Analysis for K (slow_update_freq)
# Purpose: Answer reviewer question "Why K=5 and not K=3 or K=10?"
# =============================================================================

set -e

# Configuration
DATASET="cifar10"
NUM_CLIENTS=10
NUM_ROUNDS=30
LOCAL_EPOCHS=3
BATCH_SIZE=64
SEED=42

# Output directory
OUTPUT_DIR="experiments/results/exp1_sensitivity_K"
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Experiment 1: K Sensitivity Analysis"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Clients: $NUM_CLIENTS"
echo "Rounds: $NUM_ROUNDS"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Test different values of K
for K in 1 3 5 7 10 15 20; do
    echo ""
    echo "[K=$K] Starting experiment..."
    
    python scripts/run_experiment.py \
        --mode simulation \
        --dataset $DATASET \
        --num_clients $NUM_CLIENTS \
        --num_rounds $NUM_ROUNDS \
        --local_epochs $LOCAL_EPOCHS \
        --batch_size $BATCH_SIZE \
        --slow_freq $K \
        --fast_lr_mult 3.0 \
        --seed $SEED \
        2>&1 | tee "$OUTPUT_DIR/K_${K}.log"
    
    # Move results
    LATEST=$(ls -td experiments/logs/*/ | head -1)
    mv "$LATEST" "$OUTPUT_DIR/K_${K}/"
    
    echo "[K=$K] Done. Results: $OUTPUT_DIR/K_${K}/"
done

echo ""
echo "=============================================="
echo "Experiment 1 Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To analyze, run:"
echo "  python experiments/analysis/plot_sensitivity.py --exp_dir $OUTPUT_DIR"
echo "=============================================="
