#!/bin/bash
# =============================================================================
# Experiment 4: CMS Levels Comparison
# Purpose: Answer reviewer question about lightweight CMS for edge
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
OUTPUT_DIR="experiments/results/exp4_cms_levels"
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Experiment 4: CMS Levels Comparison"
echo "=============================================="

# Test different CMS levels
for LEVELS in 0 2 3 4; do
    if [ $LEVELS -eq 0 ]; then
        CMS_NAME="disabled"
    else
        CMS_NAME="${LEVELS}level"
    fi
    
    echo ""
    echo "[CMS=$CMS_NAME] Starting experiment..."
    
    python scripts/run_experiment.py \
        --mode simulation \
        --dataset $DATASET \
        --num_clients $NUM_CLIENTS \
        --num_rounds $NUM_ROUNDS \
        --local_epochs $LOCAL_EPOCHS \
        --batch_size $BATCH_SIZE \
        --cms_levels $LEVELS \
        --seed $SEED \
        2>&1 | tee "$OUTPUT_DIR/cms_${CMS_NAME}.log"
    
    # Move results
    LATEST=$(ls -td experiments/logs/*/ | head -1)
    mv "$LATEST" "$OUTPUT_DIR/cms_${CMS_NAME}/"
    
    echo "[CMS=$CMS_NAME] Done."
done

echo ""
echo "=============================================="
echo "Experiment 4 Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
