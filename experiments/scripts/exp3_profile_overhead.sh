#!/bin/bash
# =============================================================================
# Experiment 3: Profiling Overhead
# Purpose: Answer reviewer question "What's the memory/computation overhead?"
# =============================================================================

set -e

# Configuration
DATASET="cifar10"
NUM_CLIENTS=1
NUM_ROUNDS=3
LOCAL_EPOCHS=2
BATCH_SIZE=32

# Output directory
OUTPUT_DIR="experiments/results/exp3_profile_overhead"
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Experiment 3: Feature Overhead Profiling"
echo "=============================================="

# Run with nvidia-smi monitoring
monitor_gpu() {
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > "$1" &
    echo $!
}

# 1. Baseline (no LSS, no CMS)
echo "[1/5] Baseline (no LSS, no CMS)..."
python scripts/run_experiment.py \
    --mode simulation \
    --dataset $DATASET \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --no_lss \
    --cms_levels 0 \
    2>&1 | tee "$OUTPUT_DIR/baseline.log"

# 2. With LSS only
echo "[2/5] With LSS only..."
python scripts/run_experiment.py \
    --mode simulation \
    --dataset $DATASET \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --use_lss \
    --cms_levels 0 \
    2>&1 | tee "$OUTPUT_DIR/with_lss.log"

# 3. With CMS (2 levels)
echo "[3/5] With CMS 2 levels..."
python scripts/run_experiment.py \
    --mode simulation \
    --dataset $DATASET \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --no_lss \
    --cms_levels 2 \
    2>&1 | tee "$OUTPUT_DIR/cms_2level.log"

# 4. With CMS (4 levels)
echo "[4/5] With CMS 4 levels..."
python scripts/run_experiment.py \
    --mode simulation \
    --dataset $DATASET \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --no_lss \
    --cms_levels 4 \
    2>&1 | tee "$OUTPUT_DIR/cms_4level.log"

# 5. All features
echo "[5/5] All features (LSS + CMS 4 + DMGD)..."
python scripts/run_experiment.py \
    --mode simulation \
    --dataset $DATASET \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --use_lss \
    --cms_levels 4 \
    --use_dmgd \
    2>&1 | tee "$OUTPUT_DIR/all_features.log"

echo ""
echo "=============================================="
echo "Experiment 3 Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Parse timing from logs:"
echo "  grep 'Round Time' $OUTPUT_DIR/*.log"
echo "=============================================="
