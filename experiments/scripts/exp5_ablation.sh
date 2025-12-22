#!/bin/bash
# =============================================================================
# Experiment 5: Ablation Study
# Purpose: Show contribution of each component
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
OUTPUT_DIR="experiments/results/exp5_ablation"
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Experiment 5: Ablation Study"
echo "=============================================="

# 1. Baseline: No nested learning features
echo "[1/5] Baseline (FedAvg-like)..."
python scripts/run_experiment.py \
    --mode simulation \
    --dataset $DATASET \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --slow_freq 1 \
    --fast_lr_mult 1.0 \
    --no_lss \
    --cms_levels 0 \
    --seed $SEED \
    2>&1 | tee "$OUTPUT_DIR/baseline.log"

# 2. + Nested (fast/slow separation)
echo "[2/5] + Nested Learning (K=5, fast_lr=3x)..."
python scripts/run_experiment.py \
    --mode simulation \
    --dataset $DATASET \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --slow_freq 5 \
    --fast_lr_mult 3.0 \
    --no_lss \
    --cms_levels 0 \
    --seed $SEED \
    2>&1 | tee "$OUTPUT_DIR/nested_only.log"

# 3. + LSS
echo "[3/5] + LSS..."
python scripts/run_experiment.py \
    --mode simulation \
    --dataset $DATASET \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --slow_freq 5 \
    --fast_lr_mult 3.0 \
    --use_lss \
    --cms_levels 0 \
    --seed $SEED \
    2>&1 | tee "$OUTPUT_DIR/nested_lss.log"

# 4. + CMS
echo "[4/5] + CMS (4 levels)..."
python scripts/run_experiment.py \
    --mode simulation \
    --dataset $DATASET \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --slow_freq 5 \
    --fast_lr_mult 3.0 \
    --use_lss \
    --cms_levels 4 \
    --seed $SEED \
    2>&1 | tee "$OUTPUT_DIR/full.log"

# 5. Full + DMGD
echo "[5/5] + DMGD (all features)..."
python scripts/run_experiment.py \
    --mode simulation \
    --dataset $DATASET \
    --num_clients $NUM_CLIENTS \
    --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --batch_size $BATCH_SIZE \
    --slow_freq 5 \
    --fast_lr_mult 3.0 \
    --use_lss \
    --cms_levels 4 \
    --use_dmgd \
    --seed $SEED \
    2>&1 | tee "$OUTPUT_DIR/full_dmgd.log"

echo ""
echo "=============================================="
echo "Experiment 5 Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Expected Table:"
echo "| Method     | Accuracy | Forgetting |"
echo "|------------|----------|------------|"
echo "| Baseline   | -        | -          |"
echo "| + Nested   | -        | -          |"
echo "| + LSS      | -        | -          |"
echo "| + CMS      | -        | -          |"
echo "| + DMGD     | -        | -          |"
echo "=============================================="
