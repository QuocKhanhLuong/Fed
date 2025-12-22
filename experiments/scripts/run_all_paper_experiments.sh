#!/bin/bash
# =============================================================================
# CORE EXPERIMENT: Full Paper Experiments (IEEE Standard)
# Based on: FedAvg, FedProx, JMLR 2023 standards
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION (Following Industry Standards)
# =============================================================================

# Dataset and Model
DATASET="cifar10"
NUM_CLASSES=10

# FL Setup (Standard from FedAvg paper)
NUM_CLIENTS=100              # Standard: 100 clients
SAMPLE_FRAC=0.1              # 10% per round = 10 clients
NUM_ROUNDS=200               # Standard: 100-600 rounds

# Training
LOCAL_EPOCHS=5               # E=5 (FedAvg standard)
BATCH_SIZE=32                # B=32
LEARNING_RATE=0.01           # η=0.01

# Non-IID via Dirichlet (Standard method)
PARTITION="dirichlet"
ALPHA=0.1                    # α=0.1 (severe non-IID)

# Reproducibility
SEEDS="42 123 456 789 2024"  # 5 random seeds

# Output
OUTPUT_BASE="experiments/results/paper_experiments"
mkdir -p $OUTPUT_BASE

echo "============================================================"
echo "  IEEE Standard FL Experiments"
echo "============================================================"
echo "  Dataset:      $DATASET"
echo "  Clients:      $NUM_CLIENTS (sample $SAMPLE_FRAC per round)"
echo "  Rounds:       $NUM_ROUNDS"
echo "  Partition:    $PARTITION (α=$ALPHA)"
echo "  Seeds:        $SEEDS"
echo "============================================================"

# =============================================================================
# EXPERIMENT 1: Baseline Comparison (CORE - Table III)
# =============================================================================
run_baselines() {
    echo ""
    echo "============================================================"
    echo "EXPERIMENT 1: Baseline Comparison"
    echo "============================================================"
    
    OUTPUT_DIR="$OUTPUT_BASE/exp1_baselines"
    mkdir -p $OUTPUT_DIR
    
    for SEED in $SEEDS; do
        echo "[Seed=$SEED] Running baselines..."
        
        # FedAvg-like (K=1, no features)
        python scripts/run_experiment.py \
            --mode simulation \
            --dataset $DATASET \
            --num_clients $NUM_CLIENTS \
            --client_fraction $SAMPLE_FRAC \
            --num_rounds $NUM_ROUNDS \
            --local_epochs $LOCAL_EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LEARNING_RATE \
            --partition $PARTITION \
            --alpha $ALPHA \
            --slow_freq 1 --fast_lr_mult 1.0 \
            --no_lss --cms_levels 0 \
            --seed $SEED \
            2>&1 | tee "$OUTPUT_DIR/fedavg_seed${SEED}.log"
        
        # FedProx
        python scripts/run_experiment.py \
            --mode simulation \
            --dataset $DATASET \
            --num_clients $NUM_CLIENTS \
            --client_fraction $SAMPLE_FRAC \
            --num_rounds $NUM_ROUNDS \
            --local_epochs $LOCAL_EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LEARNING_RATE \
            --fedprox_mu 0.01 \
            --partition $PARTITION \
            --alpha $ALPHA \
            --slow_freq 1 --fast_lr_mult 1.0 \
            --no_lss --cms_levels 0 \
            --seed $SEED \
            2>&1 | tee "$OUTPUT_DIR/fedprox_seed${SEED}.log"
        
        # OURS: NL-FL
        python scripts/run_experiment.py \
            --mode simulation \
            --dataset $DATASET \
            --num_clients $NUM_CLIENTS \
            --client_fraction $SAMPLE_FRAC \
            --num_rounds $NUM_ROUNDS \
            --local_epochs $LOCAL_EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LEARNING_RATE \
            --partition $PARTITION \
            --alpha $ALPHA \
            --slow_freq 5 --fast_lr_mult 3.0 \
            --use_lss --cms_levels 4 \
            --seed $SEED \
            2>&1 | tee "$OUTPUT_DIR/nlfl_seed${SEED}.log"
    done
    
    echo "[Exp1] Complete. Results: $OUTPUT_DIR"
}

# =============================================================================
# EXPERIMENT 2: Non-IID Severity (Table IV)
# =============================================================================
run_noniid_severity() {
    echo ""
    echo "============================================================"
    echo "EXPERIMENT 2: Non-IID Severity (vary α)"
    echo "============================================================"
    
    OUTPUT_DIR="$OUTPUT_BASE/exp2_noniid"
    mkdir -p $OUTPUT_DIR
    
    SEED=42  # Single seed for quick run
    
    for ALPHA_VAL in 0.01 0.1 0.5 1.0 10.0; do
        echo "[α=$ALPHA_VAL] Running..."
        
        python scripts/run_experiment.py \
            --mode simulation \
            --dataset $DATASET \
            --num_clients $NUM_CLIENTS \
            --client_fraction $SAMPLE_FRAC \
            --num_rounds $NUM_ROUNDS \
            --local_epochs $LOCAL_EPOCHS \
            --batch_size $BATCH_SIZE \
            --lr $LEARNING_RATE \
            --partition $PARTITION \
            --alpha $ALPHA_VAL \
            --slow_freq 5 --fast_lr_mult 3.0 \
            --use_lss --cms_levels 4 \
            --seed $SEED \
            2>&1 | tee "$OUTPUT_DIR/alpha_${ALPHA_VAL}.log"
    done
    
    echo "[Exp2] Complete. Results: $OUTPUT_DIR"
}

# =============================================================================
# EXPERIMENT 3: Ablation Study (Table V)
# =============================================================================
run_ablation() {
    echo ""
    echo "============================================================"
    echo "EXPERIMENT 3: Ablation Study"
    echo "============================================================"
    
    OUTPUT_DIR="$OUTPUT_BASE/exp3_ablation"
    mkdir -p $OUTPUT_DIR
    
    SEED=42
    
    # Baseline
    echo "[1/5] Baseline..."
    python scripts/run_experiment.py \
        --mode simulation \
        --dataset $DATASET --num_clients $NUM_CLIENTS \
        --client_fraction $SAMPLE_FRAC --num_rounds $NUM_ROUNDS \
        --local_epochs $LOCAL_EPOCHS --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE --partition $PARTITION --alpha $ALPHA \
        --slow_freq 1 --fast_lr_mult 1.0 --no_lss --cms_levels 0 \
        --seed $SEED 2>&1 | tee "$OUTPUT_DIR/baseline.log"
    
    # +Nested (K=5)
    echo "[2/5] +Nested..."
    python scripts/run_experiment.py \
        --mode simulation \
        --dataset $DATASET --num_clients $NUM_CLIENTS \
        --client_fraction $SAMPLE_FRAC --num_rounds $NUM_ROUNDS \
        --local_epochs $LOCAL_EPOCHS --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE --partition $PARTITION --alpha $ALPHA \
        --slow_freq 5 --fast_lr_mult 3.0 --no_lss --cms_levels 0 \
        --seed $SEED 2>&1 | tee "$OUTPUT_DIR/nested.log"
    
    # +LSS
    echo "[3/5] +LSS..."
    python scripts/run_experiment.py \
        --mode simulation \
        --dataset $DATASET --num_clients $NUM_CLIENTS \
        --client_fraction $SAMPLE_FRAC --num_rounds $NUM_ROUNDS \
        --local_epochs $LOCAL_EPOCHS --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE --partition $PARTITION --alpha $ALPHA \
        --slow_freq 5 --fast_lr_mult 3.0 --use_lss --cms_levels 0 \
        --seed $SEED 2>&1 | tee "$OUTPUT_DIR/nested_lss.log"
    
    # +CMS
    echo "[4/5] +CMS..."
    python scripts/run_experiment.py \
        --mode simulation \
        --dataset $DATASET --num_clients $NUM_CLIENTS \
        --client_fraction $SAMPLE_FRAC --num_rounds $NUM_ROUNDS \
        --local_epochs $LOCAL_EPOCHS --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE --partition $PARTITION --alpha $ALPHA \
        --slow_freq 5 --fast_lr_mult 3.0 --use_lss --cms_levels 4 \
        --seed $SEED 2>&1 | tee "$OUTPUT_DIR/full.log"
    
    # +DMGD
    echo "[5/5] +DMGD..."
    python scripts/run_experiment.py \
        --mode simulation \
        --dataset $DATASET --num_clients $NUM_CLIENTS \
        --client_fraction $SAMPLE_FRAC --num_rounds $NUM_ROUNDS \
        --local_epochs $LOCAL_EPOCHS --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE --partition $PARTITION --alpha $ALPHA \
        --slow_freq 5 --fast_lr_mult 3.0 --use_lss --cms_levels 4 --use_dmgd \
        --seed $SEED 2>&1 | tee "$OUTPUT_DIR/full_dmgd.log"
    
    echo "[Exp3] Complete. Results: $OUTPUT_DIR"
}

# =============================================================================
# EXPERIMENT 4: K Sensitivity (Figure 3)
# =============================================================================
run_sensitivity_K() {
    echo ""
    echo "============================================================"
    echo "EXPERIMENT 4: K Sensitivity Analysis"
    echo "============================================================"
    
    OUTPUT_DIR="$OUTPUT_BASE/exp4_sensitivity_K"
    mkdir -p $OUTPUT_DIR
    
    SEED=42
    
    for K in 1 2 3 5 7 10 15 20; do
        echo "[K=$K] Running..."
        
        python scripts/run_experiment.py \
            --mode simulation \
            --dataset $DATASET --num_clients $NUM_CLIENTS \
            --client_fraction $SAMPLE_FRAC --num_rounds $NUM_ROUNDS \
            --local_epochs $LOCAL_EPOCHS --batch_size $BATCH_SIZE \
            --lr $LEARNING_RATE --partition $PARTITION --alpha $ALPHA \
            --slow_freq $K --fast_lr_mult 3.0 --use_lss --cms_levels 4 \
            --seed $SEED 2>&1 | tee "$OUTPUT_DIR/K_${K}.log"
    done
    
    echo "[Exp4] Complete. Results: $OUTPUT_DIR"
}

# =============================================================================
# EXPERIMENT 5: Scalability (Figure 4)
# =============================================================================
run_scalability() {
    echo ""
    echo "============================================================"
    echo "EXPERIMENT 5: Scalability (vary # clients)"
    echo "============================================================"
    
    OUTPUT_DIR="$OUTPUT_BASE/exp5_scalability"
    mkdir -p $OUTPUT_DIR
    
    SEED=42
    
    for N in 10 20 50 100 200; do
        # Adjust sample fraction for consistent # sampled
        if [ $N -le 20 ]; then
            FRAC=0.5  # Sample 50% for small N
        else
            FRAC=0.1  # Sample 10% for large N
        fi
        
        echo "[N=$N, frac=$FRAC] Running..."
        
        python scripts/run_experiment.py \
            --mode simulation \
            --dataset $DATASET --num_clients $N \
            --client_fraction $FRAC --num_rounds $NUM_ROUNDS \
            --local_epochs $LOCAL_EPOCHS --batch_size $BATCH_SIZE \
            --lr $LEARNING_RATE --partition $PARTITION --alpha $ALPHA \
            --slow_freq 5 --fast_lr_mult 3.0 --use_lss --cms_levels 4 \
            --seed $SEED 2>&1 | tee "$OUTPUT_DIR/clients_${N}.log"
    done
    
    echo "[Exp5] Complete. Results: $OUTPUT_DIR"
}

# =============================================================================
# MAIN
# =============================================================================

echo ""
echo "Select experiment to run:"
echo "  1) Baselines (FedAvg vs FedProx vs NL-FL) - ~15 hours"
echo "  2) Non-IID Severity (α sweep) - ~5 hours"
echo "  3) Ablation Study - ~5 hours"
echo "  4) K Sensitivity - ~8 hours"
echo "  5) Scalability - ~6 hours"
echo "  all) Run all experiments - ~40 hours"
echo ""
read -p "Enter choice [1-5 or all]: " choice

case $choice in
    1) run_baselines ;;
    2) run_noniid_severity ;;
    3) run_ablation ;;
    4) run_sensitivity_K ;;
    5) run_scalability ;;
    all)
        run_baselines
        run_noniid_severity
        run_ablation
        run_sensitivity_K
        run_scalability
        ;;
    *)
        echo "Invalid choice. Running quick test..."
        python scripts/run_experiment.py \
            --mode simulation \
            --dataset cifar10 \
            --num_clients 10 \
            --num_rounds 5 \
            --local_epochs 2
        ;;
esac

echo ""
echo "============================================================"
echo "All selected experiments complete!"
echo "Results saved to: $OUTPUT_BASE"
echo "============================================================"
