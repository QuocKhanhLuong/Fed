#!/bin/bash
# Test the 3 Q1 Features that caused loss to increase
# Run each test for 10 rounds and identify the root cause

cd /Users/alvinluong/Fed/nestedfl

echo "=============================================="
echo "Testing 3 Q1 Features (One by One)"
echo "=============================================="
echo ""
echo "Baseline: CMS (residual), fixed K=5, distillation=false"
echo ""

# =====================================================
# Test 1: FullyNestedCMS (use_residual=False)
# =====================================================
echo "[Test 1/3] FullyNestedCMS (pure nesting, no residual)"
echo ""
echo "Command:"
echo "  flwr run . --run-config 'num-server-rounds=10'"
echo ""
echo ">>> First, manually edit early_exit_mobilevit.py:"
echo "    Change: ContinuumMemorySystem() → FullyNestedCMS(use_residual=False)"
echo ""
echo "Press Enter after running to continue..."
read

# =====================================================
# Test 2: Adaptive K (5→2)
# =====================================================
echo "[Test 2/3] Adaptive K (slow_update_freq decreases 5→2)"
echo ""
echo ">>> First, revert Test 1 changes, then edit nested_trainer.py:"
echo "    Uncomment adaptive_k logic in OUTER LOOP section"
echo ""
echo "Command:"
echo "  flwr run . --run-config 'num-server-rounds=10'"
echo ""
echo "Press Enter after running to continue..."
read

# =====================================================
# Test 3: Distillation 0.3
# =====================================================
echo "[Test 3/3] Distillation (weight=0.3)"
echo "" 
echo ">>> First, revert Test 2 changes"
echo ""
echo "Command:"
echo "  flwr run . --run-config 'num-server-rounds=10 use-distillation=true distillation-weight=0.3'"
echo ""
echo "Press Enter after running to continue..."
read

echo "=============================================="
echo "Results Summary"
echo "=============================================="
echo ""
echo "Fill in the results:"
echo ""
echo "Test 1 (FullyNestedCMS pure): Loss ______ (increasing/decreasing?)"
echo "Test 2 (Adaptive K 5→2):      Loss ______ (increasing/decreasing?)"
echo "Test 3 (Distillation 0.3):    Loss ______ (increasing/decreasing?)"
echo ""
echo "The feature with INCREASING loss is the root cause!"
echo "=============================================="
