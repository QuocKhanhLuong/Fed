#!/bin/bash
# =============================================================================
# Hyperparameter Benchmark Script for Nested Learning FL
# Tests: image_size, batch_size, num_workers
# =============================================================================

echo "============================================================"
echo "Nested Learning Hyperparameter Benchmark"
echo "Device: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU')"
echo "============================================================"

# Create results directory
RESULTS_DIR="./benchmark_results"
mkdir -p $RESULTS_DIR

# Benchmark configurations
IMAGE_SIZES="128 160 192 224"
BATCH_SIZES="8 16 32 64"
NUM_WORKERS="0 1 2 4"

# Python benchmark script
cat > /tmp/benchmark_params.py << 'EOF'
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import json
import argparse
from torch.utils.data import DataLoader, TensorDataset

def benchmark(image_size, batch_size, num_workers, num_iters=10):
    """Run training benchmark with given parameters."""
    from client.nested_trainer import NestedEarlyExitTrainer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy data
    x = torch.randn(batch_size * num_iters, 3, image_size, image_size)
    y = torch.randint(0, 100, (batch_size * num_iters,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    # Initialize trainer
    trainer = NestedEarlyExitTrainer(
        num_classes=100,
        device=device,
        use_mixed_precision=True,
        cms_enabled=True,
    )
    
    # Warmup
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        break
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    try:
        metrics = trainer.train(loader, epochs=1)
        success = True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            success = False
            metrics = {"error": "OOM"}
            torch.cuda.empty_cache()
        else:
            raise e
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    
    # Memory usage
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated() / 1e9
        mem_reserved = torch.cuda.max_memory_reserved() / 1e9
        torch.cuda.reset_peak_memory_stats()
    else:
        mem_allocated = 0
        mem_reserved = 0
    
    result = {
        "image_size": image_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "time_seconds": round(elapsed, 2),
        "samples_per_sec": round(batch_size * num_iters / elapsed, 1) if success else 0,
        "memory_gb": round(mem_allocated, 2),
        "success": success,
    }
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output", type=str, default="result.json")
    args = parser.parse_args()
    
    result = benchmark(args.image_size, args.batch_size, args.num_workers)
    
    print(f"Result: {json.dumps(result, indent=2)}")
    
    with open(args.output, "w") as f:
        json.dump(result, f)
EOF

# Run benchmarks
echo ""
echo "Running benchmarks..."
echo ""

RESULTS_FILE="$RESULTS_DIR/benchmark_$(date +%Y%m%d_%H%M%S).json"
echo "[" > $RESULTS_FILE

FIRST=1
for img_size in $IMAGE_SIZES; do
    for batch in $BATCH_SIZES; do
        for workers in $NUM_WORKERS; do
            echo "Testing: image=$img_size, batch=$batch, workers=$workers"
            
            TEMP_RESULT="/tmp/result_${img_size}_${batch}_${workers}.json"
            
            python /tmp/benchmark_params.py \
                --image_size $img_size \
                --batch_size $batch \
                --num_workers $workers \
                --output $TEMP_RESULT 2>/dev/null
            
            if [ -f $TEMP_RESULT ]; then
                if [ $FIRST -eq 1 ]; then
                    FIRST=0
                else
                    echo "," >> $RESULTS_FILE
                fi
                cat $TEMP_RESULT >> $RESULTS_FILE
                rm $TEMP_RESULT
            fi
        done
    done
done

echo "" >> $RESULTS_FILE
echo "]" >> $RESULTS_FILE

echo ""
echo "============================================================"
echo "Benchmark complete! Results saved to: $RESULTS_FILE"
echo "============================================================"

# Print summary
python << EOF
import json
with open("$RESULTS_FILE") as f:
    results = json.load(f)

print("\n📊 SUMMARY (sorted by samples/sec):\n")
print(f"{'Image':>6} {'Batch':>6} {'Workers':>8} {'Time':>8} {'Samples/s':>10} {'Memory':>8} {'Status':>8}")
print("-" * 70)

sorted_results = sorted([r for r in results if r.get('success', False)], 
                        key=lambda x: x.get('samples_per_sec', 0), 
                        reverse=True)

for r in sorted_results[:10]:
    print(f"{r['image_size']:>6} {r['batch_size']:>6} {r['num_workers']:>8} "
          f"{r['time_seconds']:>7.1f}s {r['samples_per_sec']:>10.1f} "
          f"{r['memory_gb']:>7.1f}GB {'✅' if r['success'] else '❌':>8}")

# Best config
if sorted_results:
    best = sorted_results[0]
    print(f"\n🏆 BEST CONFIG: image_size={best['image_size']}, batch_size={best['batch_size']}, num_workers={best['num_workers']}")
    print(f"   Throughput: {best['samples_per_sec']} samples/sec, Memory: {best['memory_gb']} GB")
EOF
