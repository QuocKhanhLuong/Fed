#!/bin/bash
# =============================================================================
# Hyperparameter Benchmark Script for Nested Learning FL
# Tests: image_size, batch_size, num_workers
# =============================================================================

echo "============================================================"
echo "Nested Learning Hyperparameter Benchmark"
echo "Device: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU')"
echo "============================================================"

cd "$(dirname "$0")/.."

# Create results directory
RESULTS_DIR="./benchmark_results"
mkdir -p $RESULTS_DIR

RESULTS_FILE="$RESULTS_DIR/benchmark_$(date +%Y%m%d_%H%M%S).csv"

# Run Python benchmark
python << 'PYTHON_SCRIPT'
import sys
import os
import time
import csv
from datetime import datetime

sys.path.insert(0, os.getcwd())

import torch
from torch.utils.data import DataLoader, TensorDataset

# Configurations to test
IMAGE_SIZES = [128, 160, 192, 224]
BATCH_SIZES = [8, 16, 32, 64]
NUM_WORKERS = [0, 2, 4]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# Import trainer
from client.nested_trainer import NestedEarlyExitTrainer

results = []
total_tests = len(IMAGE_SIZES) * len(BATCH_SIZES) * len(NUM_WORKERS)
test_num = 0

print(f"Running {total_tests} benchmark configurations...\n")
print(f"{'#':>3} | {'Image':>5} | {'Batch':>5} | {'Workers':>7} | {'Time':>7} | {'Samples/s':>10} | {'Memory':>8} | Status")
print("-" * 75)

for img_size in IMAGE_SIZES:
    for batch_size in BATCH_SIZES:
        for num_workers in NUM_WORKERS:
            test_num += 1
            
            try:
                # Clear memory
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                
                # Create dummy data
                num_samples = batch_size * 5
                x = torch.randn(num_samples, 3, img_size, img_size)
                y = torch.randint(0, 100, (num_samples,))
                loader = DataLoader(
                    TensorDataset(x, y), 
                    batch_size=batch_size, 
                    num_workers=num_workers,
                    pin_memory=True
                )
                
                # Create trainer
                trainer = NestedEarlyExitTrainer(
                    num_classes=100,
                    device=device,
                    use_mixed_precision=True,
                    cms_enabled=True,
                )
                
                # Warmup
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Benchmark training
                start = time.time()
                metrics = trainer.train(loader, epochs=1)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                elapsed = time.time() - start
                
                # Get memory
                mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                samples_per_sec = num_samples / elapsed
                
                status = "✅"
                
                results.append({
                    'image_size': img_size,
                    'batch_size': batch_size,
                    'num_workers': num_workers,
                    'time_sec': round(elapsed, 2),
                    'samples_per_sec': round(samples_per_sec, 1),
                    'memory_gb': round(mem_gb, 2),
                    'status': 'OK'
                })
                
                print(f"{test_num:>3} | {img_size:>5} | {batch_size:>5} | {num_workers:>7} | {elapsed:>6.2f}s | {samples_per_sec:>10.1f} | {mem_gb:>7.2f}G | {status}")
                
                # Cleanup
                del trainer, loader, x, y
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    status = "❌ OOM"
                    results.append({
                        'image_size': img_size,
                        'batch_size': batch_size,
                        'num_workers': num_workers,
                        'time_sec': 0,
                        'samples_per_sec': 0,
                        'memory_gb': 0,
                        'status': 'OOM'
                    })
                    print(f"{test_num:>3} | {img_size:>5} | {batch_size:>5} | {num_workers:>7} | {'N/A':>7} | {'N/A':>10} | {'N/A':>8} | {status}")
                    torch.cuda.empty_cache()
                else:
                    print(f"{test_num:>3} | {img_size:>5} | {batch_size:>5} | {num_workers:>7} | Error: {e}")
            except Exception as e:
                print(f"{test_num:>3} | {img_size:>5} | {batch_size:>5} | {num_workers:>7} | Error: {e}")

# Summary
print("\n" + "=" * 75)
print("📊 SUMMARY - Top 10 configurations (sorted by throughput)")
print("=" * 75 + "\n")

ok_results = [r for r in results if r['status'] == 'OK']
ok_results.sort(key=lambda x: x['samples_per_sec'], reverse=True)

print(f"{'Rank':>4} | {'Image':>5} | {'Batch':>5} | {'Workers':>7} | {'Samples/s':>10} | {'Memory':>8}")
print("-" * 55)

for i, r in enumerate(ok_results[:10], 1):
    print(f"{i:>4} | {r['image_size']:>5} | {r['batch_size']:>5} | {r['num_workers']:>7} | {r['samples_per_sec']:>10.1f} | {r['memory_gb']:>7.2f}G")

if ok_results:
    best = ok_results[0]
    print(f"\n🏆 BEST CONFIG:")
    print(f"   image_size={best['image_size']}, batch_size={best['batch_size']}, num_workers={best['num_workers']}")
    print(f"   Throughput: {best['samples_per_sec']} samples/sec")
    print(f"   Memory: {best['memory_gb']} GB")

# Save to CSV
csv_file = f"benchmark_results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
os.makedirs("benchmark_results", exist_ok=True)
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['image_size', 'batch_size', 'num_workers', 'time_sec', 'samples_per_sec', 'memory_gb', 'status'])
    writer.writeheader()
    writer.writerows(results)
print(f"\n📁 Results saved to: {csv_file}")
PYTHON_SCRIPT
