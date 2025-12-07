#!/bin/bash
# =============================================================================
# Quick Test Script - Find optimal settings fast
# =============================================================================

echo "🚀 Quick Parameter Test (RTX 4070)"
echo "============================================================"

cd "$(dirname "$0")/.."

# Quick test common configurations
python << 'EOF'
import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import time
from torch.utils.data import DataLoader, TensorDataset
from client.nested_trainer import NestedEarlyExitTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n" + "="*60)
print("Testing configurations...")
print("="*60 + "\n")

configs = [
    # (image_size, batch_size, num_workers)
    (128, 64, 4),   # Fast, low memory
    (160, 32, 4),   # Balanced
    (192, 32, 4),   # Medium
    (224, 16, 2),   # High quality, safe
    (224, 32, 2),   # High quality, may OOM
    (224, 64, 4),   # High quality, likely OOM
]

results = []

for img_size, batch, workers in configs:
    print(f"Testing: image={img_size}, batch={batch}, workers={workers}...", end=" ", flush=True)
    
    try:
        # Create data
        x = torch.randn(batch * 5, 3, img_size, img_size)
        y = torch.randint(0, 100, (batch * 5,))
        loader = DataLoader(TensorDataset(x, y), batch_size=batch, num_workers=workers)
        
        trainer = NestedEarlyExitTrainer(
            num_classes=100,
            device=device,
            use_mixed_precision=True,
            cms_enabled=True,
        )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        trainer.train(loader, epochs=1)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        samples_sec = batch * 5 / elapsed
        
        results.append({
            "config": f"{img_size}×{img_size}, batch={batch}, workers={workers}",
            "samples_sec": samples_sec,
            "memory_gb": mem,
            "status": "✅"
        })
        print(f"✅ {samples_sec:.1f} samples/sec, {mem:.1f} GB")
        
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        del trainer, loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ OOM")
            results.append({
                "config": f"{img_size}×{img_size}, batch={batch}, workers={workers}",
                "samples_sec": 0,
                "memory_gb": 0,
                "status": "❌ OOM"
            })
            torch.cuda.empty_cache()
        else:
            print(f"❌ Error: {e}")

print("\n" + "="*60)
print("📊 RESULTS (sorted by throughput)")
print("="*60 + "\n")

results_ok = [r for r in results if r["status"] == "✅"]
results_ok.sort(key=lambda x: x["samples_sec"], reverse=True)

for r in results_ok:
    print(f"  {r['config']:40} → {r['samples_sec']:6.1f} samples/sec, {r['memory_gb']:.1f} GB")

if results_ok:
    best = results_ok[0]
    print(f"\n🏆 RECOMMENDED: {best['config']}")
    print(f"   Throughput: {best['samples_sec']:.1f} samples/sec")
EOF
