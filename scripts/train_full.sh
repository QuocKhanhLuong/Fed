#!/bin/bash
# =============================================================================
# Full Training Test - Nested Learning FL on CIFAR-100
# Config: 224x224, batch_size=16, num_workers=2
# =============================================================================

echo "============================================================"
echo "Full Training Test - Nested Learning on CIFAR-100"
echo "Config: 224x224 | batch_size=16 | num_workers=2"
echo "============================================================"

cd "$(dirname "$0")/.."

python << 'PYTHON_SCRIPT'
import sys
import os
import time
import logging

sys.path.insert(0, os.getcwd())

import torch
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================
# Configuration
# ============================
CONFIG = {
    'image_size': 224,
    'batch_size': 16,
    'num_workers': 2,
    'num_classes': 100,
    'num_epochs': 50,  # Increased for better convergence with pretrained
    'learning_rate': 1e-3,
    'dataset': 'cifar100',
}

print("\n" + "="*60)
print("Configuration")
print("="*60)
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# ============================
# Device Info
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================
# Load Dataset
# ============================
print("\n" + "="*60)
print("Loading CIFAR-100 Dataset...")
print("="*60)

from client.data_manager import load_dataset

train_loader, val_loader, test_loader, stats = load_dataset(
    dataset_name=CONFIG['dataset'],
    data_dir="./data",
    client_id=0,
    num_clients=1,  # Single client for testing
    partition_type="iid",
    batch_size=CONFIG['batch_size'],
    num_workers=CONFIG['num_workers'],
)

print(f"Train samples: {stats['train_samples']}")
print(f"Val samples: {stats['val_samples']}")
print(f"Test samples: {stats['test_samples']}")

# ============================
# Initialize Trainer
# ============================
print("\n" + "="*60)
print("Initializing NestedEarlyExitTrainer...")
print("="*60)

from client.nested_trainer import NestedEarlyExitTrainer

trainer = NestedEarlyExitTrainer(
    num_classes=CONFIG['num_classes'],
    device=device,
    use_mixed_precision=True,
    use_self_distillation=True,
    cms_enabled=True,
    fast_lr_multiplier=3.0,  # Reduced from 10.0 for stability
    slow_update_freq=5,
    use_timm_pretrained=True,  # Full pretrained backbone
)

print(f"Total params: {trainer.stats['total_params']:,}")
print(f"Fast params: {trainer.stats['fast_params']:,}")
print(f"Slow params: {trainer.stats['slow_params']:,}")
print(f"CMS enabled: {trainer.stats['cms_enabled']}")

# ============================
# Training
# ============================
print("\n" + "="*60)
print(f"Training for {CONFIG['num_epochs']} epochs...")
print("="*60 + "\n")

train_history = []
val_history = []

start_time = time.time()

for epoch in range(1, CONFIG['num_epochs'] + 1):
    epoch_start = time.time()
    
    # Train
    train_metrics = trainer.train(
        train_loader,
        epochs=1,
        learning_rate=CONFIG['learning_rate'],
    )
    
    # Validate
    val_metrics = trainer.evaluate(val_loader, threshold=0.8)
    
    epoch_time = time.time() - epoch_start
    
    train_history.append(train_metrics)
    val_history.append(val_metrics)
    
    # Memory stats
    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated() / 1e9
    else:
        mem_used = 0
    
    print(f"Epoch {epoch}/{CONFIG['num_epochs']} | "
          f"Train Loss: {train_metrics['loss']:.4f} | "
          f"Train Acc: {train_metrics['accuracy']:.4f} | "
          f"Val Acc: {val_metrics['accuracy']:.4f} | "
          f"Time: {epoch_time:.1f}s | "
          f"Memory: {mem_used:.2f}GB")

total_time = time.time() - start_time

# ============================
# Final Evaluation
# ============================
print("\n" + "="*60)
print("Final Evaluation on Test Set...")
print("="*60)

test_metrics = trainer.evaluate(test_loader, threshold=0.8)

print(f"\nTest Results:")
print(f"  Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
print(f"  Loss: {test_metrics['loss']:.4f}")
print(f"  Exit Distribution: {test_metrics['exit_distribution']}")
print(f"  Avg Exit: {test_metrics['avg_exit']:.2f}")

# ============================
# Summary
# ============================
print("\n" + "="*60)
print("Summary")
print("="*60)

print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"Best Val Accuracy: {max(v['accuracy'] for v in val_history):.4f}")
print(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")

if torch.cuda.is_available():
    print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Save results
results = {
    'config': CONFIG,
    'train_history': train_history,
    'val_history': val_history,
    'test_metrics': test_metrics,
    'total_time': total_time,
}

import json
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Save JSON results
timestamp = time.strftime('%Y%m%d_%H%M%S')
with open(f"results/training_{timestamp}.json", 'w') as f:
    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    json.dump(results, f, indent=2, default=convert)

# Save model checkpoint
checkpoint = {
    'model_state_dict': trainer.model.state_dict(),
    'config': CONFIG,
    'final_accuracy': test_metrics['accuracy'],
    'best_val_accuracy': max(v['accuracy'] for v in val_history),
    'epochs_trained': CONFIG['num_epochs'],
}
checkpoint_path = f"checkpoints/model_{timestamp}.pth"
torch.save(checkpoint, checkpoint_path)

# Also save as 'best_model.pth' for easy loading
torch.save(checkpoint, "checkpoints/best_model.pth")

print(f"\n✅ Results saved to results/training_{timestamp}.json")
print(f"✅ Model saved to {checkpoint_path}")
print(f"✅ Best model saved to checkpoints/best_model.pth")
print("="*60)
PYTHON_SCRIPT
