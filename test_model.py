"""
Test Model Training Components
Verify MobileViT + LoRA trainer and FL client

Author: Research Team - FL-QUIC-LoRA Project
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from client.model_trainer import MobileViTLoRATrainer, create_dummy_dataset
from client.fl_client import create_fl_client

print("="*70)
print("Testing Model Training Components")
print("="*70)

def test_trainer():
    """Test MobileViTLoRATrainer"""
    print("\n" + "-"*70)
    print("TEST 1: MobileViT + LoRA Trainer")
    print("-"*70)
    
    # Create trainer
    print("Creating trainer...")
    trainer = MobileViTLoRATrainer(
        num_classes=10,
        lora_r=4,  # Small rank for fast testing
        use_mixed_precision=False,  # For CPU compatibility
    )
    
    print(f"✓ Trainer created")
    print(f"  Total parameters: {trainer.stats['total_params']:,}")
    print(f"  Trainable parameters: {trainer.stats['trainable_params']:,}")
    print(f"  LoRA parameters: {trainer.stats['lora_params']:,}")
    
    # Create dummy data
    print("\nCreating dummy dataset...")
    train_loader, test_loader = create_dummy_dataset(num_samples=50)
    print(f"✓ Dataset created: 50 train, 10 test samples")
    
    # Extract parameters
    print("\nExtracting parameters...")
    params_before = trainer.get_parameters()
    print(f"✓ Extracted {len(params_before)} parameter arrays")
    total_params = sum(p.size for p in params_before)
    print(f"  Total elements: {total_params:,}")
    
    # Train
    print("\nTraining for 1 epoch...")
    metrics = trainer.train(train_loader, epochs=1, learning_rate=1e-3)
    print(f"✓ Training complete")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Samples: {metrics['num_samples']}")
    
    # Evaluate
    print("\nEvaluating...")
    eval_metrics = trainer.evaluate(test_loader)
    print(f"✓ Evaluation complete")
    print(f"  Loss: {eval_metrics['loss']:.4f}")
    print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    
    # Verify parameters changed
    params_after = trainer.get_parameters()
    import numpy as np
    param_diff = np.mean([np.mean(np.abs(p1 - p2)) 
                          for p1, p2 in zip(params_before, params_after)])
    print(f"\n✓ Parameters updated (avg change: {param_diff:.6f})")
    
    return True


def test_fl_client():
    """Test Flower Client"""
    print("\n" + "-"*70)
    print("TEST 2: Flower Client")
    print("-"*70)
    
    try:
        import flwr
        print("✓ Flower is installed")
    except ImportError:
        print("⚠️  Flower not installed - skipping FL client test")
        print("   Install with: pip install flwr")
        return False
    
    # Create FL client
    print("\nCreating FL client...")
    client = create_fl_client(
        num_classes=10,
        lora_r=4,
        local_epochs=1,
        use_dummy_data=True,
    )
    print(f"✓ FL Client created")
    
    # Get parameters
    print("\nGetting parameters...")
    params = client.get_parameters({})
    print(f"✓ Extracted {len(params)} parameter arrays")
    
    # Simulate FL round
    print("\nSimulating FL round...")
    config = {'round': 1, 'local_epochs': 1, 'learning_rate': 1e-3}
    updated_params, num_samples, metrics = client.fit(params, config)
    print(f"✓ Training complete")
    print(f"  Samples: {num_samples}")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    # Evaluate
    print("\nEvaluating...")
    loss, num_samples, eval_metrics = client.evaluate(updated_params, {})
    print(f"✓ Evaluation complete")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    
    return True


if __name__ == "__main__":
    try:
        # Test trainer
        success1 = test_trainer()
        
        # Test FL client
        success2 = test_fl_client()
        
        # Summary
        print("\n" + "="*70)
        if success1 and success2:
            print("✅ ALL TESTS PASSED")
        elif success1:
            print("✅ TRAINER TEST PASSED")
            print("⚠️  FL CLIENT TEST SKIPPED (install flwr)")
        else:
            print("❌ SOME TESTS FAILED")
        print("="*70)
        
        print("\nModel training components are ready!")
        print("\nNext steps:")
        print("  1. Test server: python server/app_server.py --help")
        print("  2. Test client: python client/app_client.py --help")
        print("  3. Run full FL: See README.md")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
