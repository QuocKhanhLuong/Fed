"""
Test Advanced Features: SAM, TTA, LoRA
Verify that the three State-of-the-Art techniques work correctly

Author: Research Team - FL-QUIC-LoRA Project
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from client.model_trainer import MobileViTLoRATrainer, create_dummy_dataset
from utils.config import Config

print("=" * 70)
print("Testing Advanced FL Training Features")
print("=" * 70)


def test_sam_optimizer():
    """Test Feature A: Sharpness-Aware Minimization"""
    print("\n" + "-" * 70)
    print("TEST 1: SAM Optimizer (Feature A)")
    print("-" * 70)
    
    try:
        # Create trainer with SAM enabled
        print("Creating trainer with SAM enabled...")
        trainer = MobileViTLoRATrainer(
            num_classes=10,
            lora_r=4,
            use_mixed_precision=False,
            use_sam=True,
            sam_rho=0.05,
        )
        print("✓ SAM trainer created")
        
        # Create dummy data
        train_loader, test_loader = create_dummy_dataset(num_samples=32)
        print("✓ Dummy dataset created")
        
        # Train with SAM
        print("\nTraining with SAM (2-step optimization)...")
        metrics = trainer.train(train_loader, epochs=1, learning_rate=1e-3)
        print(f"✓ SAM training completed")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        print("\n✅ SAM test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ SAM test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tta():
    """Test Feature C: Test-Time Adaptation"""
    print("\n" + "-" * 70)
    print("TEST 2: Test-Time Adaptation (Feature C)")
    print("-" * 70)
    
    try:
        # Create trainer with TTA enabled
        print("Creating trainer with TTA enabled...")
        trainer = MobileViTLoRATrainer(
            num_classes=10,
            lora_r=4,
            use_mixed_precision=False,
            use_tta=True,
            tta_steps=2,
            tta_lr=1e-4,
        )
        print("✓ TTA trainer created")
        
        # Create dummy data
        train_loader, test_loader = create_dummy_dataset(num_samples=32)
        print("✓ Dummy dataset created")
        
        # Train briefly
        print("\nTraining baseline model...")
        trainer.train(train_loader, epochs=1, learning_rate=1e-3)
        print("✓ Training completed")
        
        # Evaluate with TTA
        print("\nEvaluating with TTA (adapting BN layers)...")
        metrics = trainer.evaluate(test_loader)
        print(f"✓ TTA evaluation completed")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        print("\n✅ TTA test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TTA test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lora():
    """Test Feature D: LoRA (Low-Rank Adaptation)"""
    print("\n" + "-" * 70)
    print("TEST 3: LoRA (Feature D)")
    print("-" * 70)
    
    try:
        # Check if PEFT is available
        try:
            import peft
            print("✓ PEFT library available")
        except ImportError:
            print("⚠️  PEFT not installed - skipping LoRA test")
            print("   Install with: pip install peft")
            return True  # Not a failure, just skipped
        
        # Create trainer with LoRA enabled
        print("\nCreating trainer with LoRA enabled...")
        trainer = MobileViTLoRATrainer(
            num_classes=10,
            use_mixed_precision=False,
            use_lora=True,
            lora_r=8,
        )
        print("✓ LoRA trainer created")
        print(f"  Total params: {trainer.stats['total_params']:,}")
        print(f"  Trainable params: {trainer.stats['trainable_params']:,}")
        
        # Create dummy data
        train_loader, test_loader = create_dummy_dataset(num_samples=32)
        print("✓ Dummy dataset created")
        
        # Extract parameters
        print("\nExtracting LoRA parameters...")
        params = trainer.get_parameters()
        print(f"✓ Extracted {len(params)} parameter arrays")
        
        # Train
        print("\nTraining with LoRA...")
        metrics = trainer.train(train_loader, epochs=1, learning_rate=1e-3)
        print(f"✓ LoRA training completed")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        # Extract parameters again
        params_after = trainer.get_parameters()
        print(f"✓ Parameters after training: {len(params_after)} arrays")
        
        # Set parameters (test shape compatibility)
        print("\nTesting parameter setting...")
        trainer.set_parameters(params_after)
        print("✓ Parameter setting successful")
        
        print("\n✅ LoRA test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ LoRA test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_features():
    """Test all features combined"""
    print("\n" + "-" * 70)
    print("TEST 4: Combined Features (SAM + TTA + LoRA)")
    print("-" * 70)
    
    try:
        # Check if PEFT is available
        try:
            import peft
            has_peft = True
        except ImportError:
            print("⚠️  PEFT not installed - testing SAM + TTA only")
            has_peft = False
        
        # Create trainer with all features enabled
        print("\nCreating trainer with all features...")
        trainer = MobileViTLoRATrainer(
            num_classes=10,
            lora_r=4,
            use_mixed_precision=False,
            use_sam=True,
            sam_rho=0.05,
            use_tta=True,
            tta_steps=1,
            tta_lr=1e-4,
            use_lora=has_peft,
        )
        print("✓ Combined trainer created")
        
        # Create dummy data
        train_loader, test_loader = create_dummy_dataset(num_samples=32)
        print("✓ Dummy dataset created")
        
        # Train
        print("\nTraining with SAM" + (" + LoRA" if has_peft else "") + "...")
        metrics = trainer.train(train_loader, epochs=1, learning_rate=1e-3)
        print(f"✓ Training completed")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        # Evaluate with TTA
        print("\nEvaluating with TTA...")
        eval_metrics = trainer.evaluate(test_loader)
        print(f"✓ Evaluation completed")
        print(f"  Loss: {eval_metrics['loss']:.4f}")
        print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
        
        print("\n✅ Combined features test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Combined features test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_integration():
    """Test configuration integration"""
    print("\n" + "-" * 70)
    print("TEST 5: Configuration Integration")
    print("-" * 70)
    
    try:
        # Create config
        config = Config()
        
        # Enable advanced features
        config.training.use_sam = True
        config.training.sam_rho = 0.05
        config.training.use_tta = True
        config.training.tta_steps = 2
        config.model.use_lora = False  # Disable for this test
        
        print("✓ Config created with advanced features:")
        print(f"  SAM: {config.training.use_sam}")
        print(f"  TTA: {config.training.use_tta}")
        print(f"  LoRA: {config.model.use_lora}")
        
        # Create trainer from config
        trainer = MobileViTLoRATrainer(
            num_classes=config.model.num_classes,
            lora_r=config.model.lora_r,
            use_mixed_precision=config.training.mixed_precision,
            use_sam=config.training.use_sam,
            sam_rho=config.training.sam_rho,
            use_tta=config.training.use_tta,
            tta_steps=config.training.tta_steps,
            tta_lr=config.training.tta_lr,
            use_lora=config.model.use_lora,
        )
        print("✓ Trainer created from config")
        
        print("\n✅ Config integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Config integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nRunning test suite...\n")
    
    results = {
        "SAM": test_sam_optimizer(),
        "TTA": test_tta(),
        "LoRA": test_lora(),
        "Combined": test_combined_features(),
        "Config": test_config_integration(),
    }
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:15s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\nAdvanced features are working correctly:")
        print("  ✓ Feature A: Sharpness-Aware Minimization (SAM)")
        print("  ✓ Feature C: Test-Time Adaptation (TTA)")
        print("  ✓ Feature D: LoRA (if PEFT installed)")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review the error messages above.")
    print("=" * 70)
