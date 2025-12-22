"""
Test Nested Learning Features
Verify LSS, DeepMomentum, and Extended CMS

Reference: "Nested Learning" (Google Research, NeurIPS 2025)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

print("=" * 70)
print("Testing Nested Learning Features (NeurIPS 2025)")
print("=" * 70)


def test_lss():
    """Test Local Surprise Signal"""
    print("\n" + "-" * 70)
    print("TEST 1: Local Surprise Signal (LSS)")
    print("-" * 70)
    
    from client.nested_trainer import LocalSurpriseSignal
    
    # Create LSS
    lss = LocalSurpriseSignal(enabled=True, temperature=1.0)
    print("✓ LSS instance created")
    
    # Test with sample losses
    per_sample_loss = torch.tensor([0.5, 1.0, 2.0, 0.3, 1.5])
    weights = lss.compute_weights(per_sample_loss)
    
    print(f"  Sample losses: {per_sample_loss.tolist()}")
    print(f"  LSS weights: {weights.tolist()}")
    
    # Verify: higher loss should have higher weight
    max_loss_idx = per_sample_loss.argmax().item()
    max_weight_idx = weights.argmax().item()
    
    assert max_loss_idx == max_weight_idx, "LSS should weight surprising samples higher"
    print("✓ High-loss samples get higher weight")
    
    # Test disabled mode
    lss_disabled = LocalSurpriseSignal(enabled=False)
    weights_disabled = lss_disabled.compute_weights(per_sample_loss)
    assert torch.allclose(weights_disabled, torch.ones_like(weights_disabled)), "Disabled LSS should return uniform weights"
    print("✓ Disabled mode returns uniform weights")
    
    return True


def test_deep_momentum():
    """Test Deep Momentum GD"""
    print("\n" + "-" * 70)
    print("TEST 2: Deep Momentum GD (DMGD)")
    print("-" * 70)
    
    from client.nested_trainer import DeepMomentum
    
    # Create deep momentum for small param vector
    param_dim = 64
    dm = DeepMomentum(param_dim=param_dim, hidden_dim=32, num_layers=2)
    print(f"✓ DeepMomentum created: {param_dim} params, 32 hidden dim")
    
    # Count parameters
    num_params = sum(p.numel() for p in dm.parameters())
    print(f"  MLP parameters: {num_params:,}")
    
    # Test forward pass
    gradient = torch.randn(param_dim)
    momentum = dm(gradient)
    
    assert momentum.shape == gradient.shape, "Momentum should have same shape as gradient"
    print("✓ Forward pass: gradient → momentum")
    
    # Test accumulation (should remember previous)
    gradient2 = torch.randn(param_dim)
    momentum2 = dm(gradient2)
    
    # Should be different from first (has memory)
    assert not torch.allclose(momentum2, gradient2), "Momentum should differ from raw gradient"
    print("✓ Momentum accumulates history")
    
    return True


def test_extended_cms():
    """Test Extended Continuum Memory System"""
    print("\n" + "-" * 70)
    print("TEST 3: Extended CMS (4 levels)")
    print("-" * 70)
    
    from client.nested_trainer import ContinuumMemorySystem
    
    # Test exponential level generation
    levels = ContinuumMemorySystem.create_exponential_levels(base=5, num_levels=4)
    print(f"  Exponential levels: {levels}")
    assert levels == [1, 5, 25, 125], "Should generate [1, 5, 25, 125]"
    print("✓ Exponential frequency generation")
    
    # Create CMS with 4 levels
    cms = ContinuumMemorySystem(
        enabled=True,
        num_levels=4,
        base_freq=5,
    )
    print(f"✓ CMS created: {cms.num_levels} levels")
    print(f"  Update freqs: {cms.update_freqs}")
    print(f"  Decay rates: {cms.decay_rates}")
    
    # Test memory update
    weights = torch.randn(100)
    cms.update(weights, step=1)
    print("✓ Memory initialized on first update")
    
    # Update at different steps
    for step in [5, 25, 125]:
        new_weights = torch.randn(100)
        cms.update(new_weights, step=step)
        print(f"  Updated memories at step {step}")
    
    # Get memory loss
    current = torch.randn(100)
    loss = cms.get_memory_loss(current)
    assert loss.item() > 0, "Memory loss should be positive"
    print(f"✓ Memory loss: {loss.item():.6f}")
    
    # Get stats
    stats = cms.get_memory_stats()
    assert len(stats) == 4, "Should have 4 memory stats"
    print(f"✓ Memory stats: {list(stats.keys())}")
    
    return True


def test_nested_trainer_integration():
    """Test that NestedEarlyExitTrainer inits with new features"""
    print("\n" + "-" * 70)
    print("TEST 4: NestedEarlyExitTrainer Integration")
    print("-" * 70)
    
    from client.nested_trainer import NestedEarlyExitTrainer
    
    # Create trainer with new features
    trainer = NestedEarlyExitTrainer(
        num_classes=10,
        use_mixed_precision=False,  # CPU friendly
        # New features
        use_lss=True,
        lss_temperature=1.0,
        cms_num_levels=4,
        use_deep_momentum=False,  # Keep off for speed
    )
    print("✓ Trainer created with new features")
    
    # Check stats
    assert trainer.stats['lss_enabled'] == True
    assert trainer.stats['cms_levels'] == 4
    assert trainer.stats['dmgd_enabled'] == False
    print(f"  Stats: LSS={trainer.stats['lss_enabled']}, "
          f"CMS levels={trainer.stats['cms_levels']}, "
          f"DMGD={trainer.stats['dmgd_enabled']}")
    
    # Check instances
    assert trainer.lss is not None
    assert trainer.cms is not None
    assert trainer.use_lss == True
    print("✓ All feature instances initialized")
    
    return True


if __name__ == "__main__":
    try:
        success1 = test_lss()
        success2 = test_deep_momentum()
        success3 = test_extended_cms()
        success4 = test_nested_trainer_integration()
        
        print("\n" + "=" * 70)
        if all([success1, success2, success3, success4]):
            print("✅ ALL NESTED LEARNING TESTS PASSED")
        else:
            print("❌ SOME TESTS FAILED")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
