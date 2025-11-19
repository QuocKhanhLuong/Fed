"""
Test Adaptive Parameters Integration
Verify that NetworkMonitor → Pruning → Compression pipeline works correctly
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from transport.network_monitor import NetworkMonitor
from transport.serializer import ModelSerializer
from client.model_trainer import MobileViTLoRATrainer, create_dummy_dataset

def test_network_monitor():
    """Test NetworkMonitor score calculation"""
    print("\n" + "="*60)
    print("TEST 1: Network Monitor Score Calculation")
    print("="*60)
    
    monitor = NetworkMonitor(window_size=5)
    
    # Test case 1: Perfect network
    monitor.update_stats(rtt=0.020, packet_loss=0.0)
    score = monitor.get_network_score()
    print(f"✓ Perfect Network: RTT=20ms, Loss=0% → Score={score:.3f}")
    assert score >= 0.95, "Perfect network should have score >= 0.95"
    
    # Test case 2: Good network
    monitor.update_stats(rtt=0.050, packet_loss=0.01)
    score = monitor.get_network_score()
    print(f"✓ Good Network: RTT=50ms, Loss=1% → Score={score:.3f}")
    assert 0.85 <= score <= 1.0, "Good network should have 0.85 <= score <= 1.0"
    
    # Test case 3: Fair network
    monitor.update_stats(rtt=0.150, packet_loss=0.03)
    score = monitor.get_network_score()
    print(f"✓ Fair Network: RTT=150ms, Loss=3% → Score={score:.3f}")
    assert 0.5 <= score <= 0.85, "Fair network should have 0.5 <= score <= 0.85"
    
    # Test case 4: Poor network
    monitor.update_stats(rtt=0.350, packet_loss=0.07)
    score = monitor.get_network_score()
    print(f"✓ Poor Network: RTT=350ms, Loss=7% → Score={score:.3f}")
    assert 0.3 <= score <= 0.6, "Poor network should have 0.3 <= score <= 0.6"
    
    # Test case 5: Critical network
    monitor.update_stats(rtt=0.480, packet_loss=0.095)
    score = monitor.get_network_score()
    print(f"✓ Critical Network: RTT=480ms, Loss=9.5% → Score={score:.3f}")
    assert score < 0.35, "Critical network should have score < 0.35"
    
    print("✅ Network Monitor tests passed!")


def test_adaptive_pruning():
    """Test adaptive parameter extraction with different network scores"""
    print("\n" + "="*60)
    print("TEST 2: Adaptive Pruning Based on Network Score")
    print("="*60)
    
    # Create trainer without network monitor (baseline)
    print("\n[Baseline] Creating trainer without NetworkMonitor...")
    trainer_baseline = MobileViTLoRATrainer(
        num_classes=10,
        lora_r=4,  # Small rank for fast testing
        use_mixed_precision=False,
        network_monitor=None,
    )
    
    # Get baseline parameters
    baseline_params = trainer_baseline.get_parameters()
    baseline_size = sum(p.size for p in baseline_params)
    print(f"✓ Baseline: {len(baseline_params)} arrays, {baseline_size:,} parameters")
    
    # Test case 1: Good network (no pruning)
    print("\n[Test 1] Good Network (Score ≥ 0.8)...")
    monitor_good = NetworkMonitor()
    monitor_good.update_stats(rtt=0.030, packet_loss=0.005)
    
    trainer_good = MobileViTLoRATrainer(
        num_classes=10,
        lora_r=4,
        use_mixed_precision=False,
        network_monitor=monitor_good,
    )
    
    params_good = trainer_good.get_adaptive_parameters()
    
    # Calculate sparsity from actual pruned params
    total_params_good = sum(p.size for p in params_good)
    zeros_good = sum(np.count_nonzero(p == 0) for p in params_good)
    sparsity_good = zeros_good / total_params_good if total_params_good > 0 else 0
    
    print(f"  Network Score: {monitor_good.get_network_score():.3f}")
    print(f"  Sparsity: {sparsity_good:.1%}")
    print(f"✓ Good network should have low sparsity (no additional pruning)")
    # Note: Real model may have natural zeros from initialization, so we check it's less than poor network
    assert sparsity_good < 0.6, f"Good network should not prune much, got {sparsity_good:.1%}"
    
    # Test case 2: Poor network (medium pruning)
    print("\n[Test 2] Poor Network (Score ~0.3-0.4)...")
    monitor_poor = NetworkMonitor()
    # Use RTT=200ms, Loss=4% to get score ~0.4
    for _ in range(5):
        monitor_poor.update_stats(rtt=0.200, packet_loss=0.04)
    
    trainer_poor = MobileViTLoRATrainer(
        num_classes=10,
        lora_r=4,
        use_mixed_precision=False,
        network_monitor=monitor_poor,
    )
    
    params_poor = trainer_poor.get_adaptive_parameters()
    
    # Calculate sparsity from actual pruned params
    total_params_poor = sum(p.size for p in params_poor)
    zeros_poor = sum(np.count_nonzero(p == 0) for p in params_poor)
    sparsity_poor = zeros_poor / total_params_poor if total_params_poor > 0 else 0
    
    print(f"  Network Score: {monitor_poor.get_network_score():.3f}")
    print(f"  Sparsity: {sparsity_poor:.1%}")
    print(f"✓ Poor network should have higher sparsity than good network")
    # Check that poor network has MORE sparsity than good network
    assert sparsity_poor > sparsity_good, f"Poor network should prune more than good network: {sparsity_poor:.1%} vs {sparsity_good:.1%}"
    
    # Test case 3: Critical network (heavy pruning)
    print("\n[Test 3] Critical Network (Score ~0.1)...")
    monitor_critical = NetworkMonitor()
    # Use RTT=280ms, Loss=4.5% to get very low score
    for _ in range(5):
        monitor_critical.update_stats(rtt=0.280, packet_loss=0.045)
    
    trainer_critical = MobileViTLoRATrainer(
        num_classes=10,
        lora_r=4,
        use_mixed_precision=False,
        network_monitor=monitor_critical,
    )
    
    params_critical = trainer_critical.get_adaptive_parameters()
    
    # Calculate sparsity from actual pruned params
    total_params_critical = sum(p.size for p in params_critical)
    zeros_critical = sum(np.count_nonzero(p == 0) for p in params_critical)
    sparsity_critical = zeros_critical / total_params_critical if total_params_critical > 0 else 0
    
    print(f"  Network Score: {monitor_critical.get_network_score():.3f}")
    print(f"  Sparsity: {sparsity_critical:.1%}")
    print(f"✓ Critical network should have highest sparsity")
    # Check that critical network has MORE sparsity than poor network
    assert sparsity_critical > sparsity_poor, f"Critical network should prune more: {sparsity_critical:.1%} vs {sparsity_poor:.1%}"
    
    print("\n✅ Adaptive Pruning tests passed!")


def test_compression_ratios():
    """Test compression ratios with different sparsity levels"""
    print("\n" + "="*60)
    print("TEST 3: Compression Ratios (Pruning + Quantization + LZ4)")
    print("="*60)
    
    serializer = ModelSerializer(enable_quantization=True, compression_level=4)
    
    # Create dummy weights
    def create_test_weights(size=10000, sparsity=0.0):
        weights = np.random.randn(size).astype(np.float32)
        if sparsity > 0:
            # Zero out random elements
            mask = np.random.rand(size) > sparsity
            weights = weights * mask
        return [weights]
    
    # Test case 1: Dense weights (no sparsity)
    print("\n[Test 1] Dense Weights (0% zeros)...")
    dense_weights = create_test_weights(size=10000, sparsity=0.0)
    compressed_dense = serializer.serialize_weights(dense_weights)
    ratio_dense = (10000 * 4) / len(compressed_dense)
    print(f"  Original: {10000*4:,} bytes")
    print(f"  Compressed: {len(compressed_dense):,} bytes")
    print(f"  Ratio: {ratio_dense:.2f}x")
    
    # Test case 2: 50% sparse
    print("\n[Test 2] Sparse Weights (50% zeros)...")
    sparse_50 = create_test_weights(size=10000, sparsity=0.5)
    compressed_50 = serializer.serialize_weights(sparse_50)
    ratio_50 = (10000 * 4) / len(compressed_50)
    print(f"  Original: {10000*4:,} bytes")
    print(f"  Compressed: {len(compressed_50):,} bytes")
    print(f"  Ratio: {ratio_50:.2f}x")
    print(f"✓ Should be better than dense: {ratio_50:.2f}x > {ratio_dense:.2f}x")
    assert ratio_50 > ratio_dense, "Sparse weights should compress better"
    
    # Test case 3: 90% sparse
    print("\n[Test 3] Highly Sparse Weights (90% zeros)...")
    sparse_90 = create_test_weights(size=10000, sparsity=0.9)
    compressed_90 = serializer.serialize_weights(sparse_90)
    ratio_90 = (10000 * 4) / len(compressed_90)
    print(f"  Original: {10000*4:,} bytes")
    print(f"  Compressed: {len(compressed_90):,} bytes")
    print(f"  Ratio: {ratio_90:.2f}x")
    print(f"✓ Should be much better: {ratio_90:.2f}x >> {ratio_dense:.2f}x")
    assert ratio_90 > ratio_50 * 2, "90% sparse should compress 2x better than 50%"
    
    print("\n✅ Compression tests passed!")
    
    # Summary
    print("\n" + "="*60)
    print("COMPRESSION SUMMARY")
    print("="*60)
    print(f"Dense (0% zeros):  {ratio_dense:.2f}x compression")
    print(f"Sparse 50%:        {ratio_50:.2f}x compression ({ratio_50/ratio_dense:.1f}x better)")
    print(f"Sparse 90%:        {ratio_90:.2f}x compression ({ratio_90/ratio_dense:.1f}x better)")


def test_end_to_end():
    """Test complete pipeline: Monitor → Trainer → Pruning → Compression"""
    print("\n" + "="*60)
    print("TEST 4: End-to-End Integration Test")
    print("="*60)
    
    # Simulate scenario: Network degrades during training
    monitor = NetworkMonitor(window_size=3)
    serializer = ModelSerializer()
    
    scenarios = [
        ("Perfect", 0.020, 0.000),
        ("Good", 0.050, 0.01),
        ("Fair", 0.150, 0.03),
        ("Poor", 0.350, 0.07),
        ("Critical", 0.480, 0.095),
    ]
    
    print("\nSimulating network degradation over 5 rounds:")
    print("-" * 60)
    
    results = []
    
    for round_num, (name, rtt, loss) in enumerate(scenarios, 1):
        # Update network stats
        monitor.update_stats(rtt, loss)
        score = monitor.get_network_score()
        
        # Create trainer with current network monitor
        trainer = MobileViTLoRATrainer(
            num_classes=10,
            lora_r=4,
            use_mixed_precision=False,
            network_monitor=monitor,
        )
        
        # Get adaptive parameters
        params = trainer.get_adaptive_parameters()
        
        # Calculate sparsity
        total_params = sum(p.size for p in params)
        zero_params = sum(np.count_nonzero(p == 0) for p in params)
        sparsity = zero_params / total_params if total_params > 0 else 0
        
        # Compress
        compressed = serializer.serialize_weights(params)
        original_size = sum(p.nbytes for p in params)
        compression_ratio = original_size / len(compressed)
        
        results.append({
            'round': round_num,
            'name': name,
            'rtt': rtt * 1000,
            'loss': loss * 100,
            'score': score,
            'sparsity': sparsity * 100,
            'original': original_size,
            'compressed': len(compressed),
            'ratio': compression_ratio,
        })
        
        print(f"Round {round_num} [{name:8s}] "
              f"RTT={rtt*1000:4.0f}ms Loss={loss*100:3.1f}% → "
              f"Score={score:.3f} → "
              f"Sparsity={sparsity*100:4.1f}% → "
              f"Size={len(compressed):5,}B ({compression_ratio:.1f}x)")
    
    print("\n✅ End-to-End test completed!")
    
    # Verify trends
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Score should decrease
    scores = [r['score'] for r in results]
    print(f"✓ Network scores: {' → '.join(f'{s:.2f}' for s in scores)}")
    assert scores[0] > scores[-1], "Score should degrade"
    
    # Sparsity should increase
    sparsities = [r['sparsity'] for r in results]
    print(f"✓ Sparsity levels: {' → '.join(f'{s:.0f}%' for s in sparsities)}")
    assert sparsities[-1] > sparsities[0], "Sparsity should increase"
    
    # Compression should improve
    ratios = [r['ratio'] for r in results]
    print(f"✓ Compression: {' → '.join(f'{r:.1f}x' for r in ratios)}")
    assert ratios[-1] > ratios[0], "Compression should improve"
    
    print("\n✅ All verifications passed!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ADAPTIVE PARAMETERS INTEGRATION TEST SUITE")
    print("="*60)
    
    try:
        test_network_monitor()
        test_adaptive_pruning()
        test_compression_ratios()
        test_end_to_end()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\n✅ NetworkMonitor working correctly")
        print("✅ Adaptive pruning based on network score")
        print("✅ Compression ratios improve with sparsity")
        print("✅ End-to-end pipeline functioning properly")
        print("\n🚀 System ready for deployment!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
