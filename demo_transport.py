#!/usr/bin/env python3
"""
Quick Demo: Transport Layer Functionality
Demonstrates serialization and message encoding without QUIC dependencies

Author: Research Team - FL-QUIC-LoRA Project
"""

import sys
import numpy as np
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

print("="*70)
print("FL-QUIC-LoRA Transport Layer Demo")
print("="*70)

print("\n📦 Importing transport modules...")
try:
    from transport.serializer import ModelSerializer
    print("✅ Import successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("\nPlease install dependencies first:")
    print("  pip install numpy lz4")
    sys.exit(1)

# Create sample LoRA weights
print("\n" + "-"*70)
print("1. Creating Sample LoRA Weights")
print("-"*70)

# Typical LoRA configuration: rank = 8
lora_weights = {
    'layer1.A': np.random.randn(768, 8).astype(np.float32),
    'layer1.B': np.random.randn(8, 768).astype(np.float32),
    'layer2.A': np.random.randn(768, 8).astype(np.float32),
    'layer2.B': np.random.randn(8, 768).astype(np.float32),
}

weights_list = list(lora_weights.values())
total_params = sum(w.size for w in weights_list)
total_size = sum(w.nbytes for w in weights_list)

print(f"LoRA Configuration:")
print(f"  Rank: 8")
print(f"  Layers: {len(lora_weights)}")
print(f"  Total parameters: {total_params:,}")
print(f"  Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

# Test serialization WITH quantization
print("\n" + "-"*70)
print("2. Testing Compression WITH Quantization")
print("-"*70)

serializer_quant = ModelSerializer(enable_quantization=True, compression_level=4)
compressed_quant = serializer_quant.serialize_weights(weights_list)

ratio_quant = total_size / len(compressed_quant)
print(f"Compressed size: {len(compressed_quant):,} bytes ({len(compressed_quant)/1024:.2f} KB)")
print(f"Compression ratio: {ratio_quant:.2f}x")
print(f"Bandwidth saved: {(1 - 1/ratio_quant)*100:.1f}%")

# Deserialize and check accuracy
restored_quant = serializer_quant.deserialize_weights(compressed_quant)
mse = np.mean([(w1 - w2)**2 for w1, w2 in zip(weights_list, restored_quant)])
print(f"Reconstruction MSE: {mse:.8f}")

# Test serialization WITHOUT quantization
print("\n" + "-"*70)
print("3. Testing Compression WITHOUT Quantization (baseline)")
print("-"*70)

serializer_no_quant = ModelSerializer(enable_quantization=False, compression_level=4)
compressed_no_quant = serializer_no_quant.serialize_weights(weights_list)

ratio_no_quant = total_size / len(compressed_no_quant)
print(f"Compressed size: {len(compressed_no_quant):,} bytes ({len(compressed_no_quant)/1024:.2f} KB)")
print(f"Compression ratio: {ratio_no_quant:.2f}x")
print(f"Bandwidth saved: {(1 - 1/ratio_no_quant)*100:.1f}%")

# Message encoding removed - MessageCodec was simplified
print("\n" + "-"*70)
print("4. Comparing Quantization vs No Quantization")
print("-"*70)

improvement = ratio_quant / ratio_no_quant
print(f"Quantization improvement: {improvement:.2f}x better compression")
print(f"Extra bandwidth saved: {(1 - 1/improvement)*100:.1f}%")

# Summary
print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"\nOriginal size: {total_size/1024:.2f} KB")
print(f"With Quantization: {len(compressed_quant)/1024:.2f} KB ({ratio_quant:.2f}x)")
print(f"Without Quantization: {len(compressed_no_quant)/1024:.2f} KB ({ratio_no_quant:.2f}x)")
print(f"\n✅ Per-channel quantization + LZ4 achieves {improvement:.2f}x better compression!")

# Summary
print("\n" + "="*70)
print("Summary: Compression Benefits")
print("="*70)

print(f"\n📊 Compression Comparison:")
print(f"  Original size:           {total_size:>10,} bytes ({total_size/1024/1024:>6.2f} MB)")
print(f"  With quantization:       {len(compressed_quant):>10,} bytes ({len(compressed_quant)/1024:>6.2f} KB) - {ratio_quant:.1f}x")
print(f"  Without quantization:    {len(compressed_no_quant):>10,} bytes ({len(compressed_no_quant)/1024:>6.2f} KB) - {ratio_no_quant:.1f}x")

quant_benefit = len(compressed_no_quant) / len(compressed_quant)
print(f"\n🎯 Quantization provides {quant_benefit:.2f}x additional compression!")

# Bandwidth calculation
print(f"\n📡 Bandwidth Impact (per round, per client):")
print(f"  Traditional (FP32):      {total_size/1024/1024:.2f} MB")
print(f"  Our approach:            {len(compressed_quant)/1024/1024:.2f} MB")
print(f"  Savings:                 {(total_size - len(compressed_quant))/1024/1024:.2f} MB ({(1-1/ratio_quant)*100:.1f}%)")

# For 10 rounds, 10 clients
num_rounds = 10
num_clients = 10
total_traditional = total_size * num_rounds * num_clients / 1024 / 1024 / 1024
total_optimized = len(compressed_quant) * num_rounds * num_clients / 1024 / 1024 / 1024

print(f"\n📈 Full Training Session ({num_rounds} rounds, {num_clients} clients):")
print(f"  Traditional:             {total_traditional:.2f} GB")
print(f"  Our approach:            {total_optimized:.2f} GB")
print(f"  Total savings:           {total_traditional - total_optimized:.2f} GB")

print("\n" + "="*70)
print("✅ Transport Layer Demo Complete!")
print("="*70)
print("\nNext steps:")
print("  1. Run full test suite: python test_transport.py")
print("  2. Install all dependencies: bash setup.sh")
print("  3. Review implementation: cat IMPLEMENTATION_SUMMARY.md")
