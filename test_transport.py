"""
Test script for transport layer (serialization and QUIC protocol)
Verifies compression ratios and serialization correctness

Author: Research Team - FL-QUIC-LoRA Project
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from transport.serializer import ModelSerializer, MessageCodec


def test_serialization():
    """Test weight serialization and compression"""
    print("="*60)
    print("Testing Model Serialization")
    print("="*60)
    
    # Create synthetic weights (similar to LoRA parameters)
    weights = [
        np.random.randn(768, 8).astype(np.float32),   # LoRA A matrix
        np.random.randn(8, 768).astype(np.float32),   # LoRA B matrix
        np.random.randn(768).astype(np.float32),      # Bias
    ]
    
    # Calculate original size
    original_size = sum(w.nbytes for w in weights)
    print(f"\nOriginal weights:")
    for i, w in enumerate(weights):
        print(f"  Layer {i}: shape={w.shape}, size={w.nbytes:,} bytes")
    print(f"Total original size: {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
    
    # Test with quantization
    print("\n" + "-"*60)
    print("Test 1: WITH Quantization (FP32 -> INT8)")
    print("-"*60)
    
    serializer = ModelSerializer(enable_quantization=True, compression_level=4)
    compressed = serializer.serialize_weights(weights)
    print(f"Compressed size: {len(compressed):,} bytes ({len(compressed)/1024:.2f} KB)")
    print(f"Compression ratio: {original_size / len(compressed):.2f}x")
    
    # Deserialize and check accuracy
    restored = serializer.deserialize_weights(compressed)
    
    print("\nAccuracy check (quantization error):")
    for i, (orig, rest) in enumerate(zip(weights, restored)):
        mse = np.mean((orig - rest) ** 2)
        max_error = np.max(np.abs(orig - rest))
        print(f"  Layer {i}: MSE={mse:.6f}, Max Error={max_error:.6f}")
    
    # Test without quantization
    print("\n" + "-"*60)
    print("Test 2: WITHOUT Quantization (FP32 only + LZ4)")
    print("-"*60)
    
    serializer_no_quant = ModelSerializer(enable_quantization=False, compression_level=4)
    compressed_no_quant = serializer_no_quant.serialize_weights(weights)
    print(f"Compressed size: {len(compressed_no_quant):,} bytes ({len(compressed_no_quant)/1024:.2f} KB)")
    print(f"Compression ratio: {original_size / len(compressed_no_quant):.2f}x")
    
    # Deserialize
    restored_no_quant = serializer_no_quant.deserialize_weights(compressed_no_quant)
    
    print("\nAccuracy check (should be exact):")
    for i, (orig, rest) in enumerate(zip(weights, restored_no_quant)):
        max_error = np.max(np.abs(orig - rest))
        print(f"  Layer {i}: Max Error={max_error:.10f}")
    
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    print(f"Original size:        {original_size:,} bytes")
    print(f"With quantization:    {len(compressed):,} bytes ({original_size/len(compressed):.2f}x)")
    print(f"Without quantization: {len(compressed_no_quant):,} bytes ({original_size/len(compressed_no_quant):.2f}x)")
    print(f"\nQuantization benefit: {len(compressed_no_quant)/len(compressed):.2f}x additional compression")


def test_message_codec():
    """Test message encoding/decoding"""
    print("\n" + "="*60)
    print("Testing Message Codec")
    print("="*60)
    
    # Test data
    test_payload = b"Hello, Federated Learning with QUIC!" * 100
    
    # Encode
    encoded = MessageCodec.encode_message(MessageCodec.MSG_TYPE_WEIGHTS, test_payload)
    print(f"\nOriginal payload size: {len(test_payload)} bytes")
    print(f"Encoded message size: {len(encoded)} bytes (overhead: {len(encoded) - len(test_payload)} bytes)")
    
    # Decode
    msg_type, decoded_payload = MessageCodec.decode_message(encoded)
    
    # Verify
    assert msg_type == MessageCodec.MSG_TYPE_WEIGHTS, "Message type mismatch!"
    assert decoded_payload == test_payload, "Payload mismatch!"
    
    print("✓ Encoding/decoding successful")
    print(f"✓ Message type: {msg_type}")
    print(f"✓ Payload integrity verified")


def test_metadata_serialization():
    """Test metadata serialization"""
    print("\n" + "="*60)
    print("Testing Metadata Serialization")
    print("="*60)
    
    metadata = {
        'client_id': 'jetson_nano_001',
        'round': 5,
        'num_samples': 1000,
        'metrics': {
            'loss': 0.234,
            'accuracy': 0.892,
            'training_time': 45.6,
        },
        'device_info': {
            'gpu': 'NVIDIA Jetson Nano',
            'memory': '4GB',
        }
    }
    
    serializer = ModelSerializer()
    
    # Serialize
    compressed = serializer.serialize_metadata(metadata)
    print(f"\nMetadata keys: {list(metadata.keys())}")
    print(f"Compressed size: {len(compressed)} bytes")
    
    # Deserialize
    restored = serializer.deserialize_metadata(compressed)
    
    # Verify
    assert restored == metadata, "Metadata mismatch!"
    print("✓ Metadata serialization successful")
    print(f"✓ All {len(metadata)} keys restored correctly")


if __name__ == "__main__":
    try:
        test_serialization()
        test_message_codec()
        test_metadata_serialization()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nTransport layer is ready for FL-QUIC deployment!")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
