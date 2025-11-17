"""
Custom Serialization Module for Federated Learning over QUIC
Optimized for Edge Devices and Unstable Networks

Features:
- Quantization: FP32 -> INT8 (4x compression)
- LZ4 Compression: Fast compression optimized for low-power devices
- Robust Error Handling: Network resilience
- Metadata Support: For proper deserialization

Author: Research Team - FL-QUIC-LoRA Project
"""

import pickle
import lz4.frame
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import struct
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelSerializer:
    """
    Handles serialization/deserialization of model weights (NumPy arrays).
    Optimized for bandwidth-constrained edge devices.
    """
    
    # Quantization configuration
    QUANTIZATION_ENABLED = True
    QUANTIZATION_DTYPE = np.int8
    QUANTIZATION_RANGE = 127  # For int8: -127 to 127
    
    # Compression configuration
    COMPRESSION_LEVEL = 4  # LZ4 level (0-16, higher = better compression but slower)
    
    def __init__(self, enable_quantization: bool = True, compression_level: int = 4):
        """
        Initialize the serializer.
        
        Args:
            enable_quantization: Enable FP32->INT8 quantization
            compression_level: LZ4 compression level (0-16)
        """
        self.enable_quantization = enable_quantization
        self.compression_level = compression_level
        
        logger.info(f"ModelSerializer initialized: quantization={enable_quantization}, "
                   f"compression_level={compression_level}")
    
    def _quantize_array(self, array: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Quantize FP32 array to INT8 for 4x compression.
        
        Args:
            array: Input array (float32)
            
        Returns:
            Tuple of (quantized_array, scale, zero_point)
        """
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        
        # Compute scale and zero point for quantization
        min_val = float(array.min())
        max_val = float(array.max())
        
        # Handle edge case: all values are the same
        if min_val == max_val:
            scale = 1.0
            zero_point = 0.0
            quantized = np.zeros_like(array, dtype=self.QUANTIZATION_DTYPE)
        else:
            # Symmetric quantization around zero
            abs_max = max(abs(min_val), abs(max_val))
            scale = abs_max / self.QUANTIZATION_RANGE
            zero_point = 0.0
            
            # Quantize: Q = clip(round(x / scale), -127, 127)
            quantized = np.clip(
                np.round(array / scale),
                -self.QUANTIZATION_RANGE,
                self.QUANTIZATION_RANGE
            ).astype(self.QUANTIZATION_DTYPE)
        
        return quantized, scale, zero_point
    
    def _dequantize_array(self, quantized: np.ndarray, scale: float, 
                         zero_point: float, original_shape: Tuple) -> np.ndarray:
        """
        Dequantize INT8 array back to FP32.
        
        Args:
            quantized: Quantized array (int8)
            scale: Scale factor from quantization
            zero_point: Zero point from quantization
            original_shape: Original array shape
            
        Returns:
            Dequantized array (float32)
        """
        # Dequantize: x = Q * scale + zero_point
        dequantized = quantized.astype(np.float32) * scale + zero_point
        return dequantized.reshape(original_shape)
    
    def serialize_weights(self, weights: List[np.ndarray]) -> bytes:
        """
        Serialize a list of NumPy arrays (model weights).
        
        Process:
        1. Quantize each array (FP32 -> INT8) if enabled
        2. Pickle the quantized data + metadata
        3. LZ4 compress the pickled data
        
        Args:
            weights: List of NumPy arrays (model weights)
            
        Returns:
            Compressed bytes ready for QUIC transmission
        """
        try:
            if not weights:
                raise ValueError("Empty weights list provided")
            
            original_size = sum(w.nbytes for w in weights)
            
            # Prepare data for serialization
            if self.enable_quantization:
                serialization_data = []
                for idx, weight in enumerate(weights):
                    try:
                        quantized, scale, zero_point = self._quantize_array(weight)
                        serialization_data.append({
                            'data': quantized,
                            'scale': scale,
                            'zero_point': zero_point,
                            'shape': weight.shape,
                            'dtype': str(weight.dtype),
                            'quantized': True
                        })
                    except Exception as e:
                        logger.error(f"Quantization failed for weight {idx}: {e}")
                        # Fallback: store without quantization
                        serialization_data.append({
                            'data': weight,
                            'shape': weight.shape,
                            'dtype': str(weight.dtype),
                            'quantized': False
                        })
            else:
                # No quantization - direct serialization
                serialization_data = [{
                    'data': weight,
                    'shape': weight.shape,
                    'dtype': str(weight.dtype),
                    'quantized': False
                } for weight in weights]
            
            # Pickle the data
            pickled_data = pickle.dumps(serialization_data, protocol=pickle.HIGHEST_PROTOCOL)
            
            # LZ4 compression
            compressed_data = lz4.frame.compress(
                pickled_data,
                compression_level=self.compression_level,
                block_size=lz4.frame.BLOCKSIZE_MAX1MB
            )
            
            compressed_size = len(compressed_data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            logger.info(f"Serialization complete: {original_size:,} bytes -> {compressed_size:,} bytes "
                       f"(ratio: {compression_ratio:.2f}x)")
            
            return compressed_data
            
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            raise RuntimeError(f"Failed to serialize weights: {e}")
    
    def deserialize_weights(self, data: bytes) -> List[np.ndarray]:
        """
        Deserialize compressed bytes back to NumPy arrays.
        
        Process:
        1. LZ4 decompress
        2. Unpickle the data
        3. Dequantize if needed (INT8 -> FP32)
        
        Args:
            data: Compressed bytes from QUIC stream
            
        Returns:
            List of NumPy arrays (model weights)
        """
        try:
            if not data:
                raise ValueError("Empty data provided for deserialization")
            
            # LZ4 decompression
            try:
                decompressed_data = lz4.frame.decompress(data)
            except Exception as e:
                logger.error(f"LZ4 decompression failed: {e}")
                raise RuntimeError(f"Decompression failed: {e}")
            
            # Unpickle
            try:
                serialization_data = pickle.loads(decompressed_data)
            except Exception as e:
                logger.error(f"Unpickling failed: {e}")
                raise RuntimeError(f"Deserialization failed: {e}")
            
            # Reconstruct weights
            weights = []
            for idx, item in enumerate(serialization_data):
                try:
                    if item['quantized']:
                        # Dequantize
                        weight = self._dequantize_array(
                            item['data'],
                            item['scale'],
                            item['zero_point'],
                            item['shape']
                        )
                    else:
                        # Already in original format
                        weight = item['data']
                        if isinstance(weight, np.ndarray):
                            weight = weight.reshape(item['shape'])
                    
                    weights.append(weight)
                    
                except Exception as e:
                    logger.error(f"Failed to reconstruct weight {idx}: {e}")
                    raise RuntimeError(f"Weight reconstruction failed at index {idx}: {e}")
            
            logger.info(f"Deserialization complete: {len(weights)} weight arrays restored")
            return weights
            
        except Exception as e:
            logger.error(f"Deserialization failed: {e}")
            raise RuntimeError(f"Failed to deserialize weights: {e}")
    
    def serialize_metadata(self, metadata: Dict[str, Any]) -> bytes:
        """
        Serialize metadata (metrics, config, etc.) for transmission.
        
        Args:
            metadata: Dictionary containing metadata
            
        Returns:
            Compressed bytes
        """
        try:
            pickled = pickle.dumps(metadata, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = lz4.frame.compress(pickled, compression_level=self.compression_level)
            logger.debug(f"Metadata serialized: {len(compressed)} bytes")
            return compressed
        except Exception as e:
            logger.error(f"Metadata serialization failed: {e}")
            raise RuntimeError(f"Failed to serialize metadata: {e}")
    
    def deserialize_metadata(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize metadata from compressed bytes.
        
        Args:
            data: Compressed bytes
            
        Returns:
            Dictionary containing metadata
        """
        try:
            decompressed = lz4.frame.decompress(data)
            metadata = pickle.loads(decompressed)
            logger.debug(f"Metadata deserialized: {len(metadata)} keys")
            return metadata
        except Exception as e:
            logger.error(f"Metadata deserialization failed: {e}")
            raise RuntimeError(f"Failed to deserialize metadata: {e}")


class MessageCodec:
    """
    Handles message framing for QUIC streams.
    Format: [4-byte length][1-byte type][payload]
    """
    
    # Message types
    MSG_TYPE_WEIGHTS = 0x01
    MSG_TYPE_METADATA = 0x02
    MSG_TYPE_CONFIG = 0x03
    MSG_TYPE_ACK = 0x04
    MSG_TYPE_ERROR = 0xFF
    
    @staticmethod
    def encode_message(msg_type: int, payload: bytes) -> bytes:
        """
        Encode a message with type and length prefix.
        
        Args:
            msg_type: Message type (use MSG_TYPE_* constants)
            payload: Message payload
            
        Returns:
            Encoded message with header
        """
        if not isinstance(payload, bytes):
            raise TypeError("Payload must be bytes")
        
        # Format: [4-byte length (big-endian)][1-byte type][payload]
        length = len(payload)
        header = struct.pack('>I', length) + struct.pack('B', msg_type)
        return header + payload
    
    @staticmethod
    def decode_message(data: bytes) -> Tuple[int, bytes]:
        """
        Decode a message, extracting type and payload.
        
        Args:
            data: Encoded message
            
        Returns:
            Tuple of (msg_type, payload)
        """
        if len(data) < 5:
            raise ValueError(f"Invalid message: too short ({len(data)} bytes)")
        
        # Extract header
        length = struct.unpack('>I', data[0:4])[0]
        msg_type = struct.unpack('B', data[4:5])[0]
        
        # Extract payload
        payload = data[5:]
        
        if len(payload) != length:
            raise ValueError(f"Length mismatch: expected {length}, got {len(payload)}")
        
        return msg_type, payload


# Singleton instance for easy import
default_serializer = ModelSerializer()
