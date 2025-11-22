import pickle
import lz4.frame
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSerializer:
    def __init__(self, enable_quantization=True, compression_level=4):
        self.enable_quantization = enable_quantization
        self.compression_level = compression_level

    def _quantize_array(self, array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-Channel Quantization for Conv weights (Axis 0) or Per-Tensor for others.
        Returns: (quantized_int8, scales_float32)
        """
        if array.dtype != np.float32:
            array = array.astype(np.float32)

        # 1. Per-Channel Quantization (for Conv2d: [Out, In, K, K])
        if array.ndim == 4: 
            # Calculate max abs value per output channel (axis 0)
            # Reshape to (Out, -1) to compute max
            flattened = array.reshape(array.shape[0], -1)
            max_val = np.max(np.abs(flattened), axis=1)
            # Avoid division by zero
            max_val = np.where(max_val == 0, 1.0, max_val)
            
            scales = max_val / 127.0
            
            # Broadcast scale for division: (Out, 1, 1, 1)
            scales_view = scales.reshape(-1, 1, 1, 1)
            quantized = np.round(array / scales_view)
            quantized = np.clip(quantized, -127, 127).astype(np.int8)
            
            return quantized, scales
            
        # 2. Per-Tensor Quantization (for Linear/Bias)
        else:
            max_val = np.max(np.abs(array))
            if max_val == 0:
                scale = 1.0
                quantized = np.zeros_like(array, dtype=np.int8)
            else:
                scale = max_val / 127.0
                quantized = np.round(array / scale)
                quantized = np.clip(quantized, -127, 127).astype(np.int8)
            
            # Wrap scale in array for consistency
            return quantized, np.array([scale], dtype=np.float32)

    def _dequantize_array(self, quantized: np.ndarray, scales: np.ndarray, original_shape: Tuple) -> np.ndarray:
        if len(original_shape) == 4: # Conv2d
             scales_view = scales.reshape(-1, 1, 1, 1)
             return (quantized.astype(np.float32) * scales_view).reshape(original_shape)
        else: # Linear/1D
             return (quantized.astype(np.float32) * scales[0]).reshape(original_shape)

    def serialize_weights(self, weights: List[np.ndarray]) -> bytes:
        serialized_list = []
        for w in weights:
            if self.enable_quantization:
                quant, scales = self._quantize_array(w)
                serialized_list.append({
                    'data': quant,
                    'scales': scales,
                    'shape': w.shape,
                    'q': True
                })
            else:
                serialized_list.append({'data': w, 'q': False})
        
        pickled = pickle.dumps(serialized_list)
        return lz4.frame.compress(pickled, compression_level=self.compression_level)

    def deserialize_weights(self, data: bytes) -> List[np.ndarray]:
        try:
            pickled = lz4.frame.decompress(data)
            serialized_list = pickle.loads(pickled)
            
            weights = []
            for item in serialized_list:
                if item.get('q'):
                    w = self._dequantize_array(item['data'], item['scales'], item['shape'])
                else:
                    w = item['data']
                weights.append(w)
            return weights
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return []
            
    # Metadata serialization methods remain same...
    def serialize_metadata(self, metadata):
        return lz4.frame.compress(pickle.dumps(metadata))

    def deserialize_metadata(self, data):
        return pickle.loads(lz4.frame.decompress(data))


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
        import struct
        if not isinstance(payload, bytes):
            raise TypeError("Payload must be bytes")
        
        # Format: [4-byte length (big-endian)][1-byte type][payload]
        length = len(payload)
        header = struct.pack('>I', length) + struct.pack('B', msg_type)
        return header + payload
    
    @staticmethod
    def decode_message(data: bytes) -> tuple:
        """
        Decode a message, extracting type and payload.
        
        Args:
            data: Encoded message
            
        Returns:
            Tuple of (msg_type, payload)
        """
        import struct
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