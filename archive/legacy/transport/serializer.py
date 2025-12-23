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

    def _sanitize_array(self, array: np.ndarray) -> np.ndarray:
        """Replace NaN/Inf with 0 to prevent quantization errors."""
        if not np.isfinite(array).all():
            num_bad = np.sum(~np.isfinite(array))
            logger.warning(f"Sanitizing {num_bad} NaN/Inf values in array shape {array.shape}")
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        return array

    def _quantize_array(self, array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Sanitize NaN/Inf first
        array = self._sanitize_array(array)
        if array.dtype != np.float32: 
            array = array.astype(np.float32)

        if array.ndim == 4: 
            flattened = array.reshape(array.shape[0], -1)
            max_val = np.max(np.abs(flattened), axis=1)
            max_val = np.where(max_val == 0, 1.0, max_val)
            scales = max_val / 127.0
            scales_view = scales.reshape(-1, 1, 1, 1)
            quantized = np.clip(np.round(array / scales_view), -127, 127).astype(np.int8)
            return quantized, scales
        else:
            max_val = np.max(np.abs(array))
            scale = 1.0 if max_val == 0 else max_val / 127.0
            quantized = np.clip(np.round(array / scale), -127, 127).astype(np.int8)
            return quantized, np.array([scale], dtype=np.float32)

    def _dequantize_array(self, quantized: np.ndarray, scales: np.ndarray, original_shape: Tuple) -> np.ndarray:
        if len(original_shape) == 4:
             scales_view = scales.reshape(-1, 1, 1, 1)
             return (quantized.astype(np.float32) * scales_view).reshape(original_shape)
        else:
             return (quantized.astype(np.float32) * scales[0]).reshape(original_shape)

    def serialize_weights(self, weights: List[np.ndarray]) -> bytes:
        serialized_list = []
        for i, w in enumerate(weights):
            if self.enable_quantization:
                quant, scales = self._quantize_array(w)
                serialized_list.append({'data': quant, 'scales': scales, 'shape': w.shape, 'q': True})
            else:
                # Still sanitize even without quantization
                w = self._sanitize_array(w)
                serialized_list.append({'data': w, 'q': False})
        return lz4.frame.compress(pickle.dumps(serialized_list), compression_level=self.compression_level)

    def deserialize_weights(self, data: bytes) -> List[np.ndarray]:
        try:
            serialized_list = pickle.loads(lz4.frame.decompress(data))
            weights = []
            for item in serialized_list:
                if item.get('q'):
                    weights.append(self._dequantize_array(item['data'], item['scales'], item['shape']))
                else:
                    weights.append(item['data'])
            return weights
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return []

    def serialize_metadata(self, metadata):
        return lz4.frame.compress(pickle.dumps(metadata))

    def deserialize_metadata(self, data):
        return pickle.loads(lz4.frame.decompress(data))


class MessageCodec:
    MSG_TYPE_WEIGHTS = 0x01
    MSG_TYPE_METADATA = 0x02
    MSG_TYPE_CONFIG = 0x03
    MSG_TYPE_ACK = 0x04
    MSG_TYPE_ERROR = 0xFF
    
    @staticmethod
    def encode_message(msg_type: int, payload: bytes) -> bytes:
        import struct
        if not isinstance(payload, bytes): raise TypeError("Payload must be bytes")
        header = struct.pack('>I', len(payload)) + struct.pack('B', msg_type)
        return header + payload
    
    @staticmethod
    def decode_message(data: bytes) -> tuple:
        import struct
        if len(data) < 5: raise ValueError("Invalid message")
        length = struct.unpack('>I', data[0:4])[0]
        msg_type = struct.unpack('B', data[4:5])[0]
        payload = data[5:]
        if len(payload) != length: raise ValueError("Length mismatch")
        return msg_type, payload