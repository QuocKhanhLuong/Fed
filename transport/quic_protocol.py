import asyncio
import logging
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass
from enum import IntEnum

from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import (
    QuicEvent, StreamDataReceived, ConnectionTerminated, HandshakeCompleted, StreamReset
)
from aioquic.quic.connection import QuicConnection
from .serializer import ModelSerializer, MessageCodec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamType(IntEnum):
    CONTROL = 0
    WEIGHTS = 4
    METADATA = 8
    ERROR = 12

@dataclass
class FLMessage:
    stream_type: StreamType
    msg_type: int
    payload: bytes
    metadata: Optional[Dict[str, Any]] = None

class FLQuicProtocol(QuicConnectionProtocol):
    def __init__(self, quic: QuicConnection, stream_handler: Optional[Callable] = None):
        super().__init__(quic)
        self._quic = quic
        self._stream_handler = stream_handler
        self._serializer = ModelSerializer()
        self._codec = MessageCodec()
        
        self._active_streams: Dict[int, StreamType] = {}
        self._receive_buffers: Dict[int, bytearray] = {}
        
        self._handshake_completed = False
        self._connection_ready = asyncio.Event()
        
        self._stats = {
            'bytes_sent': 0, 'bytes_received': 0, 'streams_created': 0,
            'messages_sent': 0, 'messages_received': 0,
        }
    
    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, HandshakeCompleted):
            self._handle_handshake_completed(event)
        elif isinstance(event, StreamDataReceived):
            self._handle_stream_data(event)
        elif isinstance(event, ConnectionTerminated):
            self._handle_connection_terminated(event)
        elif isinstance(event, StreamReset):
            self._handle_stream_reset(event)
    
    def _handle_handshake_completed(self, event: HandshakeCompleted) -> None:
        self._handshake_completed = True
        self._connection_ready.set()
        logger.info(f"Handshake completed - 0-RTT: {event.early_data_accepted}")
    
    def _handle_stream_data(self, event: StreamDataReceived) -> None:
        stream_id, data, end_stream = event.stream_id, event.data, event.end_stream
        self._stats['bytes_received'] += len(data)
        
        MAX_BUFFER_SIZE = 50 * 1024 * 1024
        if stream_id not in self._receive_buffers:
            self._receive_buffers[stream_id] = bytearray()
        
        if len(self._receive_buffers[stream_id]) + len(data) > MAX_BUFFER_SIZE:
            logger.warning(f"Buffer overflow stream {stream_id}")
            return
        
        self._receive_buffers[stream_id].extend(data)
        
        # CRITICAL FIX: Process messages immediately as they arrive,
        # not just when stream ends. This handles end_stream=False case.
        self._process_stream_buffer(stream_id)
        
        # Cleanup if stream is closing
        if end_stream:
            if stream_id in self._receive_buffers:
                del self._receive_buffers[stream_id]
            if stream_id in self._active_streams:
                del self._active_streams[stream_id]
    
    def _process_stream_buffer(self, stream_id: int) -> None:
        try:
            buffer = self._receive_buffers[stream_id]
            while len(buffer) > 0:
                if len(buffer) < 5: break
                import struct
                length = struct.unpack('>I', buffer[0:4])[0]
                if len(buffer) < 5 + length: break
                
                msg_data = buffer[:5+length]
                msg_type, payload = self._codec.decode_message(msg_data)
                self._stats['messages_received'] += 1
                
                if self._stream_handler:
                    asyncio.create_task(self._stream_handler(stream_id, msg_type, payload))
                
                del buffer[:5+length]
        except Exception as e:
            logger.error(f"Stream {stream_id} error: {e}")
            self._send_error_message(stream_id, str(e))
    
    def _handle_connection_terminated(self, event: ConnectionTerminated) -> None:
        logger.warning(f"Terminated: {event.reason_phrase}")
        self._connection_ready.clear()
    
    def _handle_stream_reset(self, event: StreamReset) -> None:
        if event.stream_id in self._receive_buffers: del self._receive_buffers[event.stream_id]
        if event.stream_id in self._active_streams: del self._active_streams[event.stream_id]
    
    async def wait_for_connection(self, timeout: float = 10.0) -> bool:
        try:
            await asyncio.wait_for(self._connection_ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
    
    def create_stream(self, stream_type: StreamType) -> int:
        stream_id = self._quic.get_next_available_stream_id()
        self._active_streams[stream_id] = stream_type
        self._stats['streams_created'] += 1
        return stream_id
    
    def _get_existing_stream(self, stream_type: StreamType) -> Optional[int]:
        for stream_id, stype in self._active_streams.items():
            if stype == stream_type: return stream_id
        return None

    def send_message(self, stream_id: int, msg_type: int, payload: bytes, end_stream: bool = True) -> None:
        try:
            encoded = self._codec.encode_message(msg_type, payload)
            self._quic.send_stream_data(stream_id, encoded, end_stream=end_stream)
            self._stats['bytes_sent'] += len(encoded)
            self._stats['messages_sent'] += 1
            self.transmit()
        except Exception as e:
            logger.error(f"Send failed: {e}")
            raise RuntimeError(f"Send failed: {e}")
    
    def send_weights(self, weights: List, stream_id: Optional[int] = None) -> int:
        payload = self._serializer.serialize_weights(weights)
        if stream_id is None:
            stream_id = self._get_existing_stream(StreamType.WEIGHTS) or self.create_stream(StreamType.WEIGHTS)
        self.send_message(stream_id, MessageCodec.MSG_TYPE_WEIGHTS, payload, end_stream=False)
        return stream_id
    
    def send_metadata(self, metadata: Dict[str, Any], stream_id: Optional[int] = None) -> int:
        payload = self._serializer.serialize_metadata(metadata)
        if stream_id is None:
            stream_id = self._get_existing_stream(StreamType.METADATA) or self.create_stream(StreamType.METADATA)
        self.send_message(stream_id, MessageCodec.MSG_TYPE_METADATA, payload, end_stream=False)
        return stream_id
    
    def _send_error_message(self, stream_id: int, error_msg: str) -> None:
        try:
            self.send_message(self.create_stream(StreamType.ERROR), MessageCodec.MSG_TYPE_ERROR, error_msg.encode('utf-8'))
        except: pass
    
    def get_stats(self) -> Dict[str, Any]:
        return {**self._stats, 'active_streams': len(self._active_streams), 'connected': self._handshake_completed}
    
    def close(self, error_code: int = 0, reason_phrase: str = "Normal close") -> None:
        self._quic.close(error_code=error_code, reason_phrase=reason_phrase)
        self.transmit()

class FLMessageHandler:
    def __init__(self, serializer: Optional[ModelSerializer] = None):
        self.serializer = serializer or ModelSerializer()
    
    async def handle_message(self, stream_id: int, msg_type: int, payload: bytes) -> None:
        if msg_type == MessageCodec.MSG_TYPE_WEIGHTS:
            await self.handle_weights(stream_id, payload)
        elif msg_type == MessageCodec.MSG_TYPE_METADATA:
            await self.handle_metadata(stream_id, payload)
        elif msg_type == MessageCodec.MSG_TYPE_CONFIG:
            await self.handle_config(stream_id, payload)
        elif msg_type == MessageCodec.MSG_TYPE_ERROR:
            await self.handle_error(stream_id, payload)
    
    async def handle_weights(self, stream_id: int, payload: bytes) -> None:
        weights = self.serializer.deserialize_weights(payload)
        logger.info(f"Received {len(weights)} weight arrays")
    
    async def handle_metadata(self, stream_id: int, payload: bytes) -> None:
        logger.info(f"Received metadata")
    
    async def handle_config(self, stream_id: int, payload: bytes) -> None:
        logger.info(f"Received config")
    
    async def handle_error(self, stream_id: int, payload: bytes) -> None:
        logger.error(f"Error from stream {stream_id}: {payload.decode('utf-8')}")

class SessionTicketHandler:
    def __init__(self):
        self.tickets: Dict[str, Any] = {}
    def save_ticket(self, ticket: Any) -> None:
        self.tickets['latest'] = ticket
    def load_ticket(self, key: str = 'latest') -> Optional[Any]:
        return self.tickets.get(key)

ticket_handler = SessionTicketHandler()

def create_quic_config(is_client: bool, cert_file=None, key_file=None, verify_mode=None, enable_0rtt=True, max_stream_data=10485760) -> QuicConfiguration:
    config = QuicConfiguration(is_client=is_client)
    config.max_datagram_frame_size = 65536
    config.idle_timeout = 600.0  # Increased for long training sessions
    config.max_data = 104857600
    config.max_stream_data = max_stream_data
    config.alpn_protocols = ["fl-quic-v1"]
    
    if not is_client:
        if cert_file and key_file: config.load_cert_chain(cert_file, key_file)
    
    if is_client:
        import ssl
        config.verify_mode = verify_mode if verify_mode is not None else ssl.CERT_NONE
        if enable_0rtt:
            config.session_ticket_handler = ticket_handler.save_ticket
            ticket = ticket_handler.load_ticket()
            if ticket: config.session_ticket = ticket
    
    return config
