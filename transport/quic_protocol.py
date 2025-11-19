"""
QUIC Protocol Handler for Federated Learning
Implements Stream Multiplexing and 0-RTT for low-latency communication

Key Features:
- Bidirectional Streams: Separate streams for weights, metadata, control
- 0-RTT Support: Resume sessions without handshake overhead
- Connection Migration: Handle network changes (WiFi <-> 4G)
- Congestion Control: Optimized for unstable networks

Author: Research Team - FL-QUIC-LoRA Project
"""

import asyncio
import logging
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass
from enum import IntEnum

from aioquic.asyncio import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import (
    QuicEvent,
    StreamDataReceived,
    ConnectionTerminated,
    HandshakeCompleted,
    StreamReset,
)
from aioquic.quic.connection import QuicConnection

from .serializer import ModelSerializer, MessageCodec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamType(IntEnum):
    """Stream types for multiplexing different data types"""
    CONTROL = 0      # Control messages (start/stop training, config)
    WEIGHTS = 4      # Model weights (LoRA parameters)
    METADATA = 8     # Metrics, logs, status updates
    ERROR = 12       # Error reporting


@dataclass
class FLMessage:
    """Federated Learning message container"""
    stream_type: StreamType
    msg_type: int
    payload: bytes
    metadata: Optional[Dict[str, Any]] = None


class FLQuicProtocol(QuicConnectionProtocol):
    """
    Custom QUIC Protocol for Federated Learning.
    Handles stream multiplexing and message routing.
    """
    
    def __init__(self, quic: QuicConnection, stream_handler: Optional[Callable] = None):
        """
        Initialize the QUIC protocol handler.
        
        Args:
            quic: QUIC connection instance
            stream_handler: Async callback for handling received messages
                           Signature: async def handler(stream_id, msg_type, payload) -> None
        """
        super().__init__(quic)
        self._quic = quic
        self._stream_handler = stream_handler
        self._serializer = ModelSerializer()
        self._codec = MessageCodec()
        
        # Stream management
        self._active_streams: Dict[int, StreamType] = {}
        self._receive_buffers: Dict[int, bytearray] = {}
        
        # Connection state
        self._handshake_completed = False
        self._connection_ready = asyncio.Event()
        
        # Statistics
        self._stats = {
            'bytes_sent': 0,
            'bytes_received': 0,
            'streams_created': 0,
            'messages_sent': 0,
            'messages_received': 0,
        }
        
        logger.info("FLQuicProtocol initialized")
    
    def quic_event_received(self, event: QuicEvent) -> None:
        """
        Handle QUIC events (called by aioquic).
        
        Args:
            event: QUIC event (stream data, handshake, etc.)
        """
        if isinstance(event, HandshakeCompleted):
            self._handle_handshake_completed(event)
        elif isinstance(event, StreamDataReceived):
            self._handle_stream_data(event)
        elif isinstance(event, ConnectionTerminated):
            self._handle_connection_terminated(event)
        elif isinstance(event, StreamReset):
            self._handle_stream_reset(event)
    
    def _handle_handshake_completed(self, event: HandshakeCompleted) -> None:
        """Handle successful QUIC handshake"""
        self._handshake_completed = True
        self._connection_ready.set()
        
        logger.info(f"QUIC handshake completed - 0-RTT: {event.early_data_accepted}, "
                   f"ALPN: {event.alpn_protocol}")
    
    def _handle_stream_data(self, event: StreamDataReceived) -> None:
        """
        Handle incoming stream data.
        
        Args:
            event: Stream data event
        """
        stream_id = event.stream_id
        data = event.data
        end_stream = event.end_stream
        
        # Update statistics
        self._stats['bytes_received'] += len(data)
        
        # Buffer Management
        MAX_BUFFER_SIZE = 50 * 1024 * 1024  # 50 MB limit
        
        # Initialize buffer if needed
        if stream_id not in self._receive_buffers:
            self._receive_buffers[stream_id] = bytearray()
        
        # Check buffer size (Backpressure)
        current_size = len(self._receive_buffers[stream_id])
        if current_size + len(data) > MAX_BUFFER_SIZE:
            logger.warning(f"Buffer overflow for stream {stream_id} (size={current_size}). Dropping packet.")
            # Option: Reset stream to signal congestion
            # self._quic.send_stream_reset(stream_id, 0x1) 
            return
        
        # Append data to buffer
        self._receive_buffers[stream_id].extend(data)
        
        # Process complete messages
        if end_stream:
            self._process_stream_buffer(stream_id)
            # Clean up
            del self._receive_buffers[stream_id]
            if stream_id in self._active_streams:
                del self._active_streams[stream_id]
    
    def _process_stream_buffer(self, stream_id: int) -> None:
        """
        Process a complete stream buffer and invoke handler.
        Supports multiple messages in a single stream (Multiplexing).
        
        Args:
            stream_id: QUIC stream ID
        """
        try:
            buffer = self._receive_buffers[stream_id]
            
            while len(buffer) > 0:
                # Need at least header (5 bytes)
                if len(buffer) < 5:
                    break
                    
                # Peek length
                import struct
                length = struct.unpack('>I', buffer[0:4])[0]
                
                # Check if we have full message
                if len(buffer) < 5 + length:
                    break
                
                # Extract full message data
                msg_data = buffer[:5+length]
                
                # Decode message
                msg_type, payload = self._codec.decode_message(msg_data)
                self._stats['messages_received'] += 1
                
                logger.debug(f"Received message on stream {stream_id}: "
                            f"type={msg_type}, size={len(payload)}")
                
                # Invoke handler if set
                if self._stream_handler:
                    asyncio.create_task(
                        self._stream_handler(stream_id, msg_type, payload)
                    )
                
                # Remove processed message from buffer
                del buffer[:5+length]
            
        except Exception as e:
            logger.error(f"Failed to process stream {stream_id}: {e}")
            self._send_error_message(stream_id, str(e))
    
    def _handle_connection_terminated(self, event: ConnectionTerminated) -> None:
        """Handle connection termination"""
        logger.warning(f"Connection terminated: error_code={event.error_code}, "
                      f"reason={event.reason_phrase}")
        self._connection_ready.clear()
    
    def _handle_stream_reset(self, event: StreamReset) -> None:
        """Handle stream reset"""
        logger.warning(f"Stream {event.stream_id} reset: error_code={event.error_code}")
        
        # Clean up stream resources
        if event.stream_id in self._receive_buffers:
            del self._receive_buffers[event.stream_id]
        if event.stream_id in self._active_streams:
            del self._active_streams[event.stream_id]
    
    async def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """
        Wait for QUIC connection to be ready.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if connected, False if timeout
        """
        try:
            await asyncio.wait_for(self._connection_ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.error(f"Connection timeout after {timeout}s")
            return False
    
    def create_stream(self, stream_type: StreamType) -> int:
        """
        Create a new QUIC stream for specific purpose.
        
        Args:
            stream_type: Type of stream to create
            
        Returns:
            Stream ID
        """
        # Use stream_type as base for stream ID to enable multiplexing
        # Client-initiated streams: odd IDs
        # Server-initiated streams: even IDs
        stream_id = self._quic.get_next_available_stream_id()
        
        self._active_streams[stream_id] = stream_type
        self._stats['streams_created'] += 1
        
        logger.debug(f"Created stream {stream_id} for {stream_type.name}")
        return stream_id
    
    def _get_existing_stream(self, stream_type: StreamType) -> Optional[int]:
        """Find an existing stream of the given type."""
        for stream_id, stype in self._active_streams.items():
            if stype == stream_type:
                return stream_id
        return None

    def send_message(self, stream_id: int, msg_type: int, payload: bytes, 
                    end_stream: bool = True) -> None:
        """
        Send a message on a QUIC stream.
        
        Args:
            stream_id: Target stream ID
            msg_type: Message type (from MessageCodec)
            payload: Message payload (already serialized)
            end_stream: Whether to close stream after sending
        """
        try:
            # Encode message with header
            encoded = self._codec.encode_message(msg_type, payload)
            
            # Send via QUIC
            self._quic.send_stream_data(stream_id, encoded, end_stream=end_stream)
            
            # Update statistics
            self._stats['bytes_sent'] += len(encoded)
            self._stats['messages_sent'] += 1
            
            logger.debug(f"Sent message on stream {stream_id}: "
                        f"type={msg_type}, size={len(encoded)}, end_stream={end_stream}")
            
            # Transmit immediately
            self.transmit()
            
        except Exception as e:
            logger.error(f"Failed to send message on stream {stream_id}: {e}")
            raise RuntimeError(f"Send failed: {e}")
    
    def send_weights(self, weights: List, stream_id: Optional[int] = None) -> int:
        """
        Send model weights (NumPy arrays) via QUIC.
        Reuses existing WEIGHTS stream if available (Multiplexing).
        
        Args:
            weights: List of NumPy arrays
            stream_id: Existing stream ID or None to create new
            
        Returns:
            Stream ID used
        """
        try:
            # Serialize weights
            payload = self._serializer.serialize_weights(weights)
            
            # Create or use stream
            if stream_id is None:
                stream_id = self._get_existing_stream(StreamType.WEIGHTS)
                if stream_id is None:
                    stream_id = self.create_stream(StreamType.WEIGHTS)
            
            # Send (keep stream open for multiplexing)
            self.send_message(stream_id, MessageCodec.MSG_TYPE_WEIGHTS, payload, end_stream=False)
            
            logger.info(f"Sent {len(weights)} weight arrays on stream {stream_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to send weights: {e}")
            raise
    
    def send_metadata(self, metadata: Dict[str, Any], stream_id: Optional[int] = None) -> int:
        """
        Send metadata (metrics, config, etc.) via QUIC.
        Reuses existing METADATA stream if available (Multiplexing).
        
        Args:
            metadata: Dictionary of metadata
            stream_id: Existing stream ID or None to create new
            
        Returns:
            Stream ID used
        """
        try:
            # Serialize metadata
            payload = self._serializer.serialize_metadata(metadata)
            
            # Create or use stream
            if stream_id is None:
                stream_id = self._get_existing_stream(StreamType.METADATA)
                if stream_id is None:
                    stream_id = self.create_stream(StreamType.METADATA)
            
            # Send (keep stream open for multiplexing)
            self.send_message(stream_id, MessageCodec.MSG_TYPE_METADATA, payload, end_stream=False)
            
            logger.debug(f"Sent metadata on stream {stream_id}")
            return stream_id
            
        except Exception as e:
            logger.error(f"Failed to send metadata: {e}")
            raise
    
    def _send_error_message(self, stream_id: int, error_msg: str) -> None:
        """Send error message back to sender"""
        try:
            error_payload = error_msg.encode('utf-8')
            error_stream = self.create_stream(StreamType.ERROR)
            self.send_message(error_stream, MessageCodec.MSG_TYPE_ERROR, error_payload)
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get protocol statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self._stats,
            'active_streams': len(self._active_streams),
            'pending_buffers': len(self._receive_buffers),
            'connected': self._handshake_completed,
        }
    
    def close(self, error_code: int = 0, reason_phrase: str = "Normal close") -> None:
        """
        Close the QUIC connection gracefully.
        
        Args:
            error_code: Application error code
            reason_phrase: Human-readable reason
        """
        logger.info(f"Closing connection: {reason_phrase}")
        self._quic.close(error_code=error_code, reason_phrase=reason_phrase)
        self.transmit()


class FLMessageHandler:
    """
    Base class for handling FL messages.
    Subclass this for server/client specific logic.
    """
    
    def __init__(self, serializer: Optional[ModelSerializer] = None):
        """
        Initialize message handler.
        
        Args:
            serializer: Custom serializer instance (optional)
        """
        self.serializer = serializer or ModelSerializer()
        logger.info(f"{self.__class__.__name__} initialized")
    
    async def handle_message(self, stream_id: int, msg_type: int, payload: bytes) -> None:
        """
        Handle incoming message (override in subclass).
        
        Args:
            stream_id: Stream ID
            msg_type: Message type
            payload: Raw payload bytes
        """
        logger.debug(f"Received message: stream={stream_id}, type={msg_type}, size={len(payload)}")
        
        if msg_type == MessageCodec.MSG_TYPE_WEIGHTS:
            await self.handle_weights(stream_id, payload)
        elif msg_type == MessageCodec.MSG_TYPE_METADATA:
            await self.handle_metadata(stream_id, payload)
        elif msg_type == MessageCodec.MSG_TYPE_CONFIG:
            await self.handle_config(stream_id, payload)
        elif msg_type == MessageCodec.MSG_TYPE_ERROR:
            await self.handle_error(stream_id, payload)
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    async def handle_weights(self, stream_id: int, payload: bytes) -> None:
        """Handle weights message (override in subclass)"""
        weights = self.serializer.deserialize_weights(payload)
        logger.info(f"Received {len(weights)} weight arrays")
    
    async def handle_metadata(self, stream_id: int, payload: bytes) -> None:
        """Handle metadata message (override in subclass)"""
        metadata = self.serializer.deserialize_metadata(payload)
        logger.info(f"Received metadata: {list(metadata.keys())}")
    
    async def handle_config(self, stream_id: int, payload: bytes) -> None:
        """Handle config message (override in subclass)"""
        config = self.serializer.deserialize_metadata(payload)
        logger.info(f"Received config: {list(config.keys())}")
    
    async def handle_error(self, stream_id: int, payload: bytes) -> None:
        """Handle error message"""
        error_msg = payload.decode('utf-8')
        logger.error(f"Received error from stream {stream_id}: {error_msg}")


class SessionTicketHandler:
    """
    Handles saving and loading of QUIC session tickets for 0-RTT.
    """
    def __init__(self):
        self.tickets: Dict[str, Any] = {}
        
    def save_ticket(self, ticket: Any) -> None:
        """Save session ticket"""
        # In a real app, persist this to disk
        logger.info("Saving QUIC session ticket for 0-RTT")
        self.tickets['latest'] = ticket
        
    def load_ticket(self, key: str = 'latest') -> Optional[Any]:
        """Load session ticket"""
        return self.tickets.get(key)

# Global handler instance
ticket_handler = SessionTicketHandler()

def create_quic_config(
    is_client: bool,
    cert_file: Optional[str] = None,
    key_file: Optional[str] = None,
    verify_mode: Optional[int] = None,
    enable_0rtt: bool = True,
    max_stream_data: int = 10 * 1024 * 1024,  # 10 MB per stream
) -> QuicConfiguration:
    """
    Create QUIC configuration for client or server.
    
    Args:
        is_client: True for client, False for server
        cert_file: Path to certificate file (server only)
        key_file: Path to private key file (server only)
        verify_mode: SSL verification mode
        enable_0rtt: Enable 0-RTT for faster reconnection
        max_stream_data: Maximum data per stream (bytes)
        
    Returns:
        QuicConfiguration instance
    """
    config = QuicConfiguration(is_client=is_client)
    
    # Enable 0-RTT for low latency
    config.max_datagram_frame_size = 65536
    config.idle_timeout = 60.0  # 60 seconds
    
    # Optimize for unstable networks
    config.max_data = 100 * 1024 * 1024  # 100 MB total
    config.max_stream_data = max_stream_data
    
    # ALPN protocol
    config.alpn_protocols = ["fl-quic-v1"]
    
    # Server-specific settings
    if not is_client:
        if cert_file and key_file:
            config.load_cert_chain(cert_file, key_file)
        else:
            logger.warning("Server running without TLS certificates - using self-signed")
    
    # Client-specific settings
    if is_client:
        if verify_mode is not None:
            import ssl
            config.verify_mode = verify_mode
        else:
            # For development: skip verification
            import ssl
            config.verify_mode = ssl.CERT_NONE
            
        # 0-RTT Session Ticket Handling
        if enable_0rtt:
            config.session_ticket_handler = ticket_handler.save_ticket
            ticket = ticket_handler.load_ticket()
            if ticket:
                config.session_ticket = ticket
                logger.info("Loaded session ticket for 0-RTT")
    
    logger.info(f"QUIC config created: client={is_client}, 0-RTT={enable_0rtt}")
    return config
