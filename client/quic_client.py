"""
QUIC Client for Federated Learning
Connects to FL server, receives global model, trains locally, sends updates

Optimized for:
- Edge devices (Jetson Nano, ARM64)
- Unstable networks (4G/WiFi)
- Low bandwidth via QUIC + compression

Author: Research Team - FL-QUIC-LoRA Project
"""

import asyncio
import logging
from typing import List, Optional, Callable, Any, Dict
from pathlib import Path
import numpy as np
from functools import partial
from aioquic.asyncio.client import connect
from aioquic.quic.connection import QuicConnection
from aioquic.quic.events import QuicEvent

import sys
sys.path.append(str(Path(__file__).parent.parent))

from transport.quic_protocol import (
    FLQuicProtocol,
    FLMessageHandler,
    create_quic_config,
    StreamType
)
from transport.serializer import ModelSerializer
from transport.network_monitor import NetworkMonitor
from utils.metrics import SystemMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FLClientHandler(FLMessageHandler):
    """
    Client-side message handler for Federated Learning.
    Processes global model from server and triggers local training.
    """
    
    def __init__(self, client: 'FLQuicClient'):
        """
        Initialize client handler.
        
        Args:
            client: Reference to the FL client
        """
        super().__init__()
        self.client = client
        logger.info("FLClientHandler initialized")
    
    async def handle_weights(self, stream_id: int, payload: bytes) -> None:
        """
        Handle global model weights from server.
        
        Args:
            stream_id: Stream ID
            payload: Serialized weights
        """
        try:
            # Deserialize weights
            weights = self.serializer.deserialize_weights(payload)
            
            logger.info(f"Received global model: {len(weights)} weight arrays")
            
            # Trigger local training with received weights
            await self.client._on_global_model_received(weights)
            
        except Exception as e:
            logger.error(f"Failed to handle weights: {e}")
    
    async def handle_config(self, stream_id: int, payload: bytes) -> None:
        """
        Handle configuration from server.
        
        Args:
            stream_id: Stream ID
            payload: Serialized config
        """
        try:
            config = self.serializer.deserialize_metadata(payload)
            logger.info(f"Received config from server: {config}")
            
            # Update client configuration
            self.client.config.update(config)
            
        except Exception as e:
            logger.error(f"Failed to handle config: {e}")


class FLQuicClient:
    """
    QUIC-based Federated Learning Client.
    Connects to server, trains locally, sends updates via QUIC.
    """
    
    def __init__(
        self,
        server_host: str,
        server_port: int,
        client_id: Optional[str] = None,
        local_train_fn: Optional[Callable] = None,
        local_eval_fn: Optional[Callable] = None,
        network_monitor: Optional[NetworkMonitor] = None,
    ):
        """
        Initialize FL client.
        
        Args:
            server_host: Server hostname/IP
            server_port: Server port
            client_id: Unique client identifier
            local_train_fn: Function for local training
                           Signature: async def train(weights, config) -> (updated_weights, num_samples, metrics)
            local_eval_fn: Function for local evaluation (optional)
                          Signature: async def evaluate(weights, config) -> (loss, metrics)
            network_monitor: NetworkMonitor instance for tracking network quality
        """
        self.server_host = server_host
        self.server_port = server_port
        self.client_id = client_id or f"client_{id(self)}"
        
        # Training callbacks
        self.local_train_fn = local_train_fn
        self.local_eval_fn = local_eval_fn
        
        # Network monitoring
        self.network_monitor = network_monitor or NetworkMonitor()
        
        # State
        self.protocol: Optional[FLQuicProtocol] = None
        self.connected = False
        self.current_round = 0
        self.config: Dict[str, Any] = {}
        
        # Synchronization
        self._training_complete = asyncio.Event()
        self._shutdown = asyncio.Event()
        
        # Statistics
        self.stats = {
            'rounds_completed': 0,
            'updates_sent': 0,
            'total_samples_trained': 0,
        }
        
        # System metrics tracker for communication cost
        self.system_metrics = SystemMetrics()
        
        # Message handler
        self.message_handler = FLClientHandler(self)
        
        logger.info(f"FLQuicClient initialized: {client_id}, server={server_host}:{server_port}")
    
    async def connect(self) -> bool:
        """
        Connect to FL server via QUIC.
        """
        try:
            # Create QUIC configuration
            config = create_quic_config(is_client=True)
            
            logger.info(f"Connecting to {self.server_host}:{self.server_port}...")
            
            # Establish QUIC connection
            # Fix: Use functools.partial to bind stream_handler to FLQuicProtocol constructor
            # aioquic will call this partial function with just the 'quic' argument
            # FIX: Use lambda to properly bind stream_handler (partial doesn't work with aioquic)
            handler = self.message_handler.handle_message
            def create_protocol(quic):
                return FLQuicProtocol(quic, stream_handler=handler)
            
            async with connect(
                host=self.server_host,
                port=self.server_port,
                configuration=config,
                create_protocol=FLQuicProtocol,  # Just pass the class
            ) as client:
                self.protocol = client
                
                # CRITICAL FIX: Set stream_handler AFTER connection
                # aioquic doesn't respect kwargs in create_protocol factory
                self.protocol._stream_handler = self.message_handler.handle_message
                logger.info(f"Stream handler configured: {self.protocol._stream_handler is not None}")
                
                # Wait for handshake
                if self.protocol is None:
                    logger.error("Protocol not initialized")
                    return False
                
                connected = await self.protocol.wait_for_connection(timeout=10.0)
                
                if connected:
                    self.connected = True
                    logger.info(f"Connected to server via QUIC")
                    
                    # Send initial metadata
                    await self._send_client_info()
                    
                    # Wait for shutdown or training completion
                    await self._shutdown.wait()
                else:
                    logger.error("Connection handshake failed")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _create_protocol(self, quic: QuicConnection) -> FLQuicProtocol:
        """
        Factory method for creating protocol instance.
        
        Args:
            quic: QUIC connection
            
        Returns:
            FLQuicProtocol instance
        """
        protocol = FLQuicProtocol(
            quic,
            stream_handler=self.message_handler.handle_message
        )
        return protocol
    
    async def _send_client_info(self) -> None:
        """Send client information to server"""
        try:
            # Auto-detect device type
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                # Map to device type
                if 'RTX 4070' in device_name:
                    device_type = 'rtx_4070'
                elif 'RTX' in device_name or 'GTX' in device_name:
                    device_type = 'nvidia_desktop'
                elif 'Orin' in device_name or 'Jetson' in device_name:
                    device_type = 'jetson'
                else:
                    device_type = device_name.replace(' ', '_').lower()
            else:
                device_type = 'cpu'
            
            metadata = {
                'client_id': self.client_id,
                'device_type': device_type,
                'capabilities': {
                    'gpu': torch.cuda.is_available(),
                    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    'max_batch_size': 32,
                },
            }
            
            if self.protocol is None:
                logger.error("Protocol not initialized")
                return
            self.protocol.send_metadata(metadata)
            logger.info(f"Sent client info to server: device={device_type}")
            
        except Exception as e:
            logger.error(f"Failed to send client info: {e}")
    
    async def _on_global_model_received(self, global_weights: List[np.ndarray]) -> None:
        """
        Callback when global model is received from server.
        Triggers local training.
        
        Args:
            global_weights: Global model weights from server
        """
        logger.info(f"Starting local training for round {self.current_round + 1}")
        
        try:
            if self.local_train_fn is None:
                logger.error("No training function provided!")
                return
            
            # Perform local training
            updated_weights, num_samples, metrics = await self.local_train_fn(
                global_weights,
                self.config
            )
            
            logger.info(f"Local training complete: {num_samples} samples, metrics={metrics}")
            
            # Send update to server
            await self._send_update(updated_weights, num_samples, metrics)
            
            # Update state
            self.current_round += 1
            self.stats['rounds_completed'] += 1
            self.stats['total_samples_trained'] += num_samples
            
        except Exception as e:
            logger.error(f"Local training failed: {e}")
    
    async def _send_update(
        self,
        weights: List[np.ndarray],
        num_samples: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Send local update to server with communication metrics tracking.
        CORRECT ORDER: Metadata FIRST (Stream 8), then Weights (Stream 4).
        """
        try:
            import time
            send_time = time.time()
            
            # Get connection stats before sending (if available)
            bytes_sent_before = 0
            bytes_received_before = 0
            if self.protocol and hasattr(self.protocol, '_quic'):
                stats_before = self.protocol._quic._loss.get_stats()  # type: ignore
                bytes_sent_before = getattr(stats_before, 'bytes_sent', 0)
                bytes_received_before = getattr(stats_before, 'bytes_received', 0)
            
            # 1. Send Metadata FIRST
            # This ensures the server knows WHO is sending and WHAT the status is
            # even if the large weight file gets delayed.
            metadata = {
                'client_id': self.client_id,
                'round': self.current_round,
                'num_samples': num_samples,
                'metrics': metrics,
                'send_timestamp': send_time,
            }
            logger.info(f"Sending metadata (Stream 8)...")
            
            if self.protocol is None:
                logger.error("Protocol not initialized")
                raise RuntimeError("Cannot send update: protocol not initialized")
            
            self.protocol.send_metadata(metadata)
            
            # 2. Send Weights SECOND
            logger.info(f"Sending weights (Stream 4): {len(weights)} arrays...")
            self.protocol.send_weights(weights)
            
            # Get connection stats after sending
            bytes_sent_after = bytes_sent_before
            bytes_received_after = bytes_received_before
            if self.protocol and hasattr(self.protocol, '_quic'):
                stats_after = self.protocol._quic._loss.get_stats()  # type: ignore
                bytes_sent_after = getattr(stats_after, 'bytes_sent', bytes_sent_before)
                bytes_received_after = getattr(stats_after, 'bytes_received', bytes_received_before)
            
            # Update system metrics with deltas
            comm_metrics = self.system_metrics.update_communication(
                bytes_sent=bytes_sent_after,
                bytes_received=bytes_received_after
            )
            
            self.stats['updates_sent'] += 1
            logger.info(f"Update sent successfully: "
                       f"Sent {SystemMetrics.format_bytes(comm_metrics['bytes_sent_delta'])}, "
                       f"Received {SystemMetrics.format_bytes(comm_metrics['bytes_received_delta'])}")
            
        except Exception as e:
            logger.error(f"Failed to send update: {e}")
            raise
    
    async def run(self) -> None:
        """Main client loop"""
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING FL CLIENT: {self.client_id}")
        logger.info(f"{'='*60}")
        
        # Connect to server
        connected = await self.connect()
        
        if not connected:
            logger.error("Failed to connect to server")
            return
        
        logger.info("Client running - waiting for global models...")
        
        # Keep running until shutdown
        await self._shutdown.wait()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CLIENT STOPPED: {self.client_id}")
        logger.info(f"Rounds completed: {self.stats['rounds_completed']}")
        logger.info(f"Updates sent: {self.stats['updates_sent']}")
        logger.info(f"Total samples: {self.stats['total_samples_trained']}")
        logger.info(f"{'='*60}")
    
    def shutdown(self) -> None:
        """Shutdown the client gracefully"""
        logger.info("Shutting down client...")
        
        if self.protocol:
            try:
                self.protocol.close(reason_phrase="Client shutdown")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
        
        self._shutdown.set()


# Example training function for testing
async def example_train_fn(
    weights: List[np.ndarray],
    config: Dict[str, Any]
) -> tuple[List[np.ndarray], int, Dict[str, float]]:
    """
    Example training function (replace with actual implementation).
    
    Args:
        weights: Global model weights
        config: Training configuration
        
    Returns:
        (updated_weights, num_samples, metrics)
    """
    logger.info("Example training function - simulating training...")
    
    # Simulate training delay
    await asyncio.sleep(2.0)
    
    # Simulate weight updates (add small random noise)
    updated_weights = [w + np.random.randn(*w.shape) * 0.01 for w in weights]
    
    # Simulate metrics
    num_samples = 100
    metrics = {
        'loss': 0.5 + np.random.rand() * 0.1,
        'accuracy': 0.8 + np.random.rand() * 0.1,
    }
    
    return updated_weights, num_samples, metrics


async def main():
    """Entry point for running the client"""
    # Client configuration
    client = FLQuicClient(
        server_host="localhost",  # Change to server IP
        server_port=4433,
        client_id="test_client_1",
        local_train_fn=example_train_fn,  # Replace with actual training function
    )
    
    try:
        await client.run()
    except KeyboardInterrupt:
        logger.info("Client interrupted by user")
        client.shutdown()
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
