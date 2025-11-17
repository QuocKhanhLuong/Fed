"""
QUIC Server for Federated Learning
Manages FL rounds, client connections, and model aggregation

Key Features:
- Handles multiple concurrent clients via QUIC streams
- Coordinates FL rounds (send global model, receive updates)
- Integrates with Flower Strategy for aggregation
- Robust error handling for unstable edge networks

Author: Research Team - FL-QUIC-LoRA Project
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.asyncio.server import serve
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
from transport.serializer import MessageCodec, ModelSerializer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ClientState:
    """Track state of each connected client"""
    client_id: str
    protocol: FLQuicProtocol
    address: Tuple[str, int]
    connected_at: datetime = field(default_factory=datetime.now)
    current_round: int = 0
    last_update_received: Optional[datetime] = None
    num_samples: int = 0
    is_training: bool = False
    
    def __repr__(self):
        return f"Client({self.client_id}, {self.address}, round={self.current_round})"


class FLServerHandler(FLMessageHandler):
    """
    Server-side message handler for Federated Learning.
    Processes client updates and coordinates training rounds.
    """
    
    def __init__(self, server: 'FLQuicServer'):
        """
        Initialize server handler.
        
        Args:
            server: Reference to the FL server
        """
        super().__init__()
        self.server = server
        logger.info("FLServerHandler initialized")
    
    async def handle_weights(self, stream_id: int, payload: bytes) -> None:
        """
        Handle weight updates from clients.
        
        Args:
            stream_id: Stream ID
            payload: Serialized weights
        """
        try:
            # Deserialize weights
            weights = self.serializer.deserialize_weights(payload)
            
            logger.info(f"Received {len(weights)} weight arrays from stream {stream_id}")
            
            # Find client by stream
            client_id = self.server._find_client_by_stream(stream_id)
            if client_id:
                # Store update for aggregation
                await self.server._receive_client_update(client_id, weights)
            else:
                logger.warning(f"Unknown client for stream {stream_id}")
                
        except Exception as e:
            logger.error(f"Failed to handle weights: {e}")
    
    async def handle_metadata(self, stream_id: int, payload: bytes) -> None:
        """
        Handle metadata from clients (metrics, num_samples, etc.).
        
        Args:
            stream_id: Stream ID
            payload: Serialized metadata
        """
        try:
            metadata = self.serializer.deserialize_metadata(payload)
            logger.info(f"Received metadata from stream {stream_id}: {metadata}")
            
            # Update client state
            client_id = self.server._find_client_by_stream(stream_id)
            if client_id and client_id in self.server.clients:
                client = self.server.clients[client_id]
                client.num_samples = metadata.get('num_samples', 0)
                client.last_update_received = datetime.now()
                
        except Exception as e:
            logger.error(f"Failed to handle metadata: {e}")


class FLQuicServer:
    """
    QUIC-based Federated Learning Server.
    Coordinates training rounds and aggregates client updates.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 4433,
        cert_file: Optional[str] = None,
        key_file: Optional[str] = None,
        num_rounds: int = 10,
        min_clients: int = 2,
        min_available_clients: int = 2,
    ):
        """
        Initialize FL server.
        
        Args:
            host: Server host address
            port: Server port
            cert_file: Path to TLS certificate (optional for dev)
            key_file: Path to TLS private key (optional for dev)
            num_rounds: Total number of FL rounds
            min_clients: Minimum clients to start training
            min_available_clients: Minimum clients per round
        """
        self.host = host
        self.port = port
        self.cert_file = cert_file
        self.key_file = key_file
        
        # FL configuration
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.min_available_clients = min_available_clients
        
        # State management
        self.clients: Dict[str, ClientState] = {}
        self.current_round = 0
        self.global_weights: Optional[List[np.ndarray]] = None
        self.client_updates: Dict[str, Tuple[List[np.ndarray], int]] = {}  # client_id -> (weights, num_samples)
        
        # Synchronization
        self._round_complete = asyncio.Event()
        self._server_started = asyncio.Event()
        self._shutdown = asyncio.Event()
        
        # Statistics
        self.stats = {
            'total_clients_connected': 0,
            'total_rounds_completed': 0,
            'total_updates_received': 0,
            'start_time': None,
        }
        
        # Message handler
        self.message_handler = FLServerHandler(self)
        
        logger.info(f"FLQuicServer initialized: {host}:{port}, rounds={num_rounds}")
    
    def _create_protocol(self, quic: QuicConnection) -> FLQuicProtocol:
        """
        Factory method for creating protocol instances.
        
        Args:
            quic: QUIC connection
            
        Returns:
            FLQuicProtocol instance
        """
        # Create protocol with message handler
        protocol = FLQuicProtocol(
            quic,
            stream_handler=self.message_handler.handle_message
        )
        
        # Generate client ID from connection
        client_id = f"client_{len(self.clients)}_{id(quic)}"
        
        # Get remote address
        remote_addr = quic._network_paths[0].addr if quic._network_paths else ("unknown", 0)
        
        # Register client
        client_state = ClientState(
            client_id=client_id,
            protocol=protocol,
            address=remote_addr,
        )
        self.clients[client_id] = client_state
        self.stats['total_clients_connected'] += 1
        
        logger.info(f"New client connected: {client_id} from {remote_addr}")
        
        return protocol
    
    def _find_client_by_stream(self, stream_id: int) -> Optional[str]:
        """
        Find client ID by stream ID.
        
        Args:
            stream_id: QUIC stream ID
            
        Returns:
            Client ID or None
        """
        # This is a simplified version - in production, maintain stream->client mapping
        for client_id, client in self.clients.items():
            if stream_id in client.protocol._active_streams:
                return client_id
        return None
    
    async def _receive_client_update(self, client_id: str, weights: List[np.ndarray]) -> None:
        """
        Receive and store client update for aggregation.
        
        Args:
            client_id: Client identifier
            weights: Updated model weights
        """
        if client_id not in self.clients:
            logger.warning(f"Update from unknown client: {client_id}")
            return
        
        client = self.clients[client_id]
        num_samples = client.num_samples or 1  # Default to 1 if not provided
        
        # Store update
        self.client_updates[client_id] = (weights, num_samples)
        self.stats['total_updates_received'] += 1
        
        logger.info(f"Stored update from {client_id}: {len(weights)} arrays, "
                   f"{num_samples} samples")
        
        # Check if round is complete
        if len(self.client_updates) >= self.min_available_clients:
            self._round_complete.set()
    
    def _aggregate_weights(self) -> List[np.ndarray]:
        """
        Aggregate client weights using FedAvg (weighted average).
        
        Returns:
            Aggregated global weights
        """
        if not self.client_updates:
            raise ValueError("No client updates to aggregate")
        
        logger.info(f"Aggregating {len(self.client_updates)} client updates...")
        
        # Extract weights and sample counts
        client_weights = []
        client_samples = []
        
        for client_id, (weights, num_samples) in self.client_updates.items():
            client_weights.append(weights)
            client_samples.append(num_samples)
        
        # Calculate total samples
        total_samples = sum(client_samples)
        
        # Weighted average (FedAvg)
        aggregated = []
        num_layers = len(client_weights[0])
        
        for layer_idx in range(num_layers):
            # Stack all client weights for this layer
            layer_weights = [client[layer_idx] for client in client_weights]
            
            # Weighted sum
            weighted_sum = np.zeros_like(layer_weights[0], dtype=np.float32)
            for weight, num_samples in zip(layer_weights, client_samples):
                weighted_sum += weight * (num_samples / total_samples)
            
            aggregated.append(weighted_sum)
        
        logger.info(f"Aggregation complete: {num_layers} layers aggregated")
        return aggregated
    
    async def _broadcast_global_model(self, weights: List[np.ndarray]) -> None:
        """
        Broadcast global model to all connected clients.
        
        Args:
            weights: Global model weights
        """
        logger.info(f"Broadcasting global model to {len(self.clients)} clients...")
        
        successful = 0
        for client_id, client in self.clients.items():
            try:
                # Send weights via QUIC
                client.protocol.send_weights(weights)
                client.current_round = self.current_round
                client.is_training = True
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to send model to {client_id}: {e}")
        
        logger.info(f"Broadcast complete: {successful}/{len(self.clients)} successful")
    
    async def _run_training_round(self) -> None:
        """Execute a single FL training round"""
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {self.current_round + 1}/{self.num_rounds} STARTED")
        logger.info(f"{'='*60}")
        
        # Clear previous updates
        self.client_updates.clear()
        self._round_complete.clear()
        
        # Broadcast global model to clients
        if self.global_weights is not None:
            await self._broadcast_global_model(self.global_weights)
        else:
            logger.warning("No global weights - clients will train from scratch")
        
        # Wait for client updates
        logger.info(f"Waiting for {self.min_available_clients} client updates...")
        
        try:
            # Wait with timeout (e.g., 5 minutes per round)
            await asyncio.wait_for(self._round_complete.wait(), timeout=300.0)
            
            # Aggregate updates
            self.global_weights = self._aggregate_weights()
            
            # Update round counter
            self.current_round += 1
            self.stats['total_rounds_completed'] += 1
            
            logger.info(f"ROUND {self.current_round}/{self.num_rounds} COMPLETED")
            
        except asyncio.TimeoutError:
            logger.error(f"Round {self.current_round + 1} timeout - insufficient updates")
    
    async def run_federated_learning(self) -> None:
        """Main FL loop - executes all training rounds"""
        logger.info("\n" + "="*60)
        logger.info("STARTING FEDERATED LEARNING")
        logger.info("="*60)
        
        self.stats['start_time'] = datetime.now()
        
        # Wait for minimum clients
        logger.info(f"Waiting for {self.min_clients} clients to connect...")
        while len(self.clients) < self.min_clients:
            await asyncio.sleep(1.0)
        
        logger.info(f"{len(self.clients)} clients ready - starting training")
        
        # Run all rounds
        for round_num in range(self.num_rounds):
            await self._run_training_round()
        
        # Training complete
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        logger.info("\n" + "="*60)
        logger.info("FEDERATED LEARNING COMPLETED")
        logger.info(f"Total rounds: {self.stats['total_rounds_completed']}")
        logger.info(f"Total updates: {self.stats['total_updates_received']}")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info("="*60)
    
    async def start(self) -> None:
        """Start the QUIC server"""
        # Create QUIC configuration
        config = create_quic_config(
            is_client=False,
            cert_file=self.cert_file,
            key_file=self.key_file,
        )
        
        logger.info(f"Starting QUIC server on {self.host}:{self.port}...")
        
        # Start QUIC server
        await serve(
            host=self.host,
            port=self.port,
            configuration=config,
            create_protocol=self._create_protocol,
        )
        
        self._server_started.set()
        logger.info(f"QUIC server listening on {self.host}:{self.port}")
        
        # Start FL training
        await self.run_federated_learning()
        
        # Wait for shutdown
        await self._shutdown.wait()
    
    def shutdown(self) -> None:
        """Shutdown the server gracefully"""
        logger.info("Shutting down server...")
        
        # Close all client connections
        for client_id, client in self.clients.items():
            try:
                client.protocol.close(reason_phrase="Server shutdown")
            except Exception as e:
                logger.error(f"Error closing client {client_id}: {e}")
        
        self._shutdown.set()


async def main():
    """Entry point for running the server"""
    # Server configuration
    server = FLQuicServer(
        host="0.0.0.0",
        port=4433,
        num_rounds=10,
        min_clients=2,
        min_available_clients=2,
    )
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        server.shutdown()
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
