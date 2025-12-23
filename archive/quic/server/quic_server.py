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
import time
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
from transport.serializer import ModelSerializer
from transport.quic_metrics import QUICMetrics
from server.aggregators import create_aggregator, BaseAggregator
from server.checkpoint_manager import CheckpointManager
from server.client_state_manager import create_client_manager, ClientProxy
from evaluation import FLEvaluator, ExperimentConfig

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
    
    async def handle_weights(self, stream_id: int, payload: bytes, protocol = None) -> None:
        """
        Handle weight updates from clients.
        
        Args:
            stream_id: Stream ID
            payload: Serialized weights
            protocol: Protocol instance that received this message
        """
        try:
            # Deserialize weights
            weights = self.serializer.deserialize_weights(payload)
            
            logger.info(f"Received {len(weights)} weight arrays from stream {stream_id}")
            
            # Find client by protocol object (scalable - each connection is unique)
            client_id = self.server._find_client_by_protocol(protocol)
            if client_id:
                # Store update for aggregation
                await self.server._receive_client_update(client_id, weights)
            else:
                logger.warning(f"Unknown client for stream {stream_id}")
                
        except Exception as e:
            logger.error(f"Failed to handle weights: {e}")
    
    async def handle_metadata(self, stream_id: int, payload: bytes, protocol = None) -> None:
        """
        Handle metadata from clients (metrics, num_samples, etc.).
        
        Args:
            stream_id: Stream ID
            payload: Serialized metadata
            protocol: Protocol instance
        """
        try:
            metadata = self.serializer.deserialize_metadata(payload)
            logger.info(f"Received metadata from stream {stream_id}: {metadata}")
            
            # Find client by protocol object
            client_id = self.server._find_client_by_protocol(protocol)
            if client_id:
                self.server.client_manager.update_heartbeat(client_id)
                # Update num_samples from metadata
                if 'num_samples' in metadata:
                    self.server.clients[client_id].num_samples = metadata['num_samples']
                
                # CRITICAL: Store metrics for accuracy/loss tracking
                if 'metrics' in metadata:
                    self.server.client_metrics[client_id] = metadata['metrics']
                    logger.info(f"Stored metrics for {client_id}: acc={metadata['metrics'].get('accuracy', 'N/A')}")
                
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
        strategy: str = "feddyn",
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
        
        # Flower-compatible Client Manager (auto-selects Redis or InMemory)
        self.client_manager = create_client_manager(backend="auto")
        
        # Aggregation Strategy (FedAvg, FedProx, or FedDyn)
        self.strategy = strategy
        self.aggregator = create_aggregator(strategy, alpha=0.01, mu=0.01)
        
        # Checkpoint Manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir="./checkpoints",
            save_frequency=5,
        )
        
        # FL Evaluator (IEEE metrics)
        self.evaluator = FLEvaluator(
            experiment_name=f"fl_feddyn_{datetime.now():%Y%m%d_%H%M%S}",
            save_dir="./results",
        )
        self.evaluator.set_config(ExperimentConfig(
            num_rounds=num_rounds,
            num_clients=min_clients,
            dataset="cifar100",
            strategy="FedDyn",
        ))
        
        # Track client metrics
        self.client_metrics: Dict[str, Dict[str, float]] = {}
        
        # QUIC Protocol Metrics (IEEE paper)
        self.quic_metrics = QUICMetrics()
        
        # Message handler
        self.message_handler = FLServerHandler(self)
        
        logger.info(f"FLQuicServer initialized: {host}:{port}, rounds={num_rounds}, strategy={strategy}")
    
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
        client_id = f"client_{id(quic)}"
        
        # Get remote address
        remote_addr = quic._network_paths[0].addr if quic._network_paths else ("unknown", 0)
        
        client_state = ClientState(
            client_id=client_id,
            protocol=protocol,
            address=remote_addr
        )
        self.clients[client_id] = client_state
        
        # Register client using Flower-compatible ClientProxy
        client_proxy = ClientProxy(
            cid=client_id,
            metadata={
                'address': str(remote_addr),
                'connected_at': str(datetime.now()),
                'status': 'connected'
            }
        )
        self.client_manager.register(client_proxy)
        self.stats['total_clients_connected'] += 1
        
        # Record QUIC connection metrics
        handshake_start = time.time()
        self.quic_metrics.record_connection(
            client_id=client_id,
            handshake_time_ms=(time.time() - handshake_start) * 1000,
            zero_rtt_used=False,  # Will be updated on handshake complete
            zero_rtt_accepted=False,
        )
        
        logger.info(f"New client connected: {client_id} from {remote_addr} (Total: {len(self.clients)})")
        
        return protocol
    
    def _find_client_by_protocol(self, protocol) -> Optional[str]:
        """
        Find client ID by protocol object reference.
        This is scalable - O(n) lookup by object identity, no stream ID collision.
        
        Args:
            protocol: FLQuicProtocol instance
            
        Returns:
            Client ID or None
        """
        if protocol is None:
            return None
        for client_id, client in self.clients.items():
            if client.protocol is protocol:
                return client_id
        return None
    
    def _find_client_by_stream(self, stream_id: int) -> Optional[str]:
        """
        Find client ID by stream ID.
        
        Args:
            stream_id: QUIC stream ID
            
        Returns:
            Client ID or None
        """
        # Check multiple data structures to find the client
        for client_id, client in self.clients.items():
            # Check active_streams (server-initiated)
            if stream_id in client.protocol._active_streams:
                return client_id
            # Check receive_buffers (client-initiated streams)
            if stream_id in client.protocol._receive_buffers:
                return client_id
            # Check if this stream belongs to the client's QUIC connection
            if hasattr(client.protocol, '_quic'):
                try:
                    if stream_id in client.protocol._quic._streams:
                        return client_id
                except:
                    pass
        
        # FALLBACK: If only one client is training, it must be them
        training_clients = [(cid, c) for cid, c in self.clients.items() if c.is_training]
        if len(training_clients) == 1:
            logger.info(f"  → Fallback: Only one training client, assuming {training_clients[0][0]}")
            return training_clients[0][0]
        
        # Last resort: check if stream is even (client-initiated)
        if stream_id % 4 == 0:  # QUIC bidirectional client-initiated
            # Find client that recently received model (is_training=True)
            for client_id, client in self.clients.items():
                if client.is_training and client_id not in self.client_updates:
                    logger.info(f"  → Fallback: Assuming training client {client_id}")
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
        
        # CRITICAL: Reset is_training so fallback lookup works for next client
        client.is_training = False
        client.last_update_received = datetime.now()
        
        # CRITICAL: Clear receive_buffers to prevent stream ID collision with next client
        # Each client reuses stream IDs 0,4,8 - old buffers cause wrong lookup
        client.protocol._receive_buffers.clear()
        client.protocol._active_streams.clear()
        
        logger.info(f"Stored update from {client_id}: {len(weights)} arrays, "
                   f"{num_samples} samples")
        
        # Check if round is complete
        if len(self.client_updates) >= self.min_available_clients:
            self._round_complete.set()
    
    def _aggregate_weights(self) -> List[np.ndarray]:
        """
        Aggregate client weights using FedDyn.
        
        Returns:
            Aggregated global weights
        """
        if not self.client_updates:
            raise ValueError("No client updates to aggregate")
        
        logger.info(f"FedDyn aggregating {len(self.client_updates)} client updates...")
        
        # Use FedDyn aggregator
        aggregated = self.aggregator.aggregate(
            self.client_updates,
            self.global_weights
        )
        
        logger.info(f"FedDyn aggregation complete: {len(aggregated)} layers")
        return aggregated
    
    async def _broadcast_global_model(self, weights: List[np.ndarray], sequential: bool = True) -> None:
        """
        Broadcast global model to all connected clients.
        
        Args:
            weights: Global model weights
            sequential: If True, send to one client at a time and wait for update
        """
        logger.info(f"Broadcasting global model to {len(self.clients)} clients...")
        
        if sequential:
            # SEQUENTIAL: Train one client at a time (for single GPU)
            logger.info("Using SEQUENTIAL mode (single GPU optimization)")
            for client_id, client in self.clients.items():
                try:
                    logger.info(f"  → Sending model to {client_id}...")
                    client.protocol.send_weights(weights)
                    client.current_round = self.current_round
                    client.is_training = True
                    
                    # Wait for this client's update before sending to next
                    logger.info(f"  → Waiting for {client_id} to complete training...")
                    timeout = 600  # 10 minutes per client
                    start_wait = datetime.now()
                    last_log = 0
                    
                    while client_id not in self.client_updates:
                        await asyncio.sleep(1.0)
                        elapsed = (datetime.now() - start_wait).total_seconds()
                        
                        # Log every 30s
                        if int(elapsed) // 30 > last_log:
                            last_log = int(elapsed) // 30
                            logger.info(f"    ... {client_id} training ({int(elapsed)}s)")
                        
                        if elapsed > timeout:
                            logger.warning(f"  → Timeout waiting for {client_id}")
                            break
                    
                    if client_id in self.client_updates:
                        logger.info(f"  ✓ {client_id} completed training")
                    
                except Exception as e:
                    logger.error(f"Failed to send model to {client_id}: {e}")
            
            logger.info(f"Sequential broadcast complete: {len(self.client_updates)}/{len(self.clients)} received")
        else:
            # PARALLEL: Send to all at once (original behavior)
            successful = 0
            for client_id, client in self.clients.items():
                try:
                    client.protocol.send_weights(weights)
                    client.current_round = self.current_round
                    client.is_training = True
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to send model to {client_id}: {e}")
            
            logger.info(f"Broadcast complete: {successful}/{len(self.clients)} successful")
    
    async def _run_training_round(self) -> None:
        """Execute a single FL training round"""
        round_start = datetime.now()
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {self.current_round + 1}/{self.num_rounds} STARTED")
        logger.info(f"{'='*60}")
        
        # Clear previous updates
        self.client_updates.clear()
        self.client_metrics.clear()
        self._round_complete.clear()
        
        # Broadcast global model to clients
        bytes_sent = 0
        if self.global_weights is not None:
            await self._broadcast_global_model(self.global_weights)
            # Estimate bytes sent
            bytes_sent = sum(w.nbytes for w in self.global_weights) * len(self.clients)
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
            
            # Calculate metrics from client updates
            client_accuracies = []
            total_loss = 0.0
            total_samples = 0
            bytes_received = 0
            
            for client_id, (weights, num_samples) in self.client_updates.items():
                bytes_received += sum(w.nbytes for w in weights)
                # Get metrics if available
                if client_id in self.client_metrics:
                    client_accuracies.append(self.client_metrics[client_id].get('accuracy', 0.5))
                    total_loss += self.client_metrics[client_id].get('loss', 1.0) * num_samples
                else:
                    client_accuracies.append(0.5)  # Default
                total_samples += num_samples
            
            global_accuracy = np.mean(client_accuracies) if client_accuracies else 0.5
            global_loss = total_loss / max(1, total_samples)
            round_time = (datetime.now() - round_start).total_seconds()
            
            # Log to evaluator
            self.evaluator.log_round(
                round_num=self.current_round,
                global_accuracy=global_accuracy,
                global_loss=global_loss,
                client_accuracies=client_accuracies,
                bytes_sent=bytes_sent,
                bytes_received=bytes_received,
                round_time_s=round_time,
            )
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                weights=self.global_weights,
                round_num=self.current_round,
                accuracy=global_accuracy,
                metrics={'loss': global_loss, 'clients': len(client_accuracies)},
            )
            
            logger.info(f"ROUND {self.current_round}/{self.num_rounds} COMPLETED")
            logger.info(f"  Global accuracy: {global_accuracy:.4f}")
            logger.info(f"  Best accuracy: {self.checkpoint_manager.best_accuracy:.4f}")
            
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
        
        # Save final results
        logger.info("\n" + "="*60)
        logger.info("FEDERATED LEARNING COMPLETED")
        logger.info(f"Total rounds: {self.stats['total_rounds_completed']}")
        logger.info(f"Total updates: {self.stats['total_updates_received']}")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Best accuracy: {self.checkpoint_manager.best_accuracy:.4f} (round {self.checkpoint_manager.best_round})")
        logger.info("="*60)
        
        # Generate IEEE tables and save results
        logger.info("\n📊 Generating IEEE Publication Tables...")
        print(self.evaluator.generate_tables())
        
        # QUIC Protocol metrics
        logger.info("\n📡 QUIC Transport Metrics:")
        print(self.quic_metrics.generate_tables())
        print(self.quic_metrics.generate_comparison_table())
        
        json_path, md_path = self.evaluator.save_results()
        logger.info(f"\n✅ Results saved:")
        logger.info(f"   - JSON: {json_path}")
        logger.info(f"   - Tables: {md_path}")
        logger.info(f"   - Best model: {self.checkpoint_manager.checkpoint_dir}/best_model.npz")
        
        # AUTO-SHUTDOWN after all rounds complete
        logger.info("\n🔌 Shutting down server and clients...")
        self.shutdown()
    
    async def start(self) -> None:
        """Start the QUIC server"""
        # Create QUIC configuration
        config = create_quic_config(
            is_client=False,
            cert_file=self.cert_file,
            key_file=self.key_file,
        )
        
        logger.info(f"Starting QUIC server on {self.host}:{self.port}...")
        
        # Protocol factory để tránh lỗi parameter binding
        def protocol_factory(quic_conn, *args, **kwargs):
            # Bỏ qua mọi tham số thừa từ aioquic
            return self._create_protocol(quic_conn)
        
        # Start QUIC server
        await serve(
            host=self.host,
            port=self.port,
            configuration=config,
            create_protocol=protocol_factory,
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
