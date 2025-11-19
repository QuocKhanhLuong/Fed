"""
FL-QUIC Client Application
Main entry point for Federated Learning client with QUIC transport

Integrates:
- QUIC Client (transport)
- Flower Client (FL logic)
- MobileViT + LoRA Trainer (model)

Author: Research Team - FL-QUIC-LoRA Project
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from client.quic_client import FLQuicClient
from client.fl_client import create_fl_client
from client.model_trainer import MobileViTLoRATrainer
from utils.config import Config, get_jetson_config
from transport.serializer import ModelSerializer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FLClientApp:
    """
    Complete FL Client Application.
    Combines QUIC transport with Flower FL logic.
    """
    
    def __init__(
        self,
        server_host: str,
        server_port: int,
        client_id: str,
        config: Config,
    ):
        """
        Initialize FL client application.
        
        Args:
            server_host: Server hostname/IP
            server_port: Server port
            client_id: Unique client identifier
            config: Configuration object
        """
        self.server_host = server_host
        self.server_port = server_port
        self.client_id = client_id
        self.config = config
        
        # Create NetworkMonitor
        from transport.network_monitor import NetworkMonitor
        logger.info("Creating NetworkMonitor...")
        self.network_monitor = NetworkMonitor(window_size=20)
        
        # Create Flower client with NetworkMonitor
        logger.info("Creating Flower client...")
        self.fl_client = create_fl_client(
            num_classes=config.model.num_classes,
            lora_r=config.model.lora_r,
            local_epochs=config.training.local_epochs,
            learning_rate=config.training.learning_rate,
            use_dummy_data=True,  # TODO: Load real dataset
            network_monitor=self.network_monitor,  # Pass NetworkMonitor
        )
        
        # Create async training function for QUIC client
        async def train_callback(weights, config_dict):
            """Callback for local training"""
            # Convert to Flower format
            updated_weights, num_samples, metrics = self.fl_client.fit(
                weights,
                config_dict
            )
            return updated_weights, num_samples, metrics
        
        # Create QUIC client with NetworkMonitor
        logger.info("Creating QUIC client...")
        self.quic_client = FLQuicClient(
            server_host=server_host,
            server_port=server_port,
            client_id=client_id,
            local_train_fn=train_callback,
            network_monitor=self.network_monitor,  # Pass same NetworkMonitor
        )
        
        logger.info(f"FL Client App initialized: {client_id}")
    
    async def run(self):
        """Run the FL client"""
        logger.info(f"\n{'='*60}")
        logger.info(f"FL-QUIC CLIENT: {self.client_id}")
        logger.info(f"Server: {self.server_host}:{self.server_port}")
        logger.info(f"{'='*60}\n")
        
        # Start client
        await self.quic_client.run()


async def main():
    """Main entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="FL-QUIC Client")
    parser.add_argument(
        "--server-host",
        type=str,
        default="localhost",
        help="Server hostname or IP"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=4433,
        help="Server port"
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=None,
        help="Client ID (auto-generated if not provided)"
    )
    parser.add_argument(
        "--jetson",
        action="store_true",
        help="Use Jetson Nano optimized config"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=3,
        help="Local training epochs per round"
    )
    
    args = parser.parse_args()
    
    # Generate client ID if not provided
    if args.client_id is None:
        import socket
        hostname = socket.gethostname()
        args.client_id = f"client_{hostname}_{id(object())}"
    
    # Load configuration
    if args.jetson:
        logger.info("Using Jetson Nano optimized configuration")
        config = get_jetson_config()
    else:
        config = Config()
    
    # Override config from args
    config.model.lora_r = args.lora_rank
    config.training.local_epochs = args.local_epochs
    
    # Create and run app
    app = FLClientApp(
        server_host=args.server_host,
        server_port=args.server_port,
        client_id=args.client_id,
        config=config,
    )
    
    try:
        await app.run()
    except KeyboardInterrupt:
        logger.info("\nClient stopped by user")
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   FL-QUIC-LoRA Federated Learning Client                ║
║   Accelerating FL on Edge Devices via QUIC & LoRA       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
