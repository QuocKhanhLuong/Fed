"""
FL-QUIC Server Application
Main entry point for Federated Learning server with QUIC transport

Integrates:
- QUIC Server (transport)
- Flower Strategy (aggregation)
- Model coordination

Author: Research Team - FL-QUIC-LoRA Project
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from server.quic_server import FLQuicServer
from server.fl_strategy import create_strategy
from utils.config import Config, get_server_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FLServerApp:
    """
    Complete FL Server Application.
    Combines QUIC transport with Flower FL strategy.
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        num_rounds: int,
        min_clients: int,
        config: Config,
        cert_file: Optional[str] = None,
        key_file: Optional[str] = None,
    ):
        """
        Initialize FL server application.
        
        Args:
            host: Server host address
            port: Server port
            num_rounds: Number of FL rounds
            min_clients: Minimum clients per round
            config: Configuration object
            cert_file: TLS certificate file (optional)
            key_file: TLS key file (optional)
        """
        self.host = host
        self.port = port
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.config = config
        
        # Create Flower strategy
        logger.info("Creating FL strategy...")
        self.strategy = create_strategy(
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
            local_epochs=config.training.local_epochs,
        )
        
        # Create QUIC server
        logger.info("Creating QUIC server...")
        self.quic_server = FLQuicServer(
            host=host,
            port=port,
            cert_file=cert_file,
            key_file=key_file,
            num_rounds=num_rounds,
            min_clients=min_clients,
            min_available_clients=min_clients,
        )
        
        logger.info("FL Server App initialized")
    
    async def run(self):
        """Run the FL server"""
        logger.info(f"\n{'='*60}")
        logger.info(f"FL-QUIC SERVER")
        logger.info(f"Address: {self.host}:{self.port}")
        logger.info(f"Rounds: {self.num_rounds}")
        logger.info(f"Min clients: {self.min_clients}")
        logger.info(f"{'='*60}\n")
        
        # Start server
        await self.quic_server.start()


async def main():
    """Main entry point"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="FL-QUIC Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4433,
        help="Server port"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of FL rounds"
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum clients per round"
    )
    parser.add_argument(
        "--cert",
        type=str,
        default=None,
        help="Path to TLS certificate file"
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Path to TLS key file"
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=3,
        help="Local training epochs per round"
    )
    parser.add_argument(
        "--high-performance",
        action="store_true",
        help="Use high-performance server config"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.high_performance:
        logger.info("Using high-performance server configuration")
        config = get_server_config()
    else:
        config = Config()
    
    # Override config from args
    config.training.local_epochs = args.local_epochs
    config.federated.num_rounds = args.rounds
    config.federated.min_clients = args.min_clients
    
    # TLS certificate warning
    if args.cert is None or args.key is None:
        logger.warning("⚠️  No TLS certificates provided - using self-signed (dev only)")
        logger.warning("   For production, generate certificates with:")
        logger.warning("   openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes")
    
    # Create and run app
    app = FLServerApp(
        host=args.host,
        port=args.port,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        config=config,
        cert_file=args.cert,
        key_file=args.key,
    )
    
    try:
        await app.run()
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
        app.quic_server.shutdown()
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   FL-QUIC-LoRA Federated Learning Server                ║
║   Accelerating FL on Edge Devices via QUIC & LoRA       ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Display system info
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    asyncio.run(main())
