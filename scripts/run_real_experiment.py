#!/usr/bin/env python3
"""
FL Full Experiment Runner with Real Training
Runs actual FL training with server and clients, collects metrics for paper.

Usage:
    # Quick test
    python scripts/run_real_experiment.py --quick
    
    # Full experiment (3 clients, 50 rounds)
    python scripts/run_real_experiment.py
    
    # Custom settings
    python scripts/run_real_experiment.py --clients 5 --rounds 100 --alpha 0.1

Output:
    - results/exp_YYYYMMDD_HHMMSS/
        - config.json
        - metrics.json
        - tables.md (IEEE format)
        - figures/ (accuracy curves, etc.)

Author: Research Team
"""

import sys
import os
import json
import time
import signal
import asyncio
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import threading

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    num_clients: int = 3
    num_rounds: int = 50
    local_epochs: int = 5
    dataset: str = "cifar100"
    alpha: float = 0.3  # Dirichlet concentration
    batch_size: int = 64
    learning_rate: float = 1e-3
    exit_weights: List[float] = None
    use_feddyn: bool = True
    feddyn_alpha: float = 0.01
    use_compression: bool = True
    server_port: int = 4433
    
    def __post_init__(self):
        if self.exit_weights is None:
            self.exit_weights = [0.3, 0.3, 0.4]


@dataclass
class RoundMetrics:
    """Metrics for a single FL round."""
    round_num: int
    global_accuracy: float
    global_loss: float
    client_accuracies: List[float]
    exit_distribution: List[float]  # [exit1%, exit2%, exit3%]
    bytes_sent: int
    bytes_received: int
    round_time_s: float
    avg_exit: float


class RealExperimentRunner:
    """
    Run actual FL training experiment.
    """
    
    def __init__(self, config: ExperimentConfig, output_dir: str = "./results"):
        self.config = config
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"exp_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.round_metrics: List[RoundMetrics] = []
        self.processes: List[subprocess.Popen] = []
        
        logger.info(f"Experiment output: {self.output_dir}")
    
    def save_config(self):
        """Save experiment configuration."""
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        logger.info(f"Config saved: {config_path}")
    
    def generate_certs(self):
        """Generate TLS certificates if needed."""
        cert_file = Path("server.crt")
        key_file = Path("server.key")
        
        if cert_file.exists() and key_file.exists():
            logger.info("✓ TLS certificates exist")
            return str(cert_file), str(key_file)
        
        logger.info("Generating TLS certificates...")
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", "server.key", "-out", "server.crt",
            "-days", "365", "-nodes",
            "-subj", "/CN=localhost"
        ], capture_output=True)
        
        logger.info("✓ Certificates generated")
        return str(cert_file), str(key_file)
    
    def start_server(self, cert_file: str, key_file: str) -> subprocess.Popen:
        """Start FL server process."""
        cmd = [
            sys.executable, "server/app_server.py",
            "--host", "0.0.0.0",
            "--port", str(self.config.server_port),
            "--rounds", str(self.config.num_rounds),
            "--min-clients", str(self.config.num_clients),
            "--local-epochs", str(self.config.local_epochs),
            "--cert", cert_file,
            "--key", key_file,
            "--high-performance",
        ]
        
        log_file = open(self.output_dir / "server.log", 'w')
        
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if os.name != 'nt' else None,
        )
        
        self.processes.append(process)
        logger.info(f"✓ Server started (PID: {process.pid})")
        
        return process
    
    def start_client(self, client_id: int) -> subprocess.Popen:
        """Start a FL client process."""
        cmd = [
            sys.executable, "client/app_client.py",
            "--server-host", "localhost",
            "--server-port", str(self.config.server_port),
            "--client-id", f"client_{client_id}",
            "--dataset", self.config.dataset,
            "--alpha", str(self.config.alpha),
            "--local-epochs", str(self.config.local_epochs),
        ]
        
        log_file = open(self.output_dir / f"client_{client_id}.log", 'w')
        
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if os.name != 'nt' else None,
        )
        
        self.processes.append(process)
        logger.info(f"  ✓ Client {client_id} started (PID: {process.pid})")
        
        return process
    
    def cleanup(self):
        """Stop all processes."""
        logger.info("Cleaning up processes...")
        for p in self.processes:
            try:
                if p.poll() is None:  # Still running
                    if os.name != 'nt':
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    else:
                        p.terminate()
                    p.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error stopping process: {e}")
        
        self.processes.clear()
        logger.info("✓ Cleanup complete")
    
    def monitor_progress(self, server_process: subprocess.Popen):
        """Monitor server log for completion."""
        log_path = self.output_dir / "server.log"
        
        # Wait for log file
        for _ in range(30):
            if log_path.exists():
                break
            time.sleep(1)
        
        if not log_path.exists():
            logger.error("Server log not found")
            return
        
        # Monitor log
        with open(log_path, 'r') as f:
            while server_process.poll() is None:
                line = f.readline()
                if line:
                    # Log important events
                    if "Round" in line and "complete" in line.lower():
                        logger.info(line.strip())
                    elif "accuracy" in line.lower():
                        logger.info(line.strip())
                    elif "error" in line.lower():
                        logger.warning(line.strip())
                else:
                    time.sleep(0.5)
        
        logger.info("Server process ended")
    
    def parse_metrics_from_logs(self):
        """Parse metrics from server log."""
        log_path = self.output_dir / "server.log"
        
        if not log_path.exists():
            logger.warning("No server log to parse")
            return
        
        # TODO: Parse actual metrics from log
        # This depends on what the server logs
        
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract final accuracy
        import re
        acc_matches = re.findall(r'accuracy[:\s]+(\d+\.?\d*)%?', content, re.IGNORECASE)
        if acc_matches:
            final_acc = float(acc_matches[-1])
            if final_acc > 1:  # Convert from percentage
                final_acc /= 100
            logger.info(f"Final accuracy: {final_acc:.2%}")
    
    def save_metrics(self):
        """Save collected metrics."""
        metrics_path = self.output_dir / "metrics.json"
        
        metrics = {
            'config': asdict(self.config),
            'rounds': [asdict(m) for m in self.round_metrics],
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved: {metrics_path}")
    
    def generate_tables(self):
        """Generate IEEE-format result tables."""
        tables_path = self.output_dir / "tables.md"
        
        with open(tables_path, 'w') as f:
            f.write("# Experiment Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- Clients: {self.config.num_clients}\n")
            f.write(f"- Rounds: {self.config.num_rounds}\n")
            f.write(f"- Dataset: {self.config.dataset}\n")
            f.write(f"- Non-IID α: {self.config.alpha}\n\n")
            
            f.write("## Results\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write("| Final Accuracy | TODO |\n")
            f.write("| Communication (MB) | TODO |\n")
            f.write("| Convergence Round | TODO |\n")
            f.write("\n")
            
            f.write("## Exit Distribution\n\n")
            f.write("| Threshold | Exit 1 | Exit 2 | Exit 3 | Accuracy |\n")
            f.write("|-----------|--------|--------|--------|----------|\n")
            f.write("| τ=0.5 | TODO | TODO | TODO | TODO |\n")
            f.write("| τ=0.7 | TODO | TODO | TODO | TODO |\n")
            f.write("| τ=0.9 | TODO | TODO | TODO | TODO |\n")
        
        logger.info(f"Tables saved: {tables_path}")
    
    def run(self):
        """Run the full experiment."""
        print("\n" + "=" * 60)
        print("FL FULL EXPERIMENT - RTX 4070")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Clients:    {self.config.num_clients}")
        print(f"  Rounds:     {self.config.num_rounds}")
        print(f"  Dataset:    {self.config.dataset}")
        print(f"  Non-IID α:  {self.config.alpha}")
        print(f"  Output:     {self.output_dir}")
        print("=" * 60 + "\n")
        
        try:
            # Step 1: Save config
            self.save_config()
            
            # Step 2: Generate certs
            cert_file, key_file = self.generate_certs()
            
            # Step 3: Start server
            logger.info("Starting FL Server...")
            server_proc = self.start_server(cert_file, key_file)
            time.sleep(5)  # Wait for initialization
            
            if server_proc.poll() is not None:
                logger.error("Server failed to start!")
                with open(self.output_dir / "server.log", 'r') as f:
                    print(f.read()[-2000:])  # Last 2000 chars
                return
            
            # Step 4: Start clients
            logger.info(f"Starting {self.config.num_clients} clients...")
            for i in range(self.config.num_clients):
                self.start_client(i)
                time.sleep(2)  # Stagger starts
            
            # Step 5: Monitor progress
            logger.info("\nMonitoring experiment (Ctrl+C to stop)...\n")
            self.monitor_progress(server_proc)
            
            # Step 6: Collect results
            logger.info("\nCollecting results...")
            self.parse_metrics_from_logs()
            self.save_metrics()
            self.generate_tables()
            
            print("\n" + "=" * 60)
            print("✓ EXPERIMENT COMPLETE")
            print(f"  Results: {self.output_dir}")
            print("=" * 60 + "\n")
            
        except KeyboardInterrupt:
            logger.info("\nExperiment interrupted by user")
        
        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="FL Full Experiment Runner")
    
    parser.add_argument("--clients", type=int, default=3,
                        help="Number of FL clients")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Number of FL rounds")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Local training epochs")
    parser.add_argument("--dataset", type=str, default="cifar100",
                        help="Dataset name")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Dirichlet alpha for non-IID")
    parser.add_argument("--output", type=str, default="./results",
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (2 clients, 5 rounds)")
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        args.clients = 2
        args.rounds = 5
        args.epochs = 1
    
    config = ExperimentConfig(
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        dataset=args.dataset,
        alpha=args.alpha,
    )
    
    runner = RealExperimentRunner(config, args.output)
    runner.run()


if __name__ == "__main__":
    main()
