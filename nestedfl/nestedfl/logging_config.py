"""
nestedfl: Logging configuration for Flower experiments

Creates timestamped log files for each run in logs/ directory.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: str = "logs") -> str:
    """
    Setup logging to both console and timestamped file.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Path to the created log file
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped filename: dd-mm-yyyy_hh-mm-ss.log
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log_file = log_path / f"flwr_run_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%d/%m/%Y - %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Log startup message
    root_logger.info("=" * 60)
    root_logger.info(f"Log file created: {log_file}")
    root_logger.info(f"Timestamp: {datetime.now().strftime('%d/%m/%Y - %H:%M:%S')}")
    root_logger.info("=" * 60)
    
    return str(log_file)


# Auto-setup logging when module is imported
_log_file = setup_logging()


def get_log_file() -> str:
    """Get path to current log file."""
    return _log_file
