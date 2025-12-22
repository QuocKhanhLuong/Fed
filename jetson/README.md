# Jetson Nano Client Setup
# ========================
# 
# Quick Start:
#   ./jetson/run_client.sh --server <SERVER_IP>
#
# This will automatically:
# - Create Python virtual environment
# - Install PyTorch for Jetson (1.12.0)
# - Install all FL dependencies
# - Start the FL client
#
# Requirements:
# - JetPack 4.6.x (L4T R32.7.x)
# - 4GB RAM recommended
#
# Options:
#   --server, -s     Server IP address
#   --port, -p       Server port (default: 4433)
#   --client-id, -c  Client ID (auto-generated)
#   --dataset, -d    Dataset (default: cifar100)
#   --epochs, -e     Local epochs (default: 3)
#
# Example:
#   ./jetson/run_client.sh --server 192.168.1.100 --epochs 5
