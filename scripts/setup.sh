#!/bin/bash
# Setup script for FL-QUIC-LoRA project
# Run this on both server and clients

set -e

echo "=========================================="
echo "FL-QUIC-LoRA Setup Script"
echo "=========================================="

# Detect platform
if [[ $(uname -m) == "aarch64" ]]; then
    PLATFORM="jetson"
    echo "Platform: NVIDIA Jetson Nano (ARM64)"
else
    PLATFORM="x86"
    echo "Platform: x86_64 Linux"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8+ is required"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."

if [[ $PLATFORM == "jetson" ]]; then
    echo "Installing Jetson-specific packages..."
    
    # Check if PyTorch is already installed (from NVIDIA)
    if python3 -c "import torch" 2>/dev/null; then
        echo "✓ PyTorch already installed"
    else
        echo "WARNING: PyTorch not found. Please install from NVIDIA:"
        echo "https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    fi
    
    # Install other dependencies (excluding torch/torchvision)
    pip install aioquic cryptography flwr lz4 numpy tqdm transformers peft tensorboard
else
    echo "Installing standard packages..."
    pip install -r requirements.txt
fi

# Verify installation
echo ""
echo "Verifying installation..."

python3 -c "import aioquic; print('✓ aioquic')" || echo "✗ aioquic FAILED"
python3 -c "import lz4; print('✓ lz4')" || echo "✗ lz4 FAILED"
python3 -c "import numpy; print('✓ numpy')" || echo "✗ numpy FAILED"
python3 -c "import flwr; print('✓ flwr')" || echo "✗ flwr FAILED"

# Test transport layer
echo ""
echo "Testing transport layer..."
python3 test_transport.py

# Generate self-signed certificates for development
echo ""
read -p "Generate self-signed TLS certificates for testing? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Generating certificates..."
    openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt \
        -days 365 -nodes -subj "/CN=localhost"
    echo "✓ Certificates generated: server.crt, server.key"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
if [[ $PLATFORM == "jetson" ]]; then
    echo "  On Jetson Nano (Client):"
    echo "    source venv/bin/activate"
    echo "    cd client"
    echo "    python quic_client.py --server-host <SERVER_IP>"
else
    echo "  On Server:"
    echo "    source venv/bin/activate"
    echo "    cd server"
    echo "    python quic_server.py"
fi
echo ""
echo "For more information, see README.md"
