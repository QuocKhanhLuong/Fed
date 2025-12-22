#!/bin/bash
# =============================================================================
# FL-QUIC Client for Jetson Nano - One-Click Setup & Run
# =============================================================================
# Usage: 
#   ./jetson/run_client.sh --server <SERVER_IP>
#
# This script will:
#   1. Create Python virtual environment (first run only)
#   2. Install all dependencies including PyTorch for Jetson
#   3. Run the FL client connecting to the specified server
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/venv_jetson"
PYTORCH_WHEEL="torch-1.12.0a0+git67ece03-cp38-cp38-linux_aarch64.whl"
PYTORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/$PYTORCH_WHEEL"

# Default values
SERVER_HOST="localhost"
SERVER_PORT=4433
CLIENT_ID=""
DATASET="cifar100"
LOCAL_EPOCHS=3
ALPHA=0.5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --server|-s)
            SERVER_HOST="$2"
            shift 2
            ;;
        --port|-p)
            SERVER_PORT="$2"
            shift 2
            ;;
        --client-id|-c)
            CLIENT_ID="$2"
            shift 2
            ;;
        --dataset|-d)
            DATASET="$2"
            shift 2
            ;;
        --epochs|-e)
            LOCAL_EPOCHS="$2"
            shift 2
            ;;
        --alpha|-a)
            ALPHA="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --server, -s     Server IP address (default: localhost)"
            echo "  --port, -p       Server port (default: 4433)"
            echo "  --client-id, -c  Client ID (auto-generated if not provided)"
            echo "  --dataset, -d    Dataset name (default: cifar100)"
            echo "  --epochs, -e     Local epochs per round (default: 3)"
            echo "  --alpha, -a      Dirichlet alpha (default: 0.5)"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Header
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║   FL-QUIC Client for Jetson Nano                        ║"
echo "║   Nested Early-Exit Federated Learning                  ║"
echo "║                                                          ║"
╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running on Jetson
check_jetson() {
    if [ -f /etc/nv_tegra_release ]; then
        echo -e "${GREEN}✓ Running on Jetson Nano${NC}"
        cat /etc/nv_tegra_release | head -n 1
    else
        echo -e "${YELLOW}⚠ Not running on Jetson - using standard setup${NC}"
        return 1
    fi
    return 0
}

# Setup virtual environment
setup_venv() {
    if [ -d "$VENV_DIR" ]; then
        echo -e "${GREEN}✓ Virtual environment exists${NC}"
    else
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    fi
    
    # Activate venv
    source "$VENV_DIR/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
}

# Install PyTorch for Jetson
install_pytorch_jetson() {
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        echo -e "${GREEN}✓ PyTorch already installed: $TORCH_VERSION${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}Installing PyTorch for Jetson Nano...${NC}"
    
    # Install prerequisites
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-pip libopenblas-dev
    
    pip install --upgrade pip
    pip install numpy==1.19.5
    
    # Download and install PyTorch wheel
    if [ ! -f "/tmp/$PYTORCH_WHEEL" ]; then
        echo "Downloading PyTorch wheel (this may take a while)..."
        wget -q --show-progress -O "/tmp/$PYTORCH_WHEEL" "$PYTORCH_URL" || {
            echo -e "${RED}Failed to download PyTorch wheel${NC}"
            echo "Trying pip install..."
            pip install torch torchvision
            return 0
        }
    fi
    
    pip install "/tmp/$PYTORCH_WHEEL"
    echo -e "${GREEN}✓ PyTorch installed${NC}"
}

# Install dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    
    pip install --upgrade pip -q
    
    # Core dependencies (Jetson compatible versions)
    pip install numpy -q
    pip install lz4 -q
    pip install tqdm -q
    pip install aioquic -q || {
        echo -e "${YELLOW}Installing OpenSSL for aioquic...${NC}"
        sudo apt-get install -y -qq libssl-dev
        pip install aioquic -q
    }
    pip install flwr -q
    pip install timm -q
    pip install torchvision -q || echo "torchvision may need manual install"
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Verify installation
verify_installation() {
    echo -e "${YELLOW}Verifying installation...${NC}"
    
    python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" || {
        echo -e "${RED}✗ PyTorch not working${NC}"
        return 1
    }
    
    python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')" || true
    python3 -c "import aioquic; print('  aioquic: OK')" || echo "  aioquic: FAILED"
    python3 -c "import lz4; print('  lz4: OK')" || echo "  lz4: FAILED"
    python3 -c "import flwr; print('  flwr: OK')" || echo "  flwr: FAILED"
    python3 -c "import timm; print('  timm: OK')" || echo "  timm: FAILED"
    
    echo -e "${GREEN}✓ Installation verified${NC}"
}

# Generate client ID if not provided
generate_client_id() {
    if [ -z "$CLIENT_ID" ]; then
        HOSTNAME=$(hostname)
        RANDOM_ID=$RANDOM
        CLIENT_ID="jetson_${HOSTNAME}_${RANDOM_ID}"
    fi
    echo -e "${BLUE}Client ID: $CLIENT_ID${NC}"
}

# Run FL client
run_client() {
    echo ""
    echo -e "${GREEN}Starting FL Client...${NC}"
    echo -e "  Server: ${BLUE}$SERVER_HOST:$SERVER_PORT${NC}"
    echo -e "  Dataset: ${BLUE}$DATASET${NC}"
    echo -e "  Epochs: ${BLUE}$LOCAL_EPOCHS${NC}"
    echo ""
    
    cd "$PROJECT_DIR"
    
    # Set Jetson optimizations
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    export OMP_NUM_THREADS=4
    
    python3 -m client.app_client \
        --server-host "$SERVER_HOST" \
        --server-port "$SERVER_PORT" \
        --client-id "$CLIENT_ID" \
        --dataset "$DATASET" \
        --local-epochs "$LOCAL_EPOCHS" \
        --alpha "$ALPHA" \
        --jetson
}

# Main execution
main() {
    cd "$PROJECT_DIR"
    
    IS_JETSON=false
    check_jetson && IS_JETSON=true
    
    setup_venv
    
    if [ "$IS_JETSON" = true ]; then
        install_pytorch_jetson
    fi
    
    install_dependencies
    verify_installation
    generate_client_id
    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Setup Complete! Starting FL Client...${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    
    run_client
}

# Run
main
