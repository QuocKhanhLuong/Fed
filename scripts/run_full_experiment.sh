#!/bin/bash
#===============================================================================
# FL Full Experiment Script for RTX 4070
# Runs server and multiple clients on the same machine
#
# Usage:
#   ./scripts/run_full_experiment.sh                    # Default: 3 clients, 50 rounds
#   ./scripts/run_full_experiment.sh --clients 5        # 5 clients
#   ./scripts/run_full_experiment.sh --rounds 100       # 100 rounds
#   ./scripts/run_full_experiment.sh --quick            # Quick test: 2 clients, 5 rounds
#
# Author: Research Team
#===============================================================================

set -e

# Get the project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"
echo "Working directory: $(pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
NUM_CLIENTS=3
NUM_ROUNDS=20
LOCAL_EPOCHS=100
DATASET="cifar100"
ALPHA=0.3
SERVER_PORT=4433
LOG_DIR="$PROJECT_ROOT/logs/experiment_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        --rounds)
            NUM_ROUNDS="$2"
            shift 2
            ;;
        --epochs)
            LOCAL_EPOCHS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --quick)
            NUM_CLIENTS=2
            NUM_ROUNDS=5
            LOCAL_EPOCHS=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clients N     Number of clients (default: 3)"
            echo "  --rounds N      Number of FL rounds (default: 50)"
            echo "  --epochs N      Local training epochs (default: 5)"
            echo "  --dataset NAME  Dataset name (default: cifar100)"
            echo "  --alpha FLOAT   Dirichlet alpha for non-IID (default: 0.3)"
            echo "  --quick         Quick test mode (2 clients, 5 rounds)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║   FL Full Experiment - RTX 4070                              ║"
echo "║   Early-Exit + FedDyn + QUIC                                 ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Print configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Clients:      $NUM_CLIENTS"
echo "  Rounds:       $NUM_ROUNDS"
echo "  Local Epochs: $LOCAL_EPOCHS"
echo "  Dataset:      $DATASET"
echo "  Non-IID α:    $ALPHA"
echo "  Log Dir:      $LOG_DIR"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Check CUDA
echo -e "${BLUE}Checking CUDA...${NC}"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}');"
echo ""

# Store PIDs for cleanup
declare -a PIDS=()

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping process $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

#===============================================================================
# Step 1: Generate TLS certificates if not exist
#===============================================================================
if [ ! -f "server.crt" ] || [ ! -f "server.key" ]; then
    echo -e "${YELLOW}Generating TLS certificates...${NC}"
    openssl req -x509 -newkey rsa:2048 \
        -keyout server.key -out server.crt \
        -days 365 -nodes \
        -subj "/CN=localhost" \
        2>/dev/null
    echo -e "${GREEN}✓ Certificates generated${NC}"
else
    echo -e "${GREEN}✓ Certificates exist${NC}"
fi
echo ""

#===============================================================================
# Step 1.5: Check/Start Redis (optional but recommended)
#===============================================================================
echo -e "${BLUE}Checking Redis...${NC}"
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}✓ Redis is running${NC}"
    else
        echo -e "${YELLOW}Starting Redis server...${NC}"
        if command -v redis-server &> /dev/null; then
            redis-server --daemonize yes --port 6379 2>/dev/null || true
            sleep 1
            if redis-cli ping &> /dev/null; then
                echo -e "${GREEN}✓ Redis started${NC}"
            else
                echo -e "${YELLOW}⚠ Redis failed to start - using in-memory fallback${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ redis-server not found - using in-memory fallback${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ redis-cli not found - using in-memory fallback (OK for testing)${NC}"
fi
echo ""

#===============================================================================
# Step 2: Start Server
#===============================================================================
echo -e "${BLUE}Starting FL Server...${NC}"

python3 server/app_server.py \
    --host 0.0.0.0 \
    --port $SERVER_PORT \
    --rounds $NUM_ROUNDS \
    --min-clients $NUM_CLIENTS \
    --local-epochs $LOCAL_EPOCHS \
    --cert server.crt \
    --key server.key \
    --high-performance \
    > "$LOG_DIR/server.log" 2>&1 &

SERVER_PID=$!
PIDS+=($SERVER_PID)
echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"

# Wait for server to initialize
echo "Waiting for server initialization..."
sleep 5

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}✗ Server failed to start. Check $LOG_DIR/server.log${NC}"
    tail -20 "$LOG_DIR/server.log"
    exit 1
fi
echo ""

#===============================================================================
# Step 3: Start Clients
#===============================================================================
echo -e "${BLUE}Starting $NUM_CLIENTS FL Clients...${NC}"

for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    CLIENT_ID="client_$i"
    
    # Stagger client starts to avoid race conditions
    sleep 2
    
    python3 client/app_client.py \
        --server-host localhost \
        --server-port $SERVER_PORT \
        --client-id $CLIENT_ID \
        --dataset $DATASET \
        --alpha $ALPHA \
        --local-epochs $LOCAL_EPOCHS \
        > "$LOG_DIR/${CLIENT_ID}.log" 2>&1 &
    
    CLIENT_PID=$!
    PIDS+=($CLIENT_PID)
    echo -e "${GREEN}  ✓ Started $CLIENT_ID (PID: $CLIENT_PID)${NC}"
done
echo ""

#===============================================================================
# Step 4: Monitor Progress
#===============================================================================
echo -e "${BLUE}Monitoring experiment...${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Monitor server log for progress
tail -f "$LOG_DIR/server.log" | while read line; do
    # Print round updates
    if echo "$line" | grep -q "Round"; then
        echo -e "${YELLOW}$line${NC}"
    fi
    
    # Print accuracy updates
    if echo "$line" | grep -q -i "accuracy"; then
        echo -e "${GREEN}$line${NC}"
    fi
    
    # Check for completion
    if echo "$line" | grep -q "Training complete"; then
        echo -e "\n${GREEN}═══════════════════════════════════════${NC}"
        echo -e "${GREEN}✓ EXPERIMENT COMPLETE!${NC}"
        echo -e "${GREEN}═══════════════════════════════════════${NC}"
        break
    fi
    
    # Check for errors
    if echo "$line" | grep -q -i "error\|exception\|failed"; then
        echo -e "${RED}$line${NC}"
    fi
done

#===============================================================================
# Step 5: Collect Results
#===============================================================================
echo ""
echo -e "${BLUE}Collecting results...${NC}"

# Extract final metrics from logs
echo ""
echo "════════════════════════════════════════════════════════════"
echo "EXPERIMENT SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo ""

# Parse server log for final results
if [ -f "$LOG_DIR/server.log" ]; then
    echo "Server log summary:"
    grep -E "Final|Accuracy|accuracy|Round.*complete" "$LOG_DIR/server.log" | tail -10 || echo "  No summary found"
fi

echo ""
echo "Logs saved to: $LOG_DIR"
echo ""
echo -e "${GREEN}Done!${NC}"
