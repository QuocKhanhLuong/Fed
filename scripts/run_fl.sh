#!/bin/bash
# =============================================================================
# Full Federated Learning Run Script
# Starts server and multiple clients with Nested Learning strategy
# =============================================================================

set -e

# Configuration
NUM_CLIENTS=${NUM_CLIENTS:-3}
NUM_ROUNDS=${NUM_ROUNDS:-10}
SERVER_HOST="localhost"
SERVER_PORT=4433
STRATEGY="nested_feddyn"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

cd "$(dirname "$0")/.."

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Nested Federated Learning - Full FL Simulation         ║"
echo "║   Strategy: ${STRATEGY}                               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  - Clients: $NUM_CLIENTS"
echo "  - Rounds: $NUM_ROUNDS"
echo "  - Server: $SERVER_HOST:$SERVER_PORT"
echo "  - Strategy: $STRATEGY"
echo ""

# Check if certificates exist
if [ ! -f "server.crt" ] || [ ! -f "server.key" ]; then
    echo -e "${YELLOW}⚠️  Generating self-signed certificates for development...${NC}"
    openssl req -x509 -newkey rsa:2048 -keyout server.key -out server.crt \
        -days 365 -nodes -subj "/CN=localhost" 2>/dev/null
    echo -e "${GREEN}✓ Certificates generated${NC}"
fi

# Kill any existing processes on port
echo -e "\n${YELLOW}Cleaning up existing processes...${NC}"
pkill -f "app_server.py" 2>/dev/null || true
pkill -f "app_client.py" 2>/dev/null || true
sleep 1

# Create logs directory
mkdir -p logs

# Start server
echo -e "\n${GREEN}Starting FL Server...${NC}"
python server/app_server.py \
    --min-clients $NUM_CLIENTS \
    --rounds $NUM_ROUNDS \
    --high-performance \
    --cert server.crt \
    --key server.key \
    > logs/server.log 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID"

# Wait for server to start
echo -e "${YELLOW}Waiting for server to initialize...${NC}"
sleep 5

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}❌ Server failed to start! Check logs/server.log${NC}"
    cat logs/server.log
    exit 1
fi
echo -e "${GREEN}✓ Server is running${NC}"

# Start clients
echo -e "\n${GREEN}Starting $NUM_CLIENTS clients...${NC}"
CLIENT_PIDS=()

for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    echo "  Starting Client $i..."
    python client/app_client.py \
        --server-host $SERVER_HOST \
        --server-port $SERVER_PORT \
        --client-id "client_$i" \
        > logs/client_$i.log 2>&1 &
    CLIENT_PIDS+=($!)
    echo "    Client $i PID: ${CLIENT_PIDS[$i]}"
    sleep 2
done

echo -e "\n${GREEN}✓ All clients started${NC}"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    for pid in "${CLIENT_PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    kill $SERVER_PID 2>/dev/null || true
    echo -e "${GREEN}✓ All processes stopped${NC}"
}
trap cleanup EXIT

# Monitor progress
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${BLUE}FL Training in progress...${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Logs:"
echo "  - Server: logs/server.log"
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    echo "  - Client $i: logs/client_$i.log"
done
echo ""
echo -e "${YELLOW}Monitoring server log (Ctrl+C to stop):${NC}"
echo ""

# Tail server log
tail -f logs/server.log | while read line; do
    echo "$line"
    # Check if training completed
    if echo "$line" | grep -q "FL training completed"; then
        echo -e "\n${GREEN}✅ FL Training Completed!${NC}"
        break
    fi
done

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}FL Simulation Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Results saved in:"
echo "  - Checkpoints: ./checkpoints/"
echo "  - Logs: ./logs/"
echo "  - Results: ./results/"
