#!/bin/bash
#
# Controller 1.2.0 RC - Start Script
# Starts the Controller service
#

set -e

CONTROLLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$CONTROLLER_DIR/.venv"
PID_FILE="$CONTROLLER_DIR/runtime/controller.pid"
LOG_FILE="$CONTROLLER_DIR/runtime/logs/controller.log"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 Controller 1.2.0 RC Starter                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if already running
echo "🔍 Checking if Controller is already running..."
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "⚠️  Controller is already running (PID: $PID)"
        echo "   Use ./scripts/restart.sh to restart or ./scripts/stop.sh to stop"
        exit 1
    else
        echo "🧹 Cleaning up stale PID file..."
        rm -f "$PID_FILE"
    fi
fi

# Check virtual environment
echo ""
echo "📦 Checking virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Run ./scripts/install.sh first"
    exit 1
fi
echo "✅ Virtual environment found"

# Check .env
echo ""
echo "🔐 Checking configuration..."
if [ ! -f "$CONTROLLER_DIR/.env" ]; then
    echo "❌ .env file not found. Please create it from .env.example"
    exit 1
fi

# Validate API key is set
if ! grep -q "CONTROLLER_API_KEY=" "$CONTROLLER_DIR/.env" || \
   grep -q "CONTROLLER_API_KEY=your-secure-api-key-here" "$CONTROLLER_DIR/.env"; then
    echo "❌ CONTROLLER_API_KEY not configured in .env"
    exit 1
fi
echo "✅ Configuration valid"

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

echo ""
echo "🚀 Starting Controller..."
echo "   Logs: $LOG_FILE"
echo ""

# Start the server
cd "$CONTROLLER_DIR"
nohup python -m controller.main > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait a moment and check if process is still running
sleep 2

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "❌ Controller failed to start. Check logs: $LOG_FILE"
    exit 1
fi

echo "✅ Controller started (PID: $SERVER_PID)"
echo ""

# Wait for health check
echo "⏳ Waiting for health check..."
for i in {1..10}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║              ✅ Controller is Running!                       ║"
        echo "╠══════════════════════════════════════════════════════════════╣"
        echo "║                                                              ║"
        echo "║  PID: $SERVER_PID                                              ║"
        echo "║  URL: http://localhost:8080                                  ║"
        echo "║  Docs: http://localhost:8080/docs                            ║"
        echo "║  Health: http://localhost:8080/health                        ║"
        echo "║                                                              ║"
        echo "║  Status: ./scripts/status.sh                                 ║"
        echo "║  Stop:   ./scripts/stop.sh                                   ║"
        echo "║  Logs:   tail -f $LOG_FILE                                   ║"
        echo "║                                                              ║"
        echo "╚══════════════════════════════════════════════════════════════╝"
        exit 0
    fi
    sleep 1
done

echo ""
echo "⚠️  Controller started but health check failed"
echo "   Check logs: tail -f $LOG_FILE"
exit 1
