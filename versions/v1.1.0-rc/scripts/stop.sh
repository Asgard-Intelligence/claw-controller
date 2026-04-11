#!/bin/bash
#
# Controller 1.1.0 RC - Stop Script
# Stops the Controller service
#

set -e

CONTROLLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$CONTROLLER_DIR/runtime/controller.pid"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  Controller 1.1.0 RC Stopper                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "⚠️  Controller is not running (no PID file found)"
    exit 0
fi

PID=$(cat "$PID_FILE" 2>/dev/null)

if [ -z "$PID" ]; then
    echo "⚠️  PID file is empty. Removing..."
    rm -f "$PID_FILE"
    exit 0
fi

# Check if process is running
if ! kill -0 "$PID" 2>/dev/null; then
    echo "⚠️  Controller is not running (PID: $PID not found)"
    rm -f "$PID_FILE"
    exit 0
fi

echo "🛑 Stopping Controller (PID: $PID)..."

# Try graceful shutdown first
kill "$PID" 2>/dev/null || true

# Wait for process to stop
for i in {1..10}; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "✅ Controller stopped gracefully"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
echo "⚠️  Graceful shutdown failed. Force stopping..."
kill -9 "$PID" 2>/dev/null || true
sleep 1

if ! kill -0 "$PID" 2>/dev/null; then
    echo "✅ Controller stopped (force)"
    rm -f "$PID_FILE"
else
    echo "❌ Failed to stop Controller"
    exit 1
fi
