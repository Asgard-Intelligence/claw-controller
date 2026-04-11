#!/bin/bash
#
# Controller 1.0.0 RC - Status Script
# Checks the status of the Controller service
#

set -e

CONTROLLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$CONTROLLER_DIR/runtime/controller.pid"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 Controller 1.0.0 RC Status                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "❌ Controller is NOT running"
    echo ""
    echo "   Start with: ./scripts/start.sh"
    exit 1
fi

PID=$(cat "$PID_FILE" 2>/dev/null)

if [ -z "$PID" ]; then
    echo "❌ Controller is NOT running (PID file empty)"
    rm -f "$PID_FILE"
    exit 1
fi

# Check if process is running
if ! kill -0 "$PID" 2>/dev/null; then
    echo "❌ Controller is NOT running (stale PID file)"
    rm -f "$PID_FILE"
    exit 1
fi

echo "✅ Controller is RUNNING"
echo ""
echo "   PID: $PID"
echo "   Config: $CONTROLLER_DIR/.env"
echo ""

# Check health endpoint
echo "🔍 Checking health endpoint..."
if command -v curl &> /dev/null; then
    HEALTH=$(curl -s http://localhost:8080/health 2>/dev/null || echo '{"status": "unknown"}')
    echo "   Health: $(echo "$HEALTH" | grep -o '"status": "[^"]*"' | cut -d'"' -f4)"
    echo "   Version: $(echo "$HEALTH" | grep -o '"version": "[^"]*"' | cut -d'"' -f4)"
else
    echo "   (curl not available)"
fi

echo ""
echo "   Endpoints:"
echo "     - http://localhost:8080/         (Root)"
echo "     - http://localhost:8080/health   (Health)"
echo "     - http://localhost:8080/docs     (API Docs)"
echo "     - http://localhost:8080/v1/models (Models)"
echo ""
echo "   Commands:"
echo "     - ./scripts/stop.sh    (Stop)"
echo "     - ./scripts/restart.sh (Restart)"
echo "     - ./scripts/doctor.sh  (Diagnose)"
