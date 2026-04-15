#!/bin/bash
#
# Controller 1.2.0 RC - Restart Script
# Restarts the Controller service
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                Controller 1.2.0 RC Restarter                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "🔄 Restarting Controller..."
echo ""

# Stop if running
$SCRIPT_DIR/stop.sh || true

# Wait a moment
sleep 2

# Start
$SCRIPT_DIR/start.sh
