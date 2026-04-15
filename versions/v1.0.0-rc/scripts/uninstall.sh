#!/bin/bash
#
# Controller 1.0.0 RC - Uninstall Script
# Removes the Controller installation
#

set -e

CONTROLLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║               Controller 1.0.0 RC Uninstaller                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "⚠️  This will remove Controller and all its data."
echo ""
read -p "Are you sure? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Uninstall cancelled."
    exit 0
fi

echo ""
echo "🛑 Stopping Controller..."
"$CONTROLLER_DIR/scripts/stop.sh" 2>/dev/null || true

echo ""
echo "🗑️  Removing virtual environment..."
rm -rf "$CONTROLLER_DIR/.venv"

echo "🗑️  Removing runtime data..."
rm -rf "$CONTROLLER_DIR/runtime"

echo ""
echo "✅ Controller uninstalled."
echo ""
echo "Note: Configuration files (.env) and logs were preserved."
echo "      Remove manually if needed: rm -rf $CONTROLLER_DIR"
