#!/bin/bash
#
# Controller 1.1.0 RC - Installation Script
# Sets up the Controller environment and dependencies
#

set -e

CONTROLLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$CONTROLLER_DIR/.venv"
RUNTIME_DIR="$CONTROLLER_DIR/runtime"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                Controller 1.1.0 RC Installer                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python 3.9+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python $PYTHON_VERSION found"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment already exists. Removing..."
    rm -rf "$VENV_DIR"
fi
python3 -m venv "$VENV_DIR"
echo "✅ Virtual environment created at $VENV_DIR"

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install -r "$CONTROLLER_DIR/requirements.txt"
echo "✅ Dependencies installed"

# Create runtime directories
echo ""
echo "📁 Creating runtime directories..."
mkdir -p "$RUNTIME_DIR"/{logs,cache}
echo "✅ Runtime directories created"

# Check for .env file
echo ""
echo "🔐 Checking configuration..."
if [ ! -f "$CONTROLLER_DIR/.env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    if [ -f "$CONTROLLER_DIR/.env.example" ]; then
        cp "$CONTROLLER_DIR/.env.example" "$CONTROLLER_DIR/.env"
        echo "✅ Created .env from template"
        echo "⚠️  IMPORTANT: Edit $CONTROLLER_DIR/.env and set your API keys!"
    else
        echo "❌ .env.example not found. Please create .env manually."
    fi
else
    echo "✅ .env file exists"
fi

# Make scripts executable
echo ""
echo "🔧 Setting up scripts..."
chmod +x "$CONTROLLER_DIR/scripts/"*.sh
echo "✅ Scripts made executable"

# Verify installation
echo ""
echo "🔍 Verifying installation..."
cd "$CONTROLLER_DIR"
python -c "from controller.main import app; print('✅ Controller imports successfully')" 2>/dev/null || {
    echo "❌ Controller import failed"
    exit 1
}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              ✅ Installation Complete!                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Next steps:                                                 ║"
echo "║  1. Edit .env and configure your API keys                    ║"
echo "║  2. Run: ./scripts/start.sh                                  ║"
echo "║  3. Test: curl http://localhost:8080/health                  ║"
echo "║                                                              ║"
echo "║  Documentation: ./README.md                                  ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
