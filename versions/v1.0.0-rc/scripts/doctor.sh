#!/bin/bash
#
# Controller 1.0.0 RC - Doctor Script
# Diagnoses the Controller environment
#

set -e

CONTROLLER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$CONTROLLER_DIR/.venv"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 Controller 1.0.0 RC Doctor                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

ERRORS=0
WARNINGS=0

# Check Python
echo "📋 Python Check"
echo "────────────────────────────────────────────────────────────────"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "✅ $PYTHON_VERSION"
    
    # Check version
    PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
    PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
    if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
        echo "❌ Python 3.9+ required"
        ((ERRORS++))
    fi
else
    echo "❌ Python 3 not found"
    ((ERRORS++))
fi
echo ""

# Check virtual environment
echo "📦 Virtual Environment"
echo "────────────────────────────────────────────────────────────────"
if [ -d "$VENV_DIR" ]; then
    echo "✅ Virtual environment exists"
    if [ -f "$VENV_DIR/bin/python" ]; then
        echo "✅ Python executable found"
    else
        echo "❌ Python executable not found in venv"
        ((ERRORS++))
    fi
else
    echo "❌ Virtual environment not found"
    echo "   Run: ./scripts/install.sh"
    ((ERRORS++))
fi
echo ""

# Check .env
echo "🔐 Configuration"
echo "────────────────────────────────────────────────────────────────"
if [ -f "$CONTROLLER_DIR/.env" ]; then
    echo "✅ .env file exists"
    
    # Check CONTROLLER_API_KEY
    if grep -q "CONTROLLER_API_KEY=" "$CONTROLLER_DIR/.env" && \
       ! grep -q "CONTROLLER_API_KEY=your-secure-api-key-here" "$CONTROLLER_DIR/.env" && \
       ! grep -q "CONTROLLER_API_KEY=$" "$CONTROLLER_DIR/.env"; then
        echo "✅ CONTROLLER_API_KEY configured"
    else
        echo "❌ CONTROLLER_API_KEY not configured"
        ((ERRORS++))
    fi
    
    # Check optional API keys
    if grep -q "OPENAI_API_KEY=" "$CONTROLLER_DIR/.env" && \
       ! grep -q "OPENAI_API_KEY=sk-" "$CONTROLLER_DIR/.env"; then
        echo "⚠️  OPENAI_API_KEY not configured (optional)"
        ((WARNINGS++))
    else
        echo "✅ OPENAI_API_KEY configured"
    fi
    
    if grep -q "ANTHROPIC_API_KEY=" "$CONTROLLER_DIR/.env" && \
       ! grep -q "ANTHROPIC_API_KEY=sk-" "$CONTROLLER_DIR/.env"; then
        echo "⚠️  ANTHROPIC_API_KEY not configured (optional)"
        ((WARNINGS++))
    else
        echo "✅ ANTHROPIC_API_KEY configured"
    fi
else
    echo "❌ .env file not found"
    ((ERRORS++))
fi
echo ""

# Check runtime directories
echo "📁 Runtime Directories"
echo "────────────────────────────────────────────────────────────────"
for dir in logs cache; do
    if [ -d "$CONTROLLER_DIR/runtime/$dir" ]; then
        echo "✅ runtime/$dir exists"
    else
        echo "⚠️  runtime/$dir missing"
        ((WARNINGS++))
    fi
done
echo ""

# Check if running
echo "🔄 Service Status"
echo "────────────────────────────────────────────────────────────────"
PID_FILE="$CONTROLLER_DIR/runtime/controller.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "✅ Controller is running (PID: $PID)"
        
        # Test health endpoint
        if command -v curl &> /dev/null; then
            if curl -s http://localhost:8080/health > /dev/null 2>&1; then
                echo "✅ Health endpoint responding"
            else
                echo "❌ Health endpoint not responding"
                ((ERRORS++))
            fi
        fi
    else
        echo "⚠️  Stale PID file found"
        ((WARNINGS++))
    fi
else
    echo "⚠️  Controller is not running"
    ((WARNINGS++))
fi
echo ""

# Check dependencies
echo "📥 Dependencies"
echo "────────────────────────────────────────────────────────────────"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    
    REQUIRED_PKGS=("fastapi" "uvicorn" "pydantic" "httpx")
    for pkg in "${REQUIRED_PKGS[@]}"; do
        if python -c "import $pkg" 2>/dev/null; then
            echo "✅ $pkg installed"
        else
            echo "❌ $pkg not installed"
            ((ERRORS++))
        fi
    done
else
    echo "⚠️  Cannot check dependencies (venv missing)"
fi
echo ""

# Summary
echo "════════════════════════════════════════════════════════════════"
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✅ All checks passed! Controller is healthy."
elif [ $ERRORS -eq 0 ]; then
    echo "⚠️  $WARNINGS warning(s) found. Controller should work but review warnings."
else
    echo "❌ $ERRORS error(s) and $WARNINGS warning(s) found."
    echo "   Please fix errors before starting Controller."
fi
echo "════════════════════════════════════════════════════════════════"

exit $ERRORS
