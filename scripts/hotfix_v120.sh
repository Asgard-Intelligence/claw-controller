#!/bin/bash
# Hotfix script for Controller v1.2.0
# Applies minimal changes to fix identity drift and context loss

echo "=========================================="
echo "Controller v1.2.0 Hotfix Script"
echo "=========================================="
echo ""

# Check if running in correct directory
if [ ! -f "controller/core/config.py" ]; then
    echo "❌ Error: Must run from Controller root directory"
    echo "   Looking for: controller/core/config.py"
    exit 1
fi

BACKUP_DIR=".backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup in: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Backup current config
cp controller/core/config.py "$BACKUP_DIR/"

echo ""
echo "Applying hotfix..."
echo ""

# ============================================
# HOTFIX 1: Disable Gemma for agent workloads
# ============================================
echo "[1/4] Disabling Gemma for agent workloads..."

# Add to .env if not exists
if ! grep -q "OPENAI_MODEL_BASIC=" .env 2>/dev/null; then
    echo "" >> .env
    echo "# HOTFIX: Disable Gemma for agent workloads" >> .env
    echo "OPENAI_MODEL_BASIC=" >> .env
    echo "✅ Added OPENAI_MODEL_BASIC= (empty) to .env"
else
    echo "⚠️  OPENAI_MODEL_BASIC already in .env"
fi

# ============================================
# HOTFIX 2: Force provider for known agents
# ============================================
echo ""
echo "[2/4] Adding forced provider configuration..."

if ! grep -q "FORCE_PROVIDER_FOR_AGENTS" controller/core/config.py; then
    cat >> controller/core/config.py << 'EOF'

# HOTFIX: Force specific providers for agents to prevent identity drift
FORCE_PROVIDER_FOR_AGENTS: Dict[str, str] = {
    "agent2": "openai",  # Pin agent2 to OpenAI
}
EOF
    echo "✅ Added FORCE_PROVIDER_FOR_AGENTS to config.py"
else
    echo "⚠️  FORCE_PROVIDER_FOR_AGENTS already in config.py"
fi

# ============================================
# HOTFIX 3: Add identity injection to routes
# ============================================
echo ""
echo "[3/4] Checking routes for identity injection..."

if [ -f "controller/api/routes.py" ]; then
    if ! grep -q "IDENTITY.md" controller/api/routes.py; then
        echo "⚠️  Identity injection not found in routes.py"
        echo "   Manual fix required: Add identity loading before each request"
        echo ""
        echo "   Add this before provider call:"
        echo '   # HOTFIX: Inject identity into system prompt'
        echo '   identity_content = load_identity_file(agent_id)'
        echo '   messages.insert(0, {"role": "system", "content": identity_content})'
    else
        echo "✅ Identity injection found in routes.py"
    fi
fi

# ============================================
# HOTFIX 4: Add session header support
# ============================================
echo ""
echo "[4/4] Adding session header support..."

if ! grep -q "x-session-id" controller/api/routes.py 2>/dev/null; then
    echo "⚠️  Session header support not found"
    echo "   Recommendation: Upgrade to v1.3.0 for full session management"
else
    echo "✅ Session header support found"
fi

echo ""
echo "=========================================="
echo "Hotfix Summary"
echo "=========================================="
echo ""
echo "Applied changes:"
echo "  ✅ Disabled Gemma (OPENAI_MODEL_BASIC=)"
echo "  ✅ Added forced provider config"
echo ""
echo "Manual actions required:"
echo "  1. Edit .env and add your API keys"
echo "  2. Create identity files in .identity/<agent_id>/IDENTITY.md"
echo "  3. Restart Controller"
echo ""
echo "To verify fix:"
echo "  1. Start conversation: 'Меня зовут Александр'"
echo "  2. Ask: 'Как меня зовут?'"
echo "  3. Expected: 'Your name is Александр'"
echo ""
echo "For full fix, upgrade to Controller v1.3.0"
echo "=========================================="
