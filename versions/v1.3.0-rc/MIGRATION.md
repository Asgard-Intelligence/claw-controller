# Migration Guide: Controller v1.2.0 → v1.3.0

## Executive Summary

Controller v1.3.0 решает критические проблемы v1.2.0:
- ✅ **Identity drift** — агент больше не "забывает" своё имя
- ✅ **Context loss** — контекст сохраняется между сообщениями
- ✅ **Safe routing** — переключение моделей без потери контекста

---

## Migration Phases

### Phase 0: Immediate Hotfix (Today)

**Apply to v1.2.0 to stop identity drift:**

```bash
# 1. Disable Gemma for agents
export OPENAI_MODEL_BASIC=

# 2. Pin provider for agent2
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "x-pin-provider: openai" \
  -d '{"model": "controller", "messages": [...]}'
```

### Phase 1: Deploy v1.3.0 Side-by-Side (Week 1)

```bash
# 1. Deploy v1.3.0 on port 8081
cd controller_v130
cp .env.example .env
# Edit .env with your settings
python main.py  # Runs on port 8080

# 2. Test v1.3.0
curl http://localhost:8080/health
```

### Phase 2: Create Identity Files (Week 1)

```bash
# Create identity directory
mkdir -p identity/agent2

# Create IDENTITY.md
cat > identity/agent2/IDENTITY.md << 'EOF'
# Identity for agent2

Your name is Толик.
You are a helpful AI assistant.

## Core Identity
- Name: Толик
- Role: AI Assistant
- Personality: Helpful, friendly, professional

## Constraints
- Always remember your name is Толик
- Never claim to have a different name
- If asked your name, respond "My name is Толик"
EOF
```

### Phase 3: Test Identity Retention (Week 1)

```bash
# 1. Create session
curl -X POST http://localhost:8080/v1/controller/session \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "agent2", "pin_provider": "openai"}'
# Response: {"session_id": "uuid-123", ...}

# 2. Set user's name
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: uuid-123" \
  -d '{"model": "controller", "messages": [{"role": "user", "content": "Меня зовут Александр"}]}'

# 3. Verify agent remembers user's name
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: uuid-123" \
  -d '{"model": "controller", "messages": [{"role": "user", "content": "Как меня зовут?"}]}'
# Expected: "Your name is Александр" ✅

# 4. Verify agent remembers its own name
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: uuid-123" \
  -d '{"model": "controller", "messages": [{"role": "user", "content": "Как тебя зовут?"}]}'
# Expected: "My name is Толик" ✅
```

### Phase 4: Gradual Traffic Shift (Week 2-3)

```bash
# Update load balancer weights
# Day 1-3:  10% v1.3.0, 90% v1.2.0
# Day 4-6:  25% v1.3.0, 75% v1.2.0
# Day 7-9:  50% v1.3.0, 50% v1.2.0
# Day 10-12: 75% v1.3.0, 25% v1.2.0
# Day 13+: 100% v1.3.0
```

### Phase 5: Decommission v1.2.0 (Week 4)

```bash
# Stop v1.2.0
systemctl stop controller-v120

# Remove v1.2.0 files
rm -rf /opt/controller-v120

# v1.3.0 is now production
```

---

## Configuration Changes

### v1.2.0 → v1.3.0

| Setting | v1.2.0 | v1.3.0 |
|---------|--------|--------|
| `CACHE_ENABLED` | `false` | Removed (always on) |
| `REDIS_URL` | `None` | `SESSION_REDIS_URL` |
| Session storage | None | `SESSION_STORE_TYPE` |
| Identity | None | `IDENTITY_DIR` |
| Context preservation | None | `CONTEXT_PRESERVATION_ENABLED` |
| Safe routing | None | `SAFE_ROUTING_ENABLED` |

### New Required Settings

```bash
# Session (REQUIRED)
SESSION_STORE_TYPE=file  # or redis
SESSION_TTL_SECONDS=86400

# Identity (REQUIRED)
IDENTITY_DIR=./identity
IDENTITY_CACHE_FOREVER=true

# Safe Routing (REQUIRED)
SAFE_ROUTING_ENABLED=true
CONTEXT_WINDOW_CHECK_ENABLED=true
CONTEXT_WINDOW_THRESHOLD=0.9
```

---

## API Changes

### Request Headers

| Header | v1.2.0 | v1.3.0 |
|--------|--------|--------|
| `Authorization` | Required | Optional |
| `x-session-id` | N/A | **NEW** - Session ID |
| `x-agent-id` | N/A | **NEW** - Agent ID |
| `x-pin-provider` | N/A | **NEW** - Pin provider |

### Response Fields

| Field | v1.2.0 | v1.3.0 |
|-------|--------|--------|
| `session_id` | N/A | **NEW** - Session ID |
| `routing` | N/A | **NEW** - Routing info |

### Example Request (v1.3.0)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: existing-session-uuid" \
  -H "x-agent-id: agent2" \
  -H "x-pin-provider: openai" \
  -d '{
    "model": "controller",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Example Response (v1.3.0)

```json
{
  "id": "chatcmpl-uuid",
  "object": "chat.completion",
  "model": "controller",
  "session_id": "session-uuid",
  "routing": {
    "provider": "openai",
    "reason": "Current provider healthy",
    "context_transfer": false
  },
  "choices": [...]
}
```

---

## Rollback Plan

If v1.3.0 causes issues:

```bash
# 1. Stop v1.3.0
systemctl stop controller-v130

# 2. Start v1.2.0
systemctl start controller-v120

# 3. Verify v1.2.0 is running
curl http://localhost:8080/health

# 4. Investigate issue
journalctl -u controller-v130 -f
```

---

## Troubleshooting

### Issue: Session not found

**Cause:** Session store not configured

**Fix:**
```bash
# Check SESSION_STORE_TYPE
export SESSION_STORE_TYPE=file
export SESSION_FILE_PATH=./sessions
```

### Issue: Identity not loaded

**Cause:** IDENTITY.md not found

**Fix:**
```bash
# Create identity file
mkdir -p identity/agent2
echo "Your name is Толик." > identity/agent2/IDENTITY.md
```

### Issue: Routing not working

**Cause:** Provider health check failing

**Fix:**
```bash
# Check provider health
curl http://localhost:8080/health

# Check provider config
export DEFAULT_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

---

## Verification Checklist

- [ ] v1.3.0 deployed and running
- [ ] Health check returns 200
- [ ] Identity files created
- [ ] Session persistence working
- [ ] Identity retention verified
- [ ] Context preservation verified
- [ ] Safe routing verified
- [ ] Metrics collection working
- [ ] Rollback plan tested

---

## Support

For issues during migration:

1. Check logs: `journalctl -u controller-v130 -f`
2. Run tests: `python tests/test_identity_retention.py`
3. Review config: `cat .env`
4. Contact: support@openclaw.ai
