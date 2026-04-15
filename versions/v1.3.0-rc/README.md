# Controller v1.3.0 RC

## Session-aware, Identity-preserving, Safe Routing LLM Controller

Это **runnable release candidate** Controller v1.3.0, решающий критические проблемы v1.2.0:
- ✅ **Identity drift** — агент больше не "забывает" своё имя
- ✅ **Context loss** — контекст сохраняется между сообщениями
- ✅ **Safe routing** — переключение моделей без потери контекста
- ✅ **Session persistence** — Redis/file-based session storage
- ✅ **Observability** — полная visibility в routing decisions

---

## Быстрый старт

### 1. Установка зависимостей

```bash
cd controller_v130
pip install -r requirements.txt
```

### 2. Настройка окружения

```bash
cp .env.example .env
# Отредактируйте .env под вашу конфигурацию
```

### 3. Запуск

```bash
python main.py
```

Controller запустится на `http://localhost:8080`

---

## Ключевые отличия от v1.2.0

### Session Management

**v1.2.0:** Stateless, каждый запрос независим
```python
CACHE_ENABLED=false
REDIS_URL=None
```

**v1.3.0:** Session-centric с persistence
```python
SESSION_STORE_TYPE=file  # или redis
SESSION_TTL_SECONDS=86400
```

### Identity Preservation

**v1.2.0:** Identity может "дрейфовать" между сообщениями

**v1.3.0:** Identity загружается один раз и кэшируется навсегда
```python
identity = await identity_manager.load_identity("agent2")
# identity.name = "Толик" — неизменно для всей сессии
```

### Safe Routing

**v1.2.0:** Tier routing может переключить модель без сохранения контекста

**v1.3.0:** Routing с проверкой context window и сохранением контекста
```python
routing_decision = await routing_engine.select_provider(session, request)
# Проверяет: pinned provider, context window, provider health
```

---

## API Endpoints

### Chat Completions (с сессиями)

```bash
# Создать новую сессию
curl -X POST http://localhost:8080/v1/controller/session \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "agent2", "pin_provider": "openai"}'

# Ответ: {"session_id": "uuid", "identity": {"name": "agent2"}, ...}

# Использовать сессию
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: <uuid>" \
  -d '{
    "model": "controller",
    "messages": [{"role": "user", "content": "Меня зовут Александр"}]
  }'

# Продолжить диалог (контекст сохранится)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: <uuid>" \
  -d '{
    "model": "controller",
    "messages": [{"role": "user", "content": "Как меня зовут?"}]
  }'
# Ответ: "Your name is Александр" ✅
```

### Health Check

```bash
curl http://localhost:8080/health
```

### Status

```bash
curl http://localhost:8080/v1/controller/status
```

---

## Конфигурация

### Основные параметры (.env)

```bash
# Session Store
SESSION_STORE_TYPE=file          # file, redis, memory
SESSION_TTL_SECONDS=86400        # 24 hours
SESSION_FILE_PATH=./sessions     # для file store
SESSION_REDIS_URL=redis://localhost:6379/0  # для redis

# Safe Routing
SAFE_ROUTING_ENABLED=true
CONTEXT_WINDOW_CHECK_ENABLED=true
CONTEXT_WINDOW_THRESHOLD=0.9     # Route to larger at 90% capacity
DEFAULT_PROVIDER=openai
FALLBACK_PROVIDERS=["anthropic", "openai"]

# Identity
IDENTITY_DIR=./identity
IDENTITY_CACHE_FOREVER=true
IDENTITY_AUTO_INJECT=true

# Feature Flags
FEATURE_SESSION_PERSISTENCE=true
FEATURE_IDENTITY_CACHING=true
FEATURE_CONTEXT_PRESERVATION=true
FEATURE_SAFE_ROUTING=true
FEATURE_PROVIDER_PINNING=true
```

### Провайдеры

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL_PREMIUM=gpt-4-turbo
OPENAI_MODEL_STANDARD=gpt-4

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL_PREMIUM=claude-3-opus-20240229

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4-e4b-q4-local
```

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CONTROLLER v1.3.0                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   API Layer  │───▶│  Controller  │───▶│   Response   │              │
│  │  (FastAPI)   │    │   Core       │    │   Formatter  │              │
│  └──────────────┘    └──────┬───────┘    └──────────────┘              │
│                             │                                           │
│         ┌───────────────────┼───────────────────┐                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Session    │    │    Router    │    │   Identity   │              │
│  │   Manager    │◄──►│   Engine     │◄──►│   Manager    │              │
│  └──────┬───────┘    └──────┬───────┘    └──────────────┘              │
│         │                   │                                           │
│         ▼                   ▼                                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Session    │    │   Provider   │    │   Context    │              │
│  │   Store      │    │   Registry   │    │   Engine     │              │
│  │ (Redis/File) │    │              │    │              │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Миграция с v1.2.0

### Phase 1: Hotfix (сегодня)

1. Отключить Gemma для агентов:
```bash
# .env
OPENAI_MODEL_BASIC=  # пусто — не использовать Gemma
```

2. Фиксировать модель для сессии:
```bash
curl -X POST http://localhost:8080/v1/controller/session \
  -d '{"agent_id": "agent2", "pin_provider": "openai"}'
```

### Phase 2: Полная миграция

1. Создать директорию identity:
```bash
mkdir -p identity/agent2
echo "Your name is Толик." > identity/agent2/IDENTITY.md
```

2. Запустить v1.3.0:
```bash
python main.py
```

3. Проверить работу:
```bash
# Тест identity retention
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "x-agent-id: agent2" \
  -d '{"messages": [{"role": "user", "content": "Как тебя зовут?"}]}'
```

---

## Тестирование

### Unit Tests

```bash
pytest tests/unit/
```

### Integration Tests

```bash
pytest tests/integration/
```

### Identity Retention Test

```bash
python tests/test_identity_retention.py
```

---

## Troubleshooting

### Проблема: Сессия не сохраняется

**Решение:** Проверьте `SESSION_STORE_TYPE` и `SESSION_FILE_PATH` или `SESSION_REDIS_URL`

### Проблема: Identity не загружается

**Решение:** Проверьте `IDENTITY_DIR` и наличие `IDENTITY.md` файла

### Проблема: Routing не работает

**Решение:** Проверьте `DEFAULT_PROVIDER` и `FALLBACK_PROVIDERS`

---

## Лицензия

MIT
