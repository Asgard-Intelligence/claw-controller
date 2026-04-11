# Changelog

## Controller 1.2.0 RC

### Added
- Backend model discovery and inventory refresh from provider `/models` endpoints
- Provider catalog reporting with discovered models, resolved tier mappings, and inventory issues
- Safer final-answer extraction layer for OpenAI-compatible responses
- Expanded regression coverage for routing, response extraction, structured payloads, and streaming

### Changed
- Routing decisions now use only discovered backend models
- Direct requests validate requested model ids before upstream calls
- Fallback routing resolves provider-specific target models instead of reusing incompatible ones
- Streaming response paths sanitize legacy tuple/non-string chunks before the HTTP layer
- Configuration defaults no longer assume generic `gpt-4o*` or `claude-*` model ids

### Fixed
- `404 model not found` cascades caused by invented provider model ids
- Empty successful responses caused by missing `message.content` without safe extraction
- Raw reasoning leakage to users when backend responses were missing final content
- Structured OpenClaw message payload compatibility gaps
- Streaming transport crashes caused by tuple chunks reaching `StreamingResponse`


## Controller 1.2.0 RC

### Overview

Hardening release focused on real OpenClaw runtime compatibility. This RC upgrades the controller from a narrow OpenAI-like prototype into a more reliable external provider boundary without changing OpenClaw itself.

### Highlights

- Added explicit request normalization before routing logic
- Added support for structured text-part `messages[].content`
- Added provider compatibility filtering for tool-related and OpenAI-specific payloads
- Fixed streaming boundary so HTTP streaming only emits transport-safe chunks
- Added permanent request validation diagnostics with request id and sanitized body preview
- Fixed provider startup/state logging so runtime logs match actual provider state
- Fixed route-selected model propagation to providers
- Fixed OpenAI-compatible health checks to use configured `OPENAI_BASE_URL`
- Fixed `FALLBACK_PROVIDERS` parsing for comma-separated configuration values
- Added regression coverage for real OpenClaw-like payloads and streaming paths

### Testing

```bash
python -m pytest tests/test_integration.py -v
```

---

## Controller 1.0.0 RC

### Overview

Release Candidate for Controller 1.0.0. This version combines the best of v9 (runnable standalone package) with v10 (intelligence features) while fixing critical issues from the v10 draft.

### What's New

#### Intelligence Mode (from v10)
- **Real Confidence Calculator**: Multi-factor confidence scoring based on complexity, provider health, safety, and history
- **Safety Framework**: PII detection, jailbreak pattern detection, content filtering with risk levels
- **Intent Classification**: Automatic classification of user intent (code, creative, analytical, factual, technical)
- **Route Selector**: Intelligent provider/tier selection based on confidence and safety scores
- **Fallback Manager**: Ordered fallback chain with circuit breaker pattern

#### Architecture Improvements (from v10)
- Dependency injection mindset
- Structured logging (JSON/text formats)
- Rate limiting per client
- Circuit breaker for resilience
- Pydantic models for all data
- External configuration via environment variables

#### Package Structure (from v9)
- **Runnable standalone package**: Complete with install/start/stop lifecycle scripts
- **Proper `main.py` entry point**: No broken startup path
- **Lifecycle commands**: install, start, stop, status, restart, doctor, uninstall
- **Runtime directories**: logs, cache with automatic creation
- **PID file management**: Proper process tracking

### Fixed Issues from v10 Draft

1. **No standalone package** → Fixed: Complete runnable package with all scripts
2. **No `src/` and `main.py`** → Fixed: Proper `controller/main.py` entry point
3. **Broken startup path** → Fixed: `python -m controller.main` works correctly
4. **No lifecycle scripts** → Fixed: Complete lifecycle management
5. **No real packaging shape** → Fixed: v9-style package structure

### API Changes

#### New Endpoints
- `GET /v1/controller/status` - Detailed controller status
- `POST /v1/controller/explain` - Explain routing decision without execution

#### Enhanced Endpoints
- `POST /v1/chat/completions` - Now includes controller metadata in response
- `GET /v1/models` - Lists both controller and underlying provider models

### Configuration Changes

All configuration moved to environment variables (`.env`):

```bash
# Required
CONTROLLER_API_KEY=your-secure-api-key-here-min16chars

# Provider keys (at least one recommended)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Intelligence settings
ENABLE_PII_DETECTION=true
ENABLE_JAILBREAK_DETECTION=true
CONFIDENCE_THRESHOLD_HIGH=0.8
SAFETY_RISK_THRESHOLD=0.7

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

### Dependencies

#### Required
- fastapi
- uvicorn
- pydantic
- pydantic-settings
- httpx
- python-dotenv

#### Optional (for future enhancements)
- sentence-transformers
- scikit-learn
- redis
- opentelemetry
- prometheus

### Breaking Changes

None - this is the first RC release.

### Known Limitations

1. **No persistent caching**: Cache is in-memory only in RC
2. **No distributed rate limiting**: Per-instance only
3. **No metrics endpoint**: Prometheus metrics planned for 1.1.0
4. **No tracing**: OpenTelemetry planned for 1.1.0

### Migration from v9

v9 users can migrate by:
1. Backing up your v9 `.env`
2. Running `./scripts/install.sh` in new package
3. Copying API keys to new `.env`
4. Starting with `./scripts/start.sh`

### Migration from v10 Draft

v10 draft was documentation-only. Use this RC package as fresh install.

### Testing

Run integration tests:
```bash
python -m pytest tests/test_integration.py
```

### Documentation

- `README.md` - Full usage guide
- `docs/openclaw-integration.md` - OpenClaw integration details
- API docs at `/docs` when running

---

## Previous Versions

### v9.0
- Runnable standalone package
- Basic lifecycle scripts
- OpenAI-compatible API
- Simple routing

### v10 Draft
- Intelligence mode documentation
- Confidence/safety concepts
- No runnable code

---

## Roadmap

### 1.0.0 (GA)
- Bug fixes from RC feedback
- Performance optimizations
- Documentation improvements

### 1.1.0
- Redis caching
- Prometheus metrics
- OpenTelemetry tracing

### 1.2.0
- Additional providers (Gemini, etc.)
- Advanced intent classification
- Custom routing rules
