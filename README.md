# Controller v1.4.0

Session-aware hybrid orchestration layer with route-level routing and execution.

## What Changed From v1.3

- Routing unit is now an executable `RouteTarget` (`provider + model + dialect + endpoint + health/validation`)
- Provider metadata and model metadata are separated
- API layer is orchestration-only (`normalize -> route -> execute -> normalize response`)
- Validation and health are separate services (with independent TTL caches)
- Execution is adapter-based (no mock provider path in production flow)
- Session persistence is lossless and route-aware, with legacy v1.3 migration support
- Existing endpoint and headers stay compatible:
  - `POST /v1/chat/completions`
  - `x-session-id`
  - `x-agent-id`
  - `x-pin-provider`
- Optional new hints:
  - `x-pin-model`
  - `x-routing-mode`

## Core Modules (v1.4)

- `controller/core/provider_contracts.py`
- `controller/core/provider_registry_v140.py`
- `controller/core/provider_bootstrap.py`
- `controller/core/validation_service.py`
- `controller/core/health_service.py`
- `controller/core/routing_policy_v140.py`
- `controller/core/hybrid_router.py`
- `controller/core/execution_engine.py`
- `controller/core/provider_adapters/*`
- `controller/core/request_normalizer.py`
- `controller/core/response_normalizer.py`
- `controller/core/session_manager.py`

## Quick Start

```bash
cd /Users/admin/Documents/Контролер/Контроллер_1.4/controller_v140
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

Server runs on `http://localhost:8080` by default.

## Backward-Compatible API Usage

Create session:

```bash
curl -X POST 'http://localhost:8080/v1/controller/session?agent_id=agent2&pin_provider=openai'
```

Chat completion with session:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-session-id: <session_id>' \
  -d '{
    "model": "controller",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

## Testing

```bash
. .venv/bin/activate
pytest -q
```

Current suite includes route registry, URL normalization, validation, health, hybrid routing, session migration/roundtrip, identity retention, and backward compatibility API behavior.

## Security Notes

- Secrets are never persisted in session state.
- Secrets are never written to logs by controller core.
- Session stores keep canonical conversation state only (provider payloads are not persisted).
