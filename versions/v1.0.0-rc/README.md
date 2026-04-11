# Controller 1.0.0 RC

Intelligent routing and safety layer for LLM requests. Controller acts as an **external provider** to OpenClaw, providing confidence-based routing, safety checks, and fallback management.

> **No-Touch OpenClaw Principle**: Controller does NOT modify OpenClaw internals. It operates purely as an external provider via standard API calls.

## Features

- **Intelligence Mode**: Multi-factor confidence calculation, safety checking, intent classification
- **Safety Framework**: PII detection, jailbreak pattern detection, content filtering
- **Provider Management**: OpenAI, Anthropic support with automatic fallback
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **External Configuration**: All settings via environment variables
- **Structured Logging**: JSON/text log formats
- **Rate Limiting**: Request throttling per client
- **Circuit Breaker**: Automatic failover on provider failures

## Quick Start

```bash
# 1. Install
cd controller-1.0.0-rc
./scripts/install.sh

# 2. Configure (edit .env with your API keys)
cp .env.example .env
nano .env

# 3. Start
./scripts/start.sh

# 4. Test
curl http://localhost:8080/health
```

## Requirements

- Python 3.9+
- macOS or Linux
- At least one LLM provider API key (OpenAI or Anthropic)

## Installation

### Step 1: Run Installer

```bash
./scripts/install.sh
```

This will:
- Create Python virtual environment
- Install dependencies
- Create runtime directories
- Copy `.env.example` to `.env`

### Step 2: Configure

Edit `.env` and set your API keys:

```bash
# Required: Controller API key (min 16 characters)
CONTROLLER_API_KEY=your-secure-api-key-here-min16chars

# Optional: LLM provider keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Step 3: Start

```bash
./scripts/start.sh
```

## OpenClaw Integration

Controller integrates with OpenClaw as an **external provider**:

### Configuration in OpenClaw

Add to your OpenClaw configuration:

```json
{
  "providers": {
    "controller": {
      "type": "openai-compatible",
      "base_url": "http://localhost:8080/v1",
      "api_key": "your-controller-api-key",
      "models": ["controller"],
      "default_model": "controller"
    }
  }
}
```

### Usage

In OpenClaw, use `model: controller` to route through Controller:

```json
{
  "model": "controller",
  "messages": [...]
}
```

Controller will:
1. Analyze the request (intent, complexity, safety)
2. Calculate confidence score
3. Select optimal provider and tier
4. Execute with fallback chain
5. Return response with routing metadata

## API Endpoints

### OpenAI-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion with intelligent routing |
| `/v1/models` | GET | List available models |

### Controller-Specific Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/controller/status` | GET | Detailed controller status |
| `/v1/controller/explain` | POST | Explain routing decision without executing |

### Example Requests

#### Chat Completion

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer your-controller-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "controller",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

#### Explain Request

```bash
curl -X POST http://localhost:8080/v1/controller/explain \
  -H "Authorization: Bearer your-controller-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "controller",
    "messages": [{"role": "user", "content": "Write a Python function"}]
  }'
```

Response includes:
- Intent classification
- Safety analysis
- Confidence breakdown
- Routing decision
- Alternative providers considered

## Configuration

All configuration is via environment variables in `.env`:

### Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTROLLER_HOST` | `0.0.0.0` | Server bind host |
| `CONTROLLER_PORT` | `8080` | Server port |
| `CONTROLLER_API_KEY` | *required* | API key for authentication |

### Provider Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *optional* | OpenAI API key |
| `ANTHROPIC_API_KEY` | *optional* | Anthropic API key |
| `DEFAULT_PROVIDER` | `openai` | Default provider |

### Intelligence Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_PII_DETECTION` | `true` | Enable PII detection |
| `ENABLE_JAILBREAK_DETECTION` | `true` | Enable jailbreak detection |
| `CONFIDENCE_THRESHOLD_HIGH` | `0.8` | High confidence threshold |
| `SAFETY_RISK_THRESHOLD` | `0.7` | Safety risk threshold |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | Requests per minute |

## Lifecycle Commands

| Command | Description |
|---------|-------------|
| `./scripts/install.sh` | Install dependencies |
| `./scripts/start.sh` | Start Controller |
| `./scripts/stop.sh` | Stop Controller |
| `./scripts/status.sh` | Check status |
| `./scripts/restart.sh` | Restart Controller |
| `./scripts/doctor.sh` | Diagnose environment |
| `./scripts/uninstall.sh` | Remove installation |

## Intelligence Mode

Controller analyzes each request through multiple factors:

### Confidence Calculation

- **Complexity**: Request length, message count, code indicators
- **Provider Health**: Latency, error rates, availability
- **Safety Score**: PII, jailbreak, content violations
- **Historical Success**: Past request success rates
- **Intent Clarity**: Classification confidence

### Safety Checks

- **PII Detection**: Email, phone, SSN, credit card, API keys
- **Jailbreak Detection**: Pattern matching for prompt injection
- **Content Filtering**: Policy violation detection

### Intent Classification

- `code` - Programming tasks
- `creative` - Writing, storytelling
- `analytical` - Analysis, comparison
- `factual` - Information queries
- `technical` - Architecture, configuration

### Routing Tiers

| Tier | Confidence | Providers | Use Case |
|------|------------|-----------|----------|
| Premium | ≥0.8 | Best models | Complex tasks |
| Standard | ≥0.5 | Balanced | General use |
| Basic | <0.5 | Fast/cheap | Simple queries |

## Troubleshooting

### Controller won't start

```bash
# Check environment
./scripts/doctor.sh

# Check logs
tail -f runtime/logs/controller.log
```

### No providers available

Controller runs in mock mode without API keys. Set at least one:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

### Rate limit errors

Adjust in `.env`:
```bash
RATE_LIMIT_REQUESTS_PER_MINUTE=120
```

### High safety risk blocks

Check `/v1/controller/explain` to see why a request was blocked.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Controller                              │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Intent     │  │   Safety     │  │  Confidence  │      │
│  │ Classifier   │  │  Checker     │  │  Calculator  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         └─────────────────┴─────────────────┘               │
│                           │                                  │
│                    ┌──────┴──────┐                          │
│                    │   Route     │                          │
│                    │  Selector   │                          │
│                    └──────┬──────┘                          │
│                           │                                  │
│         ┌─────────────────┼─────────────────┐               │
│         ▼                 ▼                 ▼               │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐        │
│  │  OpenAI    │    │ Anthropic  │    │  Fallback  │        │
│  │  Provider  │    │  Provider  │    │  Manager   │        │
│  └────────────┘    └────────────┘    └────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        ┌──────────┐
                        │ OpenClaw │ (external)
                        └──────────┘
```

## License

MIT License - OpenClaw Project

## Version

1.0.0 RC (Release Candidate)
