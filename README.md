# Claw Controller

Intelligent orchestration and routing layer for LLM requests, preserved here as a clean versioned repository from the first release candidate through the current v1.4 release.

## Repository structure

- `releases/v1.4.0` - latest main version
- `versions/v1.3.0-rc` - session-aware RC before v1.4
- `versions/v1.2.0-rc` - routing and transport hardening RC
- `versions/v1.1.0-rc` - compatibility and validation RC
- `versions/v1.0.0-rc` - initial controller RC

## Start here

If you want the newest controller, use:

- `releases/v1.4.0`

## Evolution summary

### v1.0.0-rc
- initial intelligent routing and safety layer
- OpenAI-compatible external provider model

### v1.1.0-rc
- improved transport compatibility
- stronger normalization and request validation

### v1.2.0-rc
- backend-aware model discovery
- safer final-answer extraction
- stricter transport safety

### v1.3.0-rc
- session-aware architecture
- identity preservation
- context retention and safe routing
- file or Redis-backed persistence

### v1.4.0
- hybrid route execution model
- route-level provider and model execution
- separate validation and health services
- adapter-based execution engine
- lossless session persistence and v1.3 migration support

## Notes

This repository is organized as a clean release archive, with older versions preserved for reference and the latest version easy to discover.
