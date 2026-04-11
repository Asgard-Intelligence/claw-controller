# Controller 1.2.0 RC Release Notes

## Focus

This release candidate hardens the external controller for real OpenClaw runtime traffic and real backend model inventories. The goal is to keep OpenClaw unchanged while making the controller safe, backend-aware, and production-runnable.

## Key fixes

- Backend model discovery from `/v1/models` for OpenAI-compatible and Anthropic-compatible providers
- Tier routing now resolves only against discovered model ids
- Configured tier/default model mappings are validated against backend inventory and surfaced in provider status
- Direct and fallback routing no longer invent generic model ids such as `gpt-4o` or `gpt-4o-mini`
- Final-answer extraction prefers `message.content` and avoids exposing raw reasoning as normal user output
- Structured `messages[].content` text-part arrays are normalized at the boundary before routing
- Streaming paths now sanitize emitted chunks so only transport-safe bytes/strings reach the HTTP layer
- Controller explain/status endpoints expose provider discovery state and resolved tier catalogs

## Regression coverage

The packaged 1.2.0 RC test suite covers:

- routing against discovered Gemma-only inventories
- no fallback to nonexistent `gpt-4o*` ids
- direct-model validation against discovered backend inventory
- clean final-answer extraction when `content` is empty
- protection against raw reasoning leakage
- persona/language fidelity regressions around transcript replay
- structured content request normalization
- OpenClaw-style mixed transcript compatibility
- transport-safe direct and fallback streaming

## Validation result

`pytest -q` on the packaged 1.2.0 RC completed successfully with all tests passing.
