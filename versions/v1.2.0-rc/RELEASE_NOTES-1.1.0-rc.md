# Controller 1.1.0 RC Release Notes

## Objective

Harden the external controller from a narrow OpenAI-like prototype into a real OpenClaw-compatible external provider boundary, without modifying OpenClaw itself.

## Key Compatibility Improvements

### 1. Request normalization layer
- Added an explicit normalization step between transport payloads and controller routing logic.
- `messages[].content` now supports both:
  - plain strings
  - OpenAI-style arrays of text content parts
- Structured text parts are normalized into plain text before intent, safety, confidence, and routing logic run.

### 2. Structured content support
- Text-part arrays are accepted and flattened for controller internals.
- Unsupported non-text content parts are rejected explicitly with clear validation errors.

### 3. Streaming contract hardening
- Streaming output is now transport-safe end-to-end.
- `StreamingResponse` only receives serialized string chunks.
- Fallback streaming no longer leaks internal `(chunk, provider)` tuples.
- If a provider fails after a partial stream already started, the controller does not splice a second provider into the same stream.

### 4. Permanent validation diagnostics
- Added a production-safe `RequestValidationError` handler.
- Validation failures now log:
  - request id
  - method
  - path
  - content type
  - validation details
  - sanitized/truncated body preview
- 422 responses now return a structured OpenAI-style error body instead of opaque failures.

### 5. OpenAI-compatible boundary improvements
- Preserves important OpenAI-compatible fields such as:
  - tools
  - tool_choice
  - parallel_tool_calls
  - response_format
  - stream_options
  - metadata
  - seed
- Unknown OpenAI-compatible passthrough fields are preserved for OpenAI-compatible providers.
- Provider compatibility is validated explicitly so incompatible payloads do not reach unsupported backends.

### 6. Routing correctness fixes
- The selected route model is now actually passed to the provider.
- Direct provider routing now correctly resolves:
  - `provider/model`
  - raw backend model ids via `DEFAULT_PROVIDER`
- Controller mode filters providers by request compatibility before routing.

### 7. Provider/runtime cleanup
- Startup/provider logs now reflect real runtime state.
- Fixed the stale-import logging issue that previously allowed contradictory startup messages.
- OpenAI-compatible provider health checks now use the configured `OPENAI_BASE_URL` instead of a hardcoded public OpenAI endpoint.
- `FALLBACK_PROVIDERS` now accepts both comma-separated and JSON-array syntax.

### 8. Operational/script cleanup
- Lifecycle scripts updated to 1.1.0 RC branding.
- `status.sh` now parses `/health` JSON robustly instead of relying on brittle text extraction.
- Removed blanket gzip middleware to keep streaming behavior predictable and avoid middleware-side transport issues.

## Regression Coverage Added

The included regression suite covers:
1. plain string `messages[].content`
2. structured text-part `messages[].content`
3. mixed OpenClaw-style transcript histories with tool payloads
4. streaming through direct provider path
5. streaming through fallback provider path
6. validation error logging
7. long admin/operator style prompts
8. route-selected model propagation
9. comma-separated fallback provider configuration parsing

## Validation Status

Test suite result on the packaged 1.1.0 RC:

```text
14 passed
```

## Recommended Deployment Notes

For DGX/Ollama/OpenAI-compatible backends, set:
- `OPENAI_BASE_URL` to the local OpenAI-compatible gateway
- `OPENAI_DEFAULT_MODEL` to the default backend model id
- optional tier overrides:
  - `OPENAI_MODEL_PREMIUM`
  - `OPENAI_MODEL_STANDARD`
  - `OPENAI_MODEL_BASIC`

This lets controller intelligence mode choose backend models without modifying OpenClaw.
