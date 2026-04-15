# Controller 1.2.0 RC Build Report

## Source inputs used
- controller-1.1.0-rc-runnable.zip
- controller-1.1.0-rc-patched-20260406-211253.tgz
- logs archive supplied by the user

## Core hardening implemented
- backend model discovery from provider `/models` endpoints
- routing only against discovered backend inventory
- no invented fallback ids like `gpt-4o` / `gpt-4o-mini`
- startup/provider status inventory validation and degraded-state reporting
- final-answer extraction layer that prefers final content and avoids raw reasoning leakage
- structured `messages[].content` normalization for OpenClaw-style payloads
- transport-safe streaming sanitization for direct and fallback paths
- controller status/explain visibility for provider catalogs and resolved tier mappings

## Validation performed
- `pytest -q`
- `python -m py_compile` across controller sources and tests
- startup smoke via `python -m controller.main` with runtime env vars

## Current validation result
- tests: 24 passed
- py_compile: passed
- startup smoke: passed
