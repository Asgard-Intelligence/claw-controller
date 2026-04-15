"""Adapter for Ollama native chat API."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import httpx

from ..endpoint_normalizer import build_url
from ..provider_contracts import (
    AdapterError,
    ApiDialect,
    CanonicalMessage,
    CanonicalRequest,
    ExecutionResult,
    ExecutionUsage,
    RouteTarget,
    ValidationKind,
)
from .base import BaseProviderAdapter


class OllamaNativeAdapter(BaseProviderAdapter):
    supported_dialects = [ApiDialect.OLLAMA_CHAT]

    async def execute(
        self,
        request: CanonicalRequest,
        route: RouteTarget,
        messages: List[CanonicalMessage],
        timeout_seconds: float,
    ) -> ExecutionResult:
        started = time.time()

        payload = {
            "model": route.model_id,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "stream": False,
            "options": {},
        }
        if request.temperature is not None:
            payload["options"]["temperature"] = request.temperature

        url = build_url(route.base_url, "/api/chat")

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(url, json=payload)
        except httpx.RequestError as exc:
            raise AdapterError(
                message=str(exc),
                kind=ValidationKind.TRANSPORT,
                retryable=True,
            ) from exc

        if response.status_code >= 400:
            raise self._error_from_response(response)

        data = response.json()
        content = str(data.get("message", {}).get("content", ""))
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)

        return ExecutionResult(
            content=content,
            provider_id=route.provider_id,
            model_id=route.model_id,
            api_dialect=route.api_dialect,
            usage=ExecutionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
            tool_calls=[],
            raw_metadata={
                "status_code": response.status_code,
                "done": bool(data.get("done", False)),
            },
            latency_ms=int((time.time() - started) * 1000),
        )

    def _error_from_response(self, response: httpx.Response) -> AdapterError:
        message = response.text[:400]
        if response.status_code in {404, 405}:
            kind = ValidationKind.ENDPOINT
        elif response.status_code in {401, 403}:
            kind = ValidationKind.CREDENTIAL
        elif 500 <= response.status_code < 600:
            kind = ValidationKind.TRANSPORT
        else:
            kind = ValidationKind.UNKNOWN

        return AdapterError(
            message=message,
            kind=kind,
            http_status=response.status_code,
            retryable=kind == ValidationKind.TRANSPORT,
        )
