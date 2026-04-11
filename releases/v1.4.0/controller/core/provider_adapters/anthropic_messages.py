"""Adapter for Anthropic Messages API."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

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


class AnthropicMessagesAdapter(BaseProviderAdapter):
    supported_dialects = [ApiDialect.ANTHROPIC_MESSAGES]

    async def execute(
        self,
        request: CanonicalRequest,
        route: RouteTarget,
        messages: List[CanonicalMessage],
        timeout_seconds: float,
    ) -> ExecutionResult:
        started = time.time()

        system_prompt, anthro_messages = self._convert_messages(messages)

        payload = {
            "model": route.model_id,
            "max_tokens": request.max_tokens or 512,
            "messages": anthro_messages,
            "temperature": request.temperature,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = self.build_auth_headers(route.credential_env, anthropic=True)
        url = build_url(route.base_url, "/v1/messages")

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.RequestError as exc:
            raise AdapterError(
                message=str(exc),
                kind=ValidationKind.TRANSPORT,
                retryable=True,
            ) from exc

        if response.status_code >= 400:
            raise self._error_from_response(response)

        data = response.json()
        content = self._extract_content(data)
        usage = self._extract_usage(data)

        return ExecutionResult(
            content=content,
            provider_id=route.provider_id,
            model_id=route.model_id,
            api_dialect=route.api_dialect,
            usage=usage,
            finish_reason=str(data.get("stop_reason") or "stop"),
            tool_calls=[],
            raw_metadata={"status_code": response.status_code},
            latency_ms=int((time.time() - started) * 1000),
        )

    def _convert_messages(self, messages: List[CanonicalMessage]) -> Tuple[str, List[Dict[str, Any]]]:
        system_prompt = ""
        output: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                if not system_prompt:
                    system_prompt = msg.content
                continue
            if msg.role not in {"user", "assistant"}:
                continue
            output.append({"role": msg.role, "content": msg.content})

        return system_prompt, output

    def _extract_content(self, data: Dict[str, Any]) -> str:
        content = data.get("content") or []
        parts: List[str] = []
        for block in content:
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join([p for p in parts if p])

    def _extract_usage(self, data: Dict[str, Any]) -> ExecutionUsage:
        usage = data.get("usage") or {}
        prompt = int(usage.get("input_tokens", 0))
        completion = int(usage.get("output_tokens", 0))
        return ExecutionUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )

    def _error_from_response(self, response: httpx.Response) -> AdapterError:
        payload = self._safe_json(response)
        message = self._extract_error_message(payload) or response.text[:400]

        if response.status_code in {401, 403}:
            kind = ValidationKind.CREDENTIAL
        elif response.status_code in {404, 405}:
            kind = ValidationKind.ENDPOINT
        elif "model" in message.lower() and ("not found" in message.lower() or "invalid" in message.lower()):
            kind = ValidationKind.MODEL
        elif 500 <= response.status_code < 600:
            kind = ValidationKind.TRANSPORT
        else:
            kind = ValidationKind.UNKNOWN

        return AdapterError(
            message=message,
            kind=kind,
            http_status=response.status_code,
            retryable=kind == ValidationKind.TRANSPORT,
            response_payload=payload,
        )

    def _safe_json(self, response: httpx.Response) -> Dict[str, Any]:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            return {"payload": payload}
        except Exception:
            return {"error": response.text[:400]}

    def _extract_error_message(self, payload: Dict[str, Any]) -> str:
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or "")
        if isinstance(error, str):
            return error
        return ""
