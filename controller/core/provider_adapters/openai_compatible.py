"""Adapter for OpenAI-compatible routes."""

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


class OpenAICompatibleAdapter(BaseProviderAdapter):
    supported_dialects = [ApiDialect.OPENAI_RESPONSES, ApiDialect.OPENAI_CHAT_COMPLETIONS]

    async def execute(
        self,
        request: CanonicalRequest,
        route: RouteTarget,
        messages: List[CanonicalMessage],
        timeout_seconds: float,
    ) -> ExecutionResult:
        started = time.time()

        if route.api_dialect == ApiDialect.OPENAI_RESPONSES:
            url = build_url(route.base_url, "/responses")
            payload = {
                "model": route.model_id,
                "input": self._to_responses_messages(messages),
                "max_output_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": False,
            }
        else:
            url = build_url(route.base_url, "/chat/completions")
            payload = {
                "model": route.model_id,
                "messages": self._to_chat_messages(messages),
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": False,
            }

        headers = self.build_auth_headers(route.credential_env)

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
        content = self._extract_content(data, route.api_dialect)
        usage = self._extract_usage(data)
        latency_ms = int((time.time() - started) * 1000)

        return ExecutionResult(
            content=content,
            provider_id=route.provider_id,
            model_id=route.model_id,
            api_dialect=route.api_dialect,
            usage=usage,
            finish_reason=self._extract_finish_reason(data, route.api_dialect),
            tool_calls=self._extract_tool_calls(data, route.api_dialect),
            raw_metadata={"status_code": response.status_code},
            latency_ms=latency_ms,
        )

    def _to_chat_messages(self, messages: List[CanonicalMessage]) -> List[Dict[str, Any]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _to_responses_messages(self, messages: List[CanonicalMessage]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            normalized.append(
                {
                    "role": msg.role,
                    "content": [{"type": "input_text", "text": msg.content}],
                }
            )
        return normalized

    def _extract_content(self, data: Dict[str, Any], dialect: ApiDialect) -> str:
        if dialect == ApiDialect.OPENAI_CHAT_COMPLETIONS:
            choices = data.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "") or ""
            return ""

        if data.get("output_text"):
            return str(data.get("output_text"))

        output = data.get("output") or []
        for item in output:
            for part in item.get("content", []):
                if part.get("type") in {"output_text", "text"} and part.get("text"):
                    return str(part["text"])
        return ""

    def _extract_usage(self, data: Dict[str, Any]) -> ExecutionUsage:
        usage = data.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        return ExecutionUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
        )

    def _extract_finish_reason(self, data: Dict[str, Any], dialect: ApiDialect) -> str:
        if dialect == ApiDialect.OPENAI_CHAT_COMPLETIONS:
            choices = data.get("choices") or []
            if choices:
                return choices[0].get("finish_reason") or "stop"
            return "stop"

        if data.get("status"):
            return str(data["status"])
        return "stop"

    def _extract_tool_calls(self, data: Dict[str, Any], dialect: ApiDialect) -> List[Dict[str, Any]]:
        if dialect == ApiDialect.OPENAI_CHAT_COMPLETIONS:
            choices = data.get("choices") or []
            if not choices:
                return []
            tool_calls = choices[0].get("message", {}).get("tool_calls")
            return tool_calls or []

        output = data.get("output") or []
        calls: List[Dict[str, Any]] = []
        for item in output:
            if item.get("type") == "tool_call":
                calls.append(item)
        return calls

    def _error_from_response(self, response: httpx.Response) -> AdapterError:
        payload = self._safe_json(response)
        message = self._extract_error_message(payload) or response.text[:400]

        if response.status_code in {401, 403}:
            kind = ValidationKind.CREDENTIAL
            retryable = False
        elif response.status_code in {404, 405}:
            kind = ValidationKind.ENDPOINT
            retryable = False
        elif "model" in message.lower() and ("not found" in message.lower() or "invalid" in message.lower()):
            kind = ValidationKind.MODEL
            retryable = False
        elif 500 <= response.status_code < 600:
            kind = ValidationKind.TRANSPORT
            retryable = True
        else:
            kind = ValidationKind.UNKNOWN
            retryable = False

        return AdapterError(
            message=message,
            kind=kind,
            http_status=response.status_code,
            retryable=retryable,
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
