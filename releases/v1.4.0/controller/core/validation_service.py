"""Route validation service with typed failure classification."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

import httpx

from .endpoint_normalizer import build_url
from .provider_contracts import ApiDialect, ValidationKind, ValidationResult
from .provider_registry_v140 import ProviderRegistryV140


class RouteValidationService:
    def __init__(
        self,
        registry: ProviderRegistryV140,
        timeout_seconds: float = 6.0,
        cache_ttl_seconds: int = 300,
    ):
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Tuple[float, ValidationResult]] = {}

    async def get_route_validation(self, route_key: str, force_refresh: bool = False) -> ValidationResult:
        route = self.registry.get_route(route_key)
        if not route:
            return ValidationResult(
                ok=False,
                validated=False,
                kind=ValidationKind.UNKNOWN,
                message=f"Unknown route: {route_key}",
            )

        if not force_refresh and route_key in self._cache:
            ts, cached = self._cache[route_key]
            if (time.time() - ts) < self.cache_ttl_seconds:
                return cached

        result = await self._validate_route(route_key)
        self._cache[route_key] = (time.time(), result)
        self.registry.set_validation_result(route_key, result)
        return result

    async def _validate_route(self, route_key: str) -> ValidationResult:
        route = self.registry.get_route(route_key)
        if not route:
            return ValidationResult(False, False, ValidationKind.UNKNOWN, "Route not found")

        provider = self.registry.get_provider(route.provider_id)
        if not provider:
            return ValidationResult(False, False, ValidationKind.UNKNOWN, "Provider not found")

        if not provider.base_url:
            return ValidationResult(False, False, ValidationKind.ENDPOINT, "Base URL missing")

        if route.credential_env:
            token = os.getenv(route.credential_env)
            if not token and provider.kind.value == "cloud":
                return ValidationResult(
                    ok=False,
                    validated=False,
                    kind=ValidationKind.CREDENTIAL,
                    message=f"Missing credential env: {route.credential_env}",
                    transport_code="MISSING_CREDENTIAL",
                )

        if route.api_dialect in {ApiDialect.OPENAI_RESPONSES, ApiDialect.OPENAI_CHAT_COMPLETIONS}:
            return await self._validate_openai_like(route_key)
        if route.api_dialect == ApiDialect.ANTHROPIC_MESSAGES:
            return await self._validate_anthropic(route_key)
        if route.api_dialect == ApiDialect.OLLAMA_CHAT:
            return await self._validate_ollama(route_key)

        return ValidationResult(
            ok=True,
            validated=False,
            kind=ValidationKind.OK,
            message="No specialized validator for dialect; assuming usable",
        )

    async def _validate_openai_like(self, route_key: str) -> ValidationResult:
        route = self.registry.get_route(route_key)
        if not route:
            return ValidationResult(False, False, ValidationKind.UNKNOWN, "Route not found")

        responses_url = build_url(route.base_url, "/responses")
        chat_url = build_url(route.base_url, "/chat/completions")

        headers = self._auth_headers(route.credential_env)

        # Probe /responses first.
        response_result = await self._post_probe(
            responses_url,
            headers,
            {
                "model": route.model_id,
                "input": [{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
                "max_output_tokens": 1,
            },
        )

        if response_result["status"] in {200, 201, 400, 401, 403}:
            return self._map_probe_to_validation(
                response_result,
                preferred=ApiDialect.OPENAI_RESPONSES,
                catalog_checked=False,
            )

        if response_result["status"] not in {404, 405}:
            return self._map_probe_to_validation(
                response_result,
                preferred=None,
                catalog_checked=False,
            )

        # If /responses is not available, try /chat/completions.
        chat_result = await self._post_probe(
            chat_url,
            headers,
            {
                "model": route.model_id,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "stream": False,
            },
        )

        if chat_result["status"] in {200, 201, 400, 401, 403}:
            result = self._map_probe_to_validation(
                chat_result,
                preferred=ApiDialect.OPENAI_CHAT_COMPLETIONS,
                catalog_checked=False,
            )
            if result.kind == ValidationKind.OK and not result.ok:
                result.ok = True
                result.validated = True
                result.message = "Endpoint supports chat/completions"
            return result

        # Both unsupported is endpoint failure.
        if chat_result["status"] in {404, 405}:
            return ValidationResult(
                ok=False,
                validated=False,
                kind=ValidationKind.ENDPOINT,
                message="Neither /responses nor /chat/completions is supported",
                http_status=chat_result["status"],
            )

        return self._map_probe_to_validation(chat_result, preferred=None, catalog_checked=False)

    async def _validate_anthropic(self, route_key: str) -> ValidationResult:
        route = self.registry.get_route(route_key)
        if not route:
            return ValidationResult(False, False, ValidationKind.UNKNOWN, "Route not found")

        url = build_url(route.base_url, "/v1/messages")
        headers = self._auth_headers(route.credential_env, anthropic=True)
        result = await self._post_probe(
            url,
            headers,
            {
                "model": route.model_id,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

        return self._map_probe_to_validation(
            result,
            preferred=ApiDialect.ANTHROPIC_MESSAGES,
            catalog_checked=False,
        )

    async def _validate_ollama(self, route_key: str) -> ValidationResult:
        route = self.registry.get_route(route_key)
        if not route:
            return ValidationResult(False, False, ValidationKind.UNKNOWN, "Route not found")

        url = build_url(route.base_url, "/api/tags")
        result = await self._get_probe(url, headers={})

        if result["status"] in {200, 201}:
            return ValidationResult(
                ok=True,
                validated=True,
                kind=ValidationKind.OK,
                message="Ollama endpoint reachable",
                http_status=result["status"],
                preferred_api_dialect=ApiDialect.OLLAMA_CHAT,
                catalog_checked=True,
            )

        # /api/tags can be unavailable on wrappers; don't hard-fail if transport is fine.
        if result["status"] in {404, 405}:
            return ValidationResult(
                ok=True,
                validated=False,
                kind=ValidationKind.OK,
                message="Catalog endpoint unavailable but route may still be usable",
                http_status=result["status"],
                preferred_api_dialect=ApiDialect.OLLAMA_CHAT,
                catalog_checked=False,
            )

        return self._map_probe_to_validation(
            result,
            preferred=ApiDialect.OLLAMA_CHAT,
            catalog_checked=False,
        )

    def _auth_headers(self, credential_env: Optional[str], anthropic: bool = False) -> Dict[str, str]:
        if not credential_env:
            return {}
        token = os.getenv(credential_env)
        if not token:
            return {}
        if anthropic:
            return {
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def _post_probe(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
            return {
                "status": response.status_code,
                "text": self._extract_text(response),
            }
        except httpx.RequestError as exc:
            return {
                "status": None,
                "transport_code": exc.__class__.__name__,
                "text": str(exc),
            }

    async def _get_probe(self, url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url, headers=headers)
            return {
                "status": response.status_code,
                "text": self._extract_text(response),
            }
        except httpx.RequestError as exc:
            return {
                "status": None,
                "transport_code": exc.__class__.__name__,
                "text": str(exc),
            }

    def _map_probe_to_validation(
        self,
        probe: Dict[str, Any],
        preferred: Optional[ApiDialect],
        catalog_checked: bool,
    ) -> ValidationResult:
        status = probe.get("status")
        text = (probe.get("text") or "").lower()

        if status is None:
            return ValidationResult(
                ok=False,
                validated=False,
                kind=ValidationKind.TRANSPORT,
                message=probe.get("text", "Transport error"),
                transport_code=probe.get("transport_code"),
                preferred_api_dialect=preferred,
                catalog_checked=catalog_checked,
            )

        if status in {200, 201}:
            return ValidationResult(
                ok=True,
                validated=True,
                kind=ValidationKind.OK,
                message="Validation successful",
                http_status=status,
                preferred_api_dialect=preferred,
                catalog_checked=catalog_checked,
            )

        if status in {401, 403}:
            return ValidationResult(
                ok=False,
                validated=True,
                kind=ValidationKind.CREDENTIAL,
                message="Credential rejected",
                http_status=status,
                preferred_api_dialect=preferred,
                catalog_checked=catalog_checked,
            )

        if status in {404, 405}:
            return ValidationResult(
                ok=False,
                validated=True,
                kind=ValidationKind.ENDPOINT,
                message="Endpoint unavailable",
                http_status=status,
                preferred_api_dialect=preferred,
                catalog_checked=catalog_checked,
            )

        if "model" in text and ("not found" in text or "does not exist" in text or "invalid" in text):
            return ValidationResult(
                ok=False,
                validated=True,
                kind=ValidationKind.MODEL,
                message="Model unavailable",
                http_status=status,
                preferred_api_dialect=preferred,
                catalog_checked=catalog_checked,
            )

        if 400 <= status < 500:
            return ValidationResult(
                ok=True,
                validated=True,
                kind=ValidationKind.OK,
                message="Endpoint reachable (client request rejected)",
                http_status=status,
                preferred_api_dialect=preferred,
                catalog_checked=catalog_checked,
            )

        return ValidationResult(
            ok=False,
            validated=True,
            kind=ValidationKind.UNKNOWN,
            message="Unknown validation failure",
            http_status=status,
            preferred_api_dialect=preferred,
            catalog_checked=catalog_checked,
        )

    def _extract_text(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                return json_string(payload)
            return str(payload)
        except Exception:
            return response.text[:400]


def json_string(payload: Dict[str, Any]) -> str:
    try:
        return str(payload)
    except Exception:
        return ""
