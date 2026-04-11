"""Integration and regression tests for Controller 1.2.0 RC."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

# Ensure the package root is importable when tests are run from the unpacked archive.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test environment must be set before importing application modules.
os.environ["CONTROLLER_API_KEY"] = "test-api-key-for-integration-tests-12345"
os.environ["LOG_LEVEL"] = "INFO"
os.environ["LOG_FORMAT"] = "text"
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ["FALLBACK_PROVIDERS"] = "openai,anthropic"
os.environ["RUNTIME_DIR"] = "/tmp/controller-tests/runtime"
os.environ["LOGS_DIR"] = "/tmp/controller-tests/runtime/logs"
os.environ["CACHE_DIR"] = "/tmp/controller-tests/runtime/cache"
os.environ["PID_FILE"] = "/tmp/controller-tests/runtime/controller.pid"
os.environ["ROUTING_REQUIRE_DISCOVERED_MODELS"] = "true"

from controller.core.config import Settings, get_settings
from controller.core.intelligence import ConfidenceScore, IntentType, RiskLevel, RouteDecision, SafetyScore
from controller.core.providers import FallbackProvider, OpenAIProvider
from controller.main import create_app
from controller.models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    ProviderHealth,
    Role,
    Usage,
)
from controller.api import routes


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def reset_route_runtime_state():
    yield
    routes.providers = {}
    routes.fallback_provider = FallbackProvider({})


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-api-key-for-integration-tests-12345"}


@dataclass
class ProviderCall:
    method: str
    target_model: Optional[str]
    request: Any


class FakeProvider:
    def __init__(
        self,
        name: str,
        *,
        default_model: str,
        response_text: str = "ok",
        validation_error: Optional[str] = None,
        fail_nonstream: bool = False,
        fail_stream_before_emit: bool = False,
        fail_stream_after_emit: bool = False,
        stream_chunks: Optional[List[str]] = None,
        discovered_models: Optional[List[str]] = None,
        resolved_tier_models: Optional[Dict[str, str]] = None,
        inventory_status: str = "healthy",
        inventory_issues: Optional[List[str]] = None,
    ):
        self.name = name
        self.default_model = default_model
        self.response_text = response_text
        self.validation_error = validation_error
        self.fail_nonstream = fail_nonstream
        self.fail_stream_before_emit = fail_stream_before_emit
        self.fail_stream_after_emit = fail_stream_after_emit
        self.stream_chunks = stream_chunks or []
        self.calls: List[ProviderCall] = []
        self.closed = False
        self.discovered_models = discovered_models or ([default_model] if default_model else [])
        self.resolved_tier_models = resolved_tier_models or {
            "premium": self.discovered_models[0] if self.discovered_models else default_model,
            "standard": default_model or (self.discovered_models[0] if self.discovered_models else ""),
            "basic": default_model or (self.discovered_models[0] if self.discovered_models else ""),
        }
        self.routing_default_model = default_model or (self.discovered_models[0] if self.discovered_models else "")
        self.inventory_status = inventory_status
        self.inventory_issues = inventory_issues or []
        self.inventory_refreshed_at = 1

    def routing_catalog(self):
        return SimpleNamespace(
            provider=self.name,
            discovered_models=list(self.discovered_models),
            resolved_tier_models=dict(self.resolved_tier_models),
            default_model=self.routing_default_model,
            routing_enabled=bool(self.resolved_tier_models or self.discovered_models),
            inventory_status=self.inventory_status,
            inventory_issues=list(self.inventory_issues),
            refreshed_at=self.inventory_refreshed_at,
        )

    def resolve_target_model(self, requested_model: Optional[str] = None, tier: Optional[str] = None):
        if requested_model:
            if self.discovered_models and requested_model not in self.discovered_models:
                raise ValueError(f"model '{requested_model}' is not present on backend for provider {self.name}")
            return requested_model
        if tier and tier in self.resolved_tier_models:
            return self.resolved_tier_models[tier]
        if self.routing_default_model:
            return self.routing_default_model
        raise ValueError(f"provider {self.name} has no discovered backend models available for routing")

    def validate_request(self, request):
        if self.validation_error:
            raise ValueError(self.validation_error)
        return None

    async def chat_completion(self, request, target_model=None):
        self.calls.append(ProviderCall("chat_completion", target_model, request))
        if self.fail_nonstream:
            raise RuntimeError(f"{self.name} non-stream failure")
        return ChatCompletionResponse(
            id=f"resp-{self.name}",
            created=1,
            model=target_model or self.default_model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role=Role.ASSISTANT, content=self.response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def chat_completion_stream(self, request, target_model=None):
        self.calls.append(ProviderCall("chat_completion_stream", target_model, request))
        if self.fail_stream_before_emit:
            raise RuntimeError(f"{self.name} stream failure before emit")
        for chunk in self.stream_chunks:
            yield chunk
        if self.fail_stream_after_emit:
            raise RuntimeError(f"{self.name} stream failure after emit")

    async def health_check(self):
        return {
            "status": self.inventory_status,
            "provider": self.name,
            "latency_ms": 1,
            "inventory_issues": self.inventory_issues,
        }

    async def close(self):
        self.closed = True


class StubRouteSelector:
    def __init__(self, selected_provider: str, selected_model: str, tier: str = "standard"):
        self.selected_provider = selected_provider
        self.selected_model = selected_model
        self.tier = tier

    def select(
        self,
        request,
        confidence,
        safety,
        intent,
        provider_health,
        available_providers,
        provider_catalogs=None,
    ):
        return RouteDecision(
            selected_provider=self.selected_provider,
            selected_model=self.selected_model,
            selected_tier=self.tier,
            confidence=confidence,
            safety=safety,
            intent=intent,
            reasoning="stubbed route",
            fallback_chain=[p for p in available_providers if p != self.selected_provider],
        )


class TestHealthAndModels:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.2.0-rc"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Controller"
        assert data["version"] == "1.2.0-rc"

    def test_models_requires_auth(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "authentication_error"

    def test_models_with_auth_returns_only_controller_without_live_providers(self, client, auth_headers):
        response = client.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200
        model_ids = [item["id"] for item in response.json()["data"]]
        assert model_ids == ["controller"]

    def test_models_list_uses_discovered_inventory_not_fake_configured_ids(self, client, auth_headers):
        routes.providers = {
            "openai": FakeProvider(
                "openai",
                default_model="gemma4-e4b-q4-local:latest",
                discovered_models=[
                    "gemma4-e4b-q4-local:latest",
                    "gemma4-31b-q4-local:latest",
                ],
                resolved_tier_models={
                    "premium": "gemma4-31b-q4-local:latest",
                    "standard": "gemma4-e4b-q4-local:latest",
                    "basic": "gemma4-e4b-q4-local:latest",
                },
            )
        }
        routes.fallback_provider = FallbackProvider(routes.providers)

        response = client.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200
        model_ids = [item["id"] for item in response.json()["data"]]
        assert "controller" in model_ids
        assert "gemma4-e4b-q4-local:latest" in model_ids
        assert "gemma4-31b-q4-local:latest" in model_ids
        assert "gpt-4o" not in model_ids
        assert "gpt-4o-mini" not in model_ids


class TestRequestNormalization:
    def test_plain_string_message_content(self, client, auth_headers):
        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "controller",
                "messages": [{"role": "user", "content": "hello controller"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"].startswith("[MOCK MODE]")
        assert data["controller_metadata"]["structured_content_messages"] == 0

    def test_structured_text_part_content_is_accepted_and_normalized(self, client, auth_headers):
        openai = FakeProvider(
            "openai",
            default_model="gemma4-e4b-q4-local:latest",
            response_text="normalized",
            discovered_models=["gemma4-e4b-q4-local:latest"],
        )
        routes.providers = {"openai": openai}
        routes.fallback_provider = FallbackProvider(routes.providers)

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "openai/gemma4-e4b-q4-local:latest",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello "},
                            {"type": "text", "text": "world"},
                        ],
                    }
                ],
            },
        )

        assert response.status_code == 200
        assert openai.calls
        assert openai.calls[0].target_model == "gemma4-e4b-q4-local:latest"
        assert openai.calls[0].request.messages[0].content == "hello world"
        data = response.json()
        assert data["controller_metadata"]["structured_content_messages"] == 1
        assert data["model"] == "gemma4-e4b-q4-local:latest"


class TestOpenClawCompatibilityRegressions:
    def test_mixed_transcript_history_with_tool_payload_routes_only_to_compatible_provider(
        self, client, auth_headers
    ):
        openai = FakeProvider(
            "openai",
            default_model="gemma4-e4b-q4-local:latest",
            response_text="tool aware",
            discovered_models=["gemma4-e4b-q4-local:latest", "gemma4-31b-q4-local:latest"],
            resolved_tier_models={
                "premium": "gemma4-31b-q4-local:latest",
                "standard": "gemma4-e4b-q4-local:latest",
                "basic": "gemma4-e4b-q4-local:latest",
            },
        )
        anthropic = FakeProvider(
            "anthropic",
            default_model="claude-test",
            validation_error="anthropic provider does not support tool call transcripts in controller 1.2.0-rc",
            discovered_models=["claude-test"],
            resolved_tier_models={
                "premium": "claude-test",
                "standard": "claude-test",
                "basic": "claude-test",
            },
        )
        routes.providers = {"openai": openai, "anthropic": anthropic}
        routes.fallback_provider = FallbackProvider(routes.providers)

        payload = {
            "model": "controller",
            "messages": [
                {
                    "role": "system",
                    "content": "You are the external provider boundary for OpenClaw. Preserve transcript fidelity.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Find the deployment status and summarize it."}
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "fetch_status", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "cluster healthy; last rollout finished 2 minutes ago",
                },
                {
                    "role": "user",
                    "content": "Now explain whether we can proceed to smoke tests.",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "fetch_status",
                        "description": "Fetch deployment status",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
            "response_format": {"type": "json_object"},
            "metadata": {"session": "openclaw-replay-1"},
        }

        response = client.post("/v1/chat/completions", headers=auth_headers, json=payload)
        assert response.status_code == 200
        assert len(openai.calls) == 1
        assert openai.calls[0].request.messages[1].content == "Find the deployment status and summarize it."
        assert openai.calls[0].request.messages[3].role == Role.TOOL
        assert openai.calls[0].request.tools[0]["function"]["name"] == "fetch_status"

        data = response.json()
        assert data["controller_metadata"]["mode"] == "controller"
        assert data["controller_metadata"]["compatible_providers"] == ["openai"]
        assert data["controller_metadata"]["provider_diagnostics"]["anthropic"].startswith(
            "anthropic provider does not support"
        )

    def test_controller_selected_model_is_propagated_to_provider(self, client, auth_headers, monkeypatch):
        openai = FakeProvider(
            "openai",
            default_model="gemma4-e4b-q4-local:latest",
            response_text="routed",
            discovered_models=["gemma4-e4b-q4-local:latest", "gemma4-31b-q4-local:latest", "dgx-routed-model"],
            resolved_tier_models={
                "premium": "dgx-routed-model",
                "standard": "gemma4-e4b-q4-local:latest",
                "basic": "gemma4-e4b-q4-local:latest",
            },
        )
        routes.providers = {"openai": openai}
        routes.fallback_provider = FallbackProvider(routes.providers)
        monkeypatch.setattr(routes, "route_selector", StubRouteSelector("openai", "dgx-routed-model"))

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "controller",
                "messages": [{"role": "user", "content": "Route me intelligently"}],
            },
        )

        assert response.status_code == 200
        assert openai.calls[0].target_model == "dgx-routed-model"
        assert response.json()["controller_metadata"]["routed_model"] == "dgx-routed-model"

    def test_long_admin_operator_prompt_survives_normalization_and_routing(self, client, auth_headers):
        openai = FakeProvider(
            "openai",
            default_model="gemma4-e4b-q4-local:latest",
            response_text="processed",
            discovered_models=["gemma4-e4b-q4-local:latest", "gemma4-31b-q4-local:latest"],
            resolved_tier_models={
                "premium": "gemma4-31b-q4-local:latest",
                "standard": "gemma4-e4b-q4-local:latest",
                "basic": "gemma4-e4b-q4-local:latest",
            },
        )
        routes.providers = {"openai": openai}
        routes.fallback_provider = FallbackProvider(routes.providers)

        long_system = " ".join(
            [
                "You are operating as the admin controller boundary.",
                "Preserve auditability.",
                "Preserve tool state.",
                "Reject lossy normalization.",
            ]
            * 120
        )

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "controller",
                "messages": [
                    {"role": "system", "content": long_system},
                    {"role": "user", "content": "Summarize operator concerns."},
                ],
            },
        )

        assert response.status_code == 200
        assert len(openai.calls) == 1
        assert len(openai.calls[0].request.messages[0].content) > 4000

    def test_controller_fallback_uses_provider_specific_target_models(self, client, auth_headers, monkeypatch):
        anthropic = FakeProvider(
            "anthropic",
            default_model="claude-basic",
            fail_nonstream=True,
            discovered_models=["claude-basic", "claude-premium"],
            resolved_tier_models={
                "premium": "claude-premium",
                "standard": "claude-basic",
                "basic": "claude-basic",
            },
        )
        openai = FakeProvider(
            "openai",
            default_model="gemma4-e4b-q4-local:latest",
            response_text="fallback succeeded",
            discovered_models=["gemma4-e4b-q4-local:latest", "gemma4-31b-q4-local:latest"],
            resolved_tier_models={
                "premium": "gemma4-31b-q4-local:latest",
                "standard": "gemma4-e4b-q4-local:latest",
                "basic": "gemma4-e4b-q4-local:latest",
            },
        )
        routes.providers = {"anthropic": anthropic, "openai": openai}
        routes.fallback_provider = FallbackProvider(routes.providers)
        monkeypatch.setattr(routes, "route_selector", StubRouteSelector("anthropic", "claude-premium", tier="premium"))

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "controller",
                "messages": [{"role": "user", "content": "Route with provider-specific fallback"}],
            },
        )

        assert response.status_code == 200
        assert anthropic.calls[0].target_model == "claude-premium"
        assert openai.calls[0].target_model == "gemma4-31b-q4-local:latest"
        assert openai.calls[0].target_model != "claude-premium"
        assert response.json()["controller_metadata"]["actual_provider"] == "openai"
        assert response.json()["controller_metadata"]["actual_routed_model"] == "gemma4-31b-q4-local:latest"

    def test_direct_request_rejects_undiscovered_model(self, client, auth_headers):
        openai = FakeProvider(
            "openai",
            default_model="gemma4-e4b-q4-local:latest",
            discovered_models=["gemma4-e4b-q4-local:latest"],
            resolved_tier_models={
                "premium": "gemma4-e4b-q4-local:latest",
                "standard": "gemma4-e4b-q4-local:latest",
                "basic": "gemma4-e4b-q4-local:latest",
            },
        )
        routes.providers = {"openai": openai}
        routes.fallback_provider = FallbackProvider(routes.providers)

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "This should fail before upstream"}],
            },
        )

        assert response.status_code == 400
        payload = response.json()["error"]
        assert payload["message"] == "Requested model is not available on discovered backend inventory"
        assert payload["details"]["provider"] == "openai"
        assert payload["details"]["discovered_models"] == ["gemma4-e4b-q4-local:latest"]
        assert not openai.calls


class TestStreamingRegressions:
    def test_streaming_direct_provider_path_yields_transport_safe_chunks(self, client, auth_headers):
        openai = FakeProvider(
            "openai",
            default_model="gemma4-e4b-q4-local:latest",
            discovered_models=["gemma4-e4b-q4-local:latest"],
            stream_chunks=[
                'data: {"id":"chunk-1","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}\n\n',
                "data: [DONE]\n\n",
            ],
        )
        routes.providers = {"openai": openai}
        routes.fallback_provider = FallbackProvider(routes.providers)

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "openai/gemma4-e4b-q4-local:latest",
                "stream": True,
                "messages": [{"role": "user", "content": "stream directly"}],
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert 'data: {"id":"chunk-1"' in response.text
        assert "data: [DONE]" in response.text
        assert openai.calls[0].target_model == "gemma4-e4b-q4-local:latest"

    def test_streaming_fallback_provider_path_uses_second_provider_when_first_fails_before_emit(
        self, client, auth_headers
    ):
        anthropic = FakeProvider(
            "anthropic",
            default_model="claude-test",
            fail_stream_before_emit=True,
            discovered_models=["claude-test"],
            resolved_tier_models={
                "premium": "claude-test",
                "standard": "claude-test",
                "basic": "claude-test",
            },
        )
        openai = FakeProvider(
            "openai",
            default_model="gemma4-e4b-q4-local:latest",
            discovered_models=["gemma4-e4b-q4-local:latest"],
            stream_chunks=[
                'data: {"id":"chunk-2","choices":[{"index":0,"delta":{"content":"fallback ok"},"finish_reason":null}]}\n\n',
                "data: [DONE]\n\n",
            ],
        )
        routes.providers = {"anthropic": anthropic, "openai": openai}
        routes.fallback_provider = FallbackProvider(routes.providers)

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "controller",
                "stream": True,
                "force_provider": "anthropic",
                "messages": [{"role": "user", "content": "fallback stream"}],
            },
        )

        assert response.status_code == 200
        assert anthropic.calls[0].method == "chat_completion_stream"
        assert openai.calls[0].method == "chat_completion_stream"
        assert "fallback ok" in response.text
        assert "tuple" not in response.text

    def test_streaming_legacy_tuple_chunks_are_sanitized_before_http_response(self, client, auth_headers):
        openai = FakeProvider(
            "openai",
            default_model="gemma4-e4b-q4-local:latest",
            discovered_models=["gemma4-e4b-q4-local:latest"],
        )

        async def legacy_stream(request, target_model=None):
            openai.calls.append(ProviderCall("chat_completion_stream", target_model, request))
            yield ('data: {"id":"chunk-legacy","choices":[{"index":0,"delta":{"content":"legacy"},"finish_reason":null}]}\n\n', "openai")
            yield "data: [DONE]\n\n"

        openai.chat_completion_stream = legacy_stream  # type: ignore[assignment]
        routes.providers = {"openai": openai}
        routes.fallback_provider = FallbackProvider(routes.providers)

        response = client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "openai/gemma4-e4b-q4-local:latest",
                "stream": True,
                "messages": [{"role": "user", "content": "tuple stream"}],
            },
        )

        assert response.status_code == 200
        assert "legacy" in response.text
        assert "tuple" not in response.text


class TestValidationDiagnostics:
    def test_validation_error_logging_includes_context_and_body_preview(self, client, auth_headers, caplog):
        with caplog.at_level(logging.WARNING):
            response = client.post(
                "/v1/chat/completions",
                headers=auth_headers,
                json={
                    "model": "controller",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.com/test.png"},
                                }
                            ],
                        }
                    ],
                },
            )

        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["message"] == "request validation failed"
        assert payload["error"]["type"] == "invalid_request_error"
        assert payload["error"]["details"]
        assert response.headers["x-request-id"] == payload["error"]["request_id"]

        log_text = caplog.text
        assert "Request validation failed method=POST path=/v1/chat/completions" in log_text
        assert "content_type=application/json" in log_text
        assert "image_url" in log_text
        assert payload["error"]["request_id"] in log_text


class TestConfigurationRegressions:
    def test_settings_support_comma_separated_fallback_providers(self):
        settings = Settings(
            CONTROLLER_API_KEY="test-api-key-for-integration-tests-12345",
            FALLBACK_PROVIDERS="openai,anthropic",
        )
        assert settings.FALLBACK_PROVIDERS == ["openai", "anthropic"]

    def test_settings_provider_models_include_tier_overrides(self):
        settings = Settings(
            CONTROLLER_API_KEY="test-api-key-for-integration-tests-12345",
            OPENAI_API_KEY="dummy",
            OPENAI_DEFAULT_MODEL="dgx-default",
            OPENAI_MODEL_PREMIUM="dgx-premium",
            OPENAI_MODEL_STANDARD="dgx-standard",
            OPENAI_MODEL_BASIC="dgx-basic",
        )
        assert settings.get_tier_model("openai", "premium") == "dgx-premium"
        assert settings.get_provider_models("openai") == [
            "dgx-default",
            "dgx-premium",
            "dgx-standard",
            "dgx-basic",
        ]


class TestDiscoveryAndRouting:
    @pytest.mark.asyncio
    async def test_openai_model_discovery_uses_real_backend_inventory(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/models"
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "gemma4-e4b-q4-local:latest"},
                        {"id": "gemma4-31b-q4-local:latest"},
                    ],
                },
            )

        provider = OpenAIProvider(
            {
                "api_key": "dummy",
                "base_url": "http://backend.local/v1",
                "default_model": "",
                "tier_models": {},
                "routing_require_discovered_models": True,
            }
        )
        provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            catalog = await provider.refresh_model_inventory(force=True)
            assert catalog.discovered_models == [
                "gemma4-e4b-q4-local:latest",
                "gemma4-31b-q4-local:latest",
            ]
            assert catalog.resolved_tier_models["premium"] == "gemma4-31b-q4-local:latest"
            assert catalog.resolved_tier_models["standard"] == "gemma4-e4b-q4-local:latest"
            assert catalog.resolved_tier_models["basic"] == "gemma4-e4b-q4-local:latest"
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_invalid_configured_models_degrade_without_inventing_fake_ids(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "gemma4-e4b-q4-local:latest"},
                        {"id": "gemma4-31b-q4-local:latest"},
                    ],
                },
            )

        provider = OpenAIProvider(
            {
                "api_key": "dummy",
                "base_url": "http://backend.local/v1",
                "default_model": "gpt-4o-mini",
                "tier_models": {
                    "premium": "gpt-4o",
                    "standard": "gpt-4o-mini",
                    "basic": "gpt-4o-mini",
                },
                "routing_require_discovered_models": True,
            }
        )
        provider.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            catalog = await provider.refresh_model_inventory(force=True)
            assert catalog.inventory_status == "degraded"
            assert catalog.resolved_tier_models["premium"] == "gemma4-31b-q4-local:latest"
            assert catalog.resolved_tier_models["standard"] == "gemma4-e4b-q4-local:latest"
            assert catalog.resolved_tier_models["basic"] == "gemma4-e4b-q4-local:latest"
            assert "gpt-4o" not in catalog.discovered_models
            assert "gpt-4o-mini" not in catalog.discovered_models
            assert any("gpt-4o" in issue for issue in catalog.inventory_issues)
        finally:
            await provider.close()


class TestResponseExtraction:
    def _provider(self) -> OpenAIProvider:
        provider = OpenAIProvider(
            {
                "api_key": "dummy",
                "base_url": "http://backend.local/v1",
                "default_model": "gemma4-e4b-q4-local:latest",
                "tier_models": {
                    "premium": "gemma4-31b-q4-local:latest",
                    "standard": "gemma4-e4b-q4-local:latest",
                    "basic": "gemma4-e4b-q4-local:latest",
                },
                "routing_require_discovered_models": True,
            }
        )
        provider.discovered_models = [
            "gemma4-e4b-q4-local:latest",
            "gemma4-31b-q4-local:latest",
        ]
        provider.resolved_tier_models = {
            "premium": "gemma4-31b-q4-local:latest",
            "standard": "gemma4-e4b-q4-local:latest",
            "basic": "gemma4-e4b-q4-local:latest",
        }
        provider.routing_default_model = "gemma4-e4b-q4-local:latest"
        return provider

    def test_openai_parser_prefers_content_when_present(self):
        provider = self._provider()
        response = provider.parse_response(
            {
                "id": "resp-1",
                "created": 1,
                "model": "gemma4-e4b-q4-local:latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Меня зовут Толя.",
                            "reasoning": "Final answer: не использовать",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
            "gemma4-e4b-q4-local:latest",
        )
        assert response.choices[0].message.content == "Меня зовут Толя."

    def test_openai_parser_extracts_final_answer_without_leaking_reasoning(self):
        provider = self._provider()
        request = ChatCompletionRequest(
            model="controller",
            messages=[{"role": "user", "content": "Как тебя зовут?"}],
        ).normalize()
        response = provider.parse_response(
            {
                "id": "resp-2",
                "created": 1,
                "model": "gemma4-e4b-q4-local:latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning": "Constraints: answer only in Russian.\nDo not mention you are a model.\nFinal answer: Меня зовут Толя.",
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
            "gemma4-e4b-q4-local:latest",
            request=request,
        )
        assert response.choices[0].message.content == "Меня зовут Толя."
        assert "Constraints" not in response.choices[0].message.content

    def test_openai_parser_returns_safe_fallback_when_reasoning_has_no_clean_final_answer(self):
        provider = self._provider()
        request = ChatCompletionRequest(
            model="controller",
            messages=[{"role": "user", "content": "Представься кратко."}],
        ).normalize()
        response = provider.parse_response(
            {
                "id": "resp-3",
                "created": 1,
                "model": "gemma4-e4b-q4-local:latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning": "User asks in Russian. Constraints: answer in Russian. I need to keep persona. Response should be short.",
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
            "gemma4-e4b-q4-local:latest",
            request=request,
        )
        content = response.choices[0].message.content
        assert content
        assert "Constraints" not in content
        assert "I need" not in content
        assert "Извините" in content

    def test_openai_parser_preserves_tool_calls_when_content_is_empty(self):
        provider = self._provider()
        response = provider.parse_response(
            {
                "id": "resp-4",
                "created": 1,
                "model": "gemma4-e4b-q4-local:latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning": "Final answer: ignore this because tool calls exist",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
            "gemma4-e4b-q4-local:latest",
        )
        message = response.choices[0].message
        assert message.tool_calls is not None
        assert message.content in {None, ""}


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__)), "-v"]))
