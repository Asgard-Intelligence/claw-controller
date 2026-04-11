"""Integration and regression tests for Controller 1.1.0 RC."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from controller.core.config import Settings, get_settings
from controller.core.intelligence import ConfidenceScore, IntentType, RiskLevel, RouteDecision, SafetyScore
from controller.core.providers import FallbackProvider
from controller.main import create_app
from controller.models.schemas import (
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
        return {"status": "healthy", "provider": self.name, "latency_ms": 1}

    async def close(self):
        self.closed = True


class StubRouteSelector:
    def __init__(self, selected_provider: str, selected_model: str, tier: str = "standard"):
        self.selected_provider = selected_provider
        self.selected_model = selected_model
        self.tier = tier

    def select(self, request, confidence, safety, intent, provider_health, available_providers):
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
        assert data["version"] == "1.1.0-rc"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Controller"
        assert data["version"] == "1.1.0-rc"

    def test_models_requires_auth(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "authentication_error"

    def test_models_with_auth(self, client, auth_headers):
        response = client.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200
        model_ids = [item["id"] for item in response.json()["data"]]
        assert "controller" in model_ids


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
        openai = FakeProvider("openai", default_model="dgx-mini", response_text="normalized")
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
        openai = FakeProvider("openai", default_model="dgx-standard", response_text="tool aware")
        anthropic = FakeProvider(
            "anthropic",
            default_model="claude-test",
            validation_error="anthropic provider does not support tool call transcripts in controller 1.1.0",
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
        openai = FakeProvider("openai", default_model="dgx-default", response_text="routed")
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
        openai = FakeProvider("openai", default_model="dgx-admin", response_text="processed")
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


class TestStreamingRegressions:
    def test_streaming_direct_provider_path_yields_transport_safe_chunks(self, client, auth_headers):
        openai = FakeProvider(
            "openai",
            default_model="dgx-mini",
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
        assert "data: {\"id\":\"chunk-1\"" in response.text
        assert "data: [DONE]" in response.text
        assert openai.calls[0].target_model == "gemma4-e4b-q4-local:latest"

    def test_streaming_fallback_provider_path_uses_second_provider_when_first_fails_before_emit(
        self, client, auth_headers
    ):
        anthropic = FakeProvider(
            "anthropic",
            default_model="claude-test",
            fail_stream_before_emit=True,
        )
        openai = FakeProvider(
            "openai",
            default_model="dgx-fallback",
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__)), "-v"]))
