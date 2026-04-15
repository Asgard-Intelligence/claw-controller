import pytest

from controller.core.provider_contracts import ApiDialect, ModelDescriptor, ProviderDescriptor, ProviderKind, ValidationKind
from controller.core.provider_registry_v140 import ProviderRegistryV140
from controller.core.validation_service import RouteValidationService


@pytest.mark.asyncio
async def test_openai_probe_prefers_chat_when_responses_unavailable(monkeypatch):
    registry = ProviderRegistryV140()
    registry.register_provider(
        ProviderDescriptor(
            provider_id="p1",
            label="P1",
            kind=ProviderKind.CLOUD,
            dialect_family="openai_compatible",
            base_url="https://example.invalid",
            validation_base_url="https://example.invalid",
            health_url="https://example.invalid/models",
            credential_env=None,
            default_model_id="m1",
        )
    )
    registry.register_model(
        ModelDescriptor(
            provider_id="p1",
            model_id="m1",
            aliases=[],
            family="gpt",
            context_window=8000,
            max_output_tokens=1000,
            supports_tools=False,
            supports_vision=False,
            supports_json_mode=True,
            supports_streaming=True,
            supports_system_prompt=True,
            system_prompt_mode="native",
            preferred_api_dialects=[ApiDialect.OPENAI_RESPONSES],
            allowed_api_dialects=[ApiDialect.OPENAI_RESPONSES, ApiDialect.OPENAI_CHAT_COMPLETIONS],
        )
    )

    service = RouteValidationService(registry=registry)
    route_key = registry.make_route_key("p1", "m1", ApiDialect.OPENAI_RESPONSES)

    calls = {"count": 0}

    async def fake_post_probe(url, headers, payload):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": 404, "text": "not found"}
        return {"status": 200, "text": "ok"}

    monkeypatch.setattr(service, "_post_probe", fake_post_probe)

    result = await service.get_route_validation(route_key, force_refresh=True)
    assert result.ok is True
    assert result.kind == ValidationKind.OK
    assert result.preferred_api_dialect == ApiDialect.OPENAI_CHAT_COMPLETIONS


@pytest.mark.asyncio
async def test_ollama_catalog_404_not_fatal(monkeypatch):
    registry = ProviderRegistryV140()
    registry.register_provider(
        ProviderDescriptor(
            provider_id="ollama",
            label="Ollama",
            kind=ProviderKind.LOCAL,
            dialect_family="ollama_native",
            base_url="http://localhost:11434",
            validation_base_url="http://localhost:11434",
            health_url="http://localhost:11434/api/tags",
            credential_env=None,
            default_model_id="local-model",
        )
    )
    registry.register_model(
        ModelDescriptor(
            provider_id="ollama",
            model_id="local-model",
            aliases=[],
            family="local",
            context_window=32000,
            max_output_tokens=1024,
            supports_tools=False,
            supports_vision=False,
            supports_json_mode=False,
            supports_streaming=True,
            supports_system_prompt=True,
            system_prompt_mode="native",
            preferred_api_dialects=[ApiDialect.OLLAMA_CHAT],
            allowed_api_dialects=[ApiDialect.OLLAMA_CHAT],
        )
    )

    service = RouteValidationService(registry=registry)
    route_key = registry.make_route_key("ollama", "local-model", ApiDialect.OLLAMA_CHAT)

    async def fake_get_probe(url, headers):
        return {"status": 404, "text": "not found"}

    monkeypatch.setattr(service, "_get_probe", fake_get_probe)

    result = await service.get_route_validation(route_key, force_refresh=True)
    assert result.ok is True
    assert result.validated is False
    assert result.kind == ValidationKind.OK
