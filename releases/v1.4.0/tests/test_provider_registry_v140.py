from controller.core.provider_contracts import ApiDialect, ModelDescriptor, ProviderDescriptor, ProviderKind, RoutingRequirements
from controller.core.provider_registry_v140 import ProviderRegistryV140


def _build_registry():
    registry = ProviderRegistryV140()
    registry.register_provider(
        ProviderDescriptor(
            provider_id="openai",
            label="OpenAI",
            kind=ProviderKind.CLOUD,
            dialect_family="openai_compatible",
            base_url="https://api.openai.com",
            validation_base_url="https://api.openai.com",
            health_url="https://api.openai.com/models",
            credential_env="OPENAI_API_KEY",
            default_model_id="gpt-4",
        )
    )
    registry.register_model(
        ModelDescriptor(
            provider_id="openai",
            model_id="gpt-4",
            aliases=["gpt4"],
            family="gpt",
            context_window=8192,
            max_output_tokens=4096,
            supports_tools=True,
            supports_vision=False,
            supports_json_mode=True,
            supports_streaming=True,
            supports_system_prompt=True,
            system_prompt_mode="native",
            preferred_api_dialects=[ApiDialect.OPENAI_CHAT_COMPLETIONS],
            allowed_api_dialects=[ApiDialect.OPENAI_CHAT_COMPLETIONS, ApiDialect.OPENAI_RESPONSES],
        )
    )
    return registry


def test_provider_model_separation_and_alias_resolution():
    registry = _build_registry()

    provider = registry.get_provider("openai")
    model = registry.get_model("openai", "gpt4")

    assert provider is not None
    assert provider.provider_id == "openai"
    assert model is not None
    assert model.model_id == "gpt-4"


def test_find_candidate_routes_filters_by_capabilities_and_context():
    registry = _build_registry()

    req = RoutingRequirements(
        estimated_prompt_tokens=7000,
        requires_tools=True,
        requires_streaming=True,
    )
    candidates = registry.find_candidate_routes(req)
    assert candidates
    assert all(c.capabilities_snapshot["supports_tools"] for c in candidates)

    req_too_large = RoutingRequirements(estimated_prompt_tokens=900000)
    assert registry.find_candidate_routes(req_too_large) == []
