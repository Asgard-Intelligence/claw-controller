import pytest

from controller.core.provider_contracts import ApiDialect, ModelDescriptor, ProviderDescriptor, ProviderKind, RoutingRequirements, ValidationKind, ValidationResult
from controller.core.provider_registry_v140 import ProviderRegistryV140
from controller.core.routing_policy_v140 import RoutingPolicyV140
from controller.core.hybrid_router import HybridRouter
from controller.core.session_manager import FileSessionStore, IdentityManager, SessionManager


class StubValidation:
    def __init__(self, mapping):
        self.mapping = mapping

    async def get_route_validation(self, route_key, force_refresh=False):
        return self.mapping[route_key]


class StubHealth:
    def __init__(self, mapping):
        self.mapping = mapping

    async def get_route_health(self, route_key, force_refresh=False):
        return self.mapping[route_key]


@pytest.mark.asyncio
async def test_identity_and_context_preserved_when_route_changes(tmp_path):
    identity_dir = tmp_path / "identity"
    (identity_dir / "agentx").mkdir(parents=True)
    (identity_dir / "agentx" / "IDENTITY.md").write_text("Your name is RouteAgent.")

    manager = SessionManager(
        store=FileSessionStore(base_path=str(tmp_path / "sessions")),
        identity_manager=IdentityManager(identity_dir=str(identity_dir)),
    )
    session = await manager.create_session(agent_id="agentx")
    await manager.add_message(session.session_id, "user", "Меня зовут Алекс", tokens=4)
    await manager.add_message(session.session_id, "assistant", "Приятно познакомиться, Алекс", tokens=5)

    registry = ProviderRegistryV140()
    registry.register_provider(
        ProviderDescriptor(
            provider_id="p1",
            label="P1",
            kind=ProviderKind.CLOUD,
            dialect_family="openai_compatible",
            base_url="https://x",
            validation_base_url="https://x",
            health_url="https://x/models",
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
            context_window=4096,
            max_output_tokens=512,
            supports_tools=False,
            supports_vision=False,
            supports_json_mode=True,
            supports_streaming=True,
            supports_system_prompt=True,
            system_prompt_mode="native",
            preferred_api_dialects=[ApiDialect.OPENAI_CHAT_COMPLETIONS],
            allowed_api_dialects=[ApiDialect.OPENAI_CHAT_COMPLETIONS],
        )
    )
    route_key = registry.make_route_key("p1", "m1", ApiDialect.OPENAI_CHAT_COMPLETIONS)

    validation = {route_key: ValidationResult(ok=True, validated=True, kind=ValidationKind.OK, message="ok")}
    from controller.core.provider_contracts import HealthState
    health = {route_key: type("H", (), {"status": HealthState.HEALTHY})()}

    router = HybridRouter(
        registry=registry,
        validation_service=StubValidation(validation),
        health_service=StubHealth(health),
        policy=RoutingPolicyV140(),
        default_provider="p1",
    )

    session = await manager.get_session(session.session_id)
    decision = await router.select_route(session, RoutingRequirements(estimated_prompt_tokens=100))

    assert session.identity.system_prompt
    assert "RouteAgent" in session.identity.system_prompt
    assert decision.route_key == route_key
