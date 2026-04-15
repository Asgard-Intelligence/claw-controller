import pytest

from controller.core.health_service import RouteHealthService
from controller.core.hybrid_router import HybridRouter
from controller.core.provider_contracts import ApiDialect, HealthState, ModelDescriptor, ProviderDescriptor, ProviderKind, RoutingRequirements, ValidationKind, ValidationResult
from controller.core.provider_registry_v140 import ProviderRegistryV140
from controller.core.routing_policy_v140 import RoutingPolicyV140
from controller.core.session_manager import IdentityState, SessionState
from datetime import datetime


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


def _session():
    ident = IdentityState(
        name="agent",
        loaded_from="test",
        loaded_at=datetime.utcnow(),
        hash="h",
        system_prompt="You are agent",
    )
    return SessionState(
        session_id="s1",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        identity=ident,
    )


@pytest.mark.asyncio
async def test_router_selects_healthy_route_and_builds_fallback_chain():
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
            context_window=16000,
            max_output_tokens=1000,
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

    routes = registry.get_routes_for_model("p1", "m1")
    route_a = routes[0].route_key
    route_b = routes[1].route_key

    validation = {
        route_a: ValidationResult(ok=True, validated=True, kind=ValidationKind.OK, message="ok"),
        route_b: ValidationResult(ok=True, validated=True, kind=ValidationKind.OK, message="ok"),
    }
    health = {
        route_a: type("H", (), {"status": HealthState.HEALTHY})(),
        route_b: type("H", (), {"status": HealthState.DEGRADED})(),
    }

    router = HybridRouter(
        registry=registry,
        validation_service=StubValidation(validation),
        health_service=StubHealth(health),
        policy=RoutingPolicyV140(fallback_max_hops=2),
        default_provider="p1",
    )

    decision = await router.select_route(
        _session(),
        RoutingRequirements(estimated_prompt_tokens=1000, requires_tools=True),
    )

    assert decision.route_key == route_a
    assert route_b in decision.fallback_chain
