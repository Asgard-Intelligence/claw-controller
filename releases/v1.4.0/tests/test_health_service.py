from datetime import datetime, timedelta

import pytest

from controller.core.health_service import RouteHealthService
from controller.core.provider_contracts import ApiDialect, HealthState, ModelDescriptor, ProviderDescriptor, ProviderKind, ValidationKind, ValidationResult
from controller.core.provider_registry_v140 import ProviderRegistryV140


class StubValidationService:
    def __init__(self, result):
        self.result = result

    async def get_route_validation(self, route_key, force_refresh=False):
        return self.result


@pytest.mark.asyncio
async def test_health_maps_credential_failure_to_unauthorized():
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
            credential_env="OPENAI_API_KEY",
            default_model_id="m1",
        )
    )
    registry.register_model(
        ModelDescriptor(
            provider_id="p1",
            model_id="m1",
            aliases=[],
            family="gpt",
            context_window=8192,
            max_output_tokens=1024,
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
    validator = StubValidationService(
        ValidationResult(
            ok=False,
            validated=True,
            kind=ValidationKind.CREDENTIAL,
            message="Missing key",
        )
    )
    service = RouteHealthService(registry=registry, validation_service=validator)

    snapshot = await service.get_route_health(route_key, force_refresh=True)
    assert snapshot.status == HealthState.UNAUTHORIZED


@pytest.mark.asyncio
async def test_health_returns_stale_snapshot_without_refresh():
    registry = ProviderRegistryV140()
    validator = StubValidationService(
        ValidationResult(ok=True, validated=True, kind=ValidationKind.OK, message="ok")
    )
    service = RouteHealthService(registry=registry, validation_service=validator, cache_ttl_seconds=1)

    service._cache["r1"] = (
        0.0,
        type("Snap", (), {
            "route_key": "r1",
            "status": HealthState.HEALTHY,
            "last_checked_at": datetime.utcnow() - timedelta(seconds=10),
            "last_success_at": datetime.utcnow() - timedelta(seconds=10),
            "failure_kind": None,
            "failure_message": None,
            "host_reachable": True,
            "runtime_reachable": True,
            "validated_model_catalog": True,
            "preferred_api_dialect": None,
        })(),
    )

    snapshot = await service.get_route_health("r1", force_refresh=False)
    assert snapshot.status == HealthState.STALE
