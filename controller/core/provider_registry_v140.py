"""Route-aware provider registry for Controller v1.4."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .capability_matrix import CapabilityMatrix
from .provider_contracts import (
    ApiDialect,
    HealthState,
    ModelDescriptor,
    ProviderDescriptor,
    RouteHealthSnapshot,
    RouteTarget,
    RoutingRequirements,
    ValidationResult,
)


class ProviderRegistryV140:
    """Metadata-only registry with route-level indexes and caches."""

    def __init__(self):
        self._providers: Dict[str, ProviderDescriptor] = {}
        self._models: Dict[Tuple[str, str], ModelDescriptor] = {}
        self._model_aliases: Dict[Tuple[str, str], str] = {}
        self._routes: Dict[str, RouteTarget] = {}
        self._provider_routes: Dict[str, List[str]] = {}
        self._health_cache: Dict[str, RouteHealthSnapshot] = {}
        self._validation_cache: Dict[str, ValidationResult] = {}

    @staticmethod
    def make_route_key(provider_id: str, model_id: str, api_dialect: ApiDialect) -> str:
        return f"{provider_id}:{model_id}:{api_dialect.value}"

    def register_provider(self, provider: ProviderDescriptor) -> None:
        self._providers[provider.provider_id] = provider
        self._provider_routes.setdefault(provider.provider_id, [])

    def register_model(self, model: ModelDescriptor, create_routes: bool = True) -> None:
        if model.provider_id not in self._providers:
            raise ValueError(f"Provider not registered: {model.provider_id}")

        self._models[(model.provider_id, model.model_id)] = model
        self._model_aliases[(model.provider_id, model.model_id.lower())] = model.model_id
        for alias in model.aliases:
            self._model_aliases[(model.provider_id, alias.lower())] = model.model_id

        if not create_routes:
            return

        provider = self._providers[model.provider_id]
        for dialect in model.allowed_api_dialects:
            route_key = self.make_route_key(model.provider_id, model.model_id, dialect)
            route = RouteTarget(
                route_key=route_key,
                provider_id=model.provider_id,
                model_id=model.model_id,
                api_dialect=dialect,
                base_url=provider.base_url,
                credential_env=provider.credential_env,
                kind=provider.kind,
                timeout_profile=provider.timeout_profile,
                capabilities_snapshot=CapabilityMatrix.build_snapshot(provider, model),
                metadata={},
            )
            self.register_route(route)

    def register_route(self, route: RouteTarget) -> None:
        self._routes[route.route_key] = route
        routes = self._provider_routes.setdefault(route.provider_id, [])
        if route.route_key not in routes:
            routes.append(route.route_key)

    def get_provider(self, provider_id: str) -> Optional[ProviderDescriptor]:
        return self._providers.get(provider_id)

    def get_model(self, provider_id: str, model_id_or_alias: str) -> Optional[ModelDescriptor]:
        resolved = self.resolve_model_alias(provider_id, model_id_or_alias)
        if not resolved:
            return None
        return self._models.get((provider_id, resolved))

    def resolve_model_alias(self, provider_id: str, model_id_or_alias: str) -> Optional[str]:
        return self._model_aliases.get((provider_id, model_id_or_alias.lower()))

    def list_routes(self, enabled_only: bool = True) -> List[RouteTarget]:
        result: List[RouteTarget] = []
        for route in self._routes.values():
            provider = self._providers.get(route.provider_id)
            if enabled_only and provider and not provider.enabled:
                continue
            result.append(route)
        return result

    def get_route(self, route_key: str) -> Optional[RouteTarget]:
        return self._routes.get(route_key)

    def get_routes_for_provider(self, provider_id: str) -> List[RouteTarget]:
        return [self._routes[key] for key in self._provider_routes.get(provider_id, []) if key in self._routes]

    def set_route_health(self, snapshot: RouteHealthSnapshot) -> None:
        self._health_cache[snapshot.route_key] = snapshot

    def get_route_health(self, route_key: str) -> Optional[RouteHealthSnapshot]:
        return self._health_cache.get(route_key)

    def set_validation_result(self, route_key: str, result: ValidationResult) -> None:
        self._validation_cache[route_key] = result

    def get_validation_result(self, route_key: str) -> Optional[ValidationResult]:
        return self._validation_cache.get(route_key)

    def get_aggregate_provider_health(self, provider_id: str) -> Dict[str, object]:
        route_keys = self._provider_routes.get(provider_id, [])
        snapshots = [self._health_cache.get(key) for key in route_keys if key in self._health_cache]

        if not snapshots:
            return {
                "provider_id": provider_id,
                "status": HealthState.UNKNOWN.value,
                "healthy_routes": 0,
                "total_routes": len(route_keys),
            }

        statuses = [s.status for s in snapshots]
        healthy = sum(1 for s in statuses if s == HealthState.HEALTHY)

        if healthy > 0:
            status = HealthState.HEALTHY
        elif any(s == HealthState.DEGRADED for s in statuses):
            status = HealthState.DEGRADED
        elif all(s == HealthState.UNAUTHORIZED for s in statuses):
            status = HealthState.UNAUTHORIZED
        elif all(s == HealthState.UNREACHABLE for s in statuses):
            status = HealthState.UNREACHABLE
        else:
            status = HealthState.UNKNOWN

        return {
            "provider_id": provider_id,
            "status": status.value,
            "healthy_routes": healthy,
            "total_routes": len(route_keys),
        }

    def find_candidate_routes(self, requirements: RoutingRequirements) -> List[RouteTarget]:
        candidates: List[RouteTarget] = []

        for route in self.list_routes(enabled_only=True):
            if not self._route_meets_requirements(route, requirements):
                continue
            candidates.append(route)

        # deterministic ordering: hints first, then larger context window
        def sort_key(route: RouteTarget) -> Tuple[int, int, str]:
            pref = 0
            if requirements.provider_hint and route.provider_id == requirements.provider_hint:
                pref += 20
            if requirements.model_hint and route.model_id == requirements.model_hint:
                pref += 20
            if requirements.locality_preference and route.kind == requirements.locality_preference:
                pref += 10
            context = int(route.capabilities_snapshot.get("context_window", 0))
            return (-pref, -context, route.route_key)

        candidates.sort(key=sort_key)
        return candidates

    def _route_meets_requirements(self, route: RouteTarget, requirements: RoutingRequirements) -> bool:
        caps = route.capabilities_snapshot

        if requirements.requires_tools and not caps.get("supports_tools", False):
            return False
        if requirements.requires_vision and not caps.get("supports_vision", False):
            return False
        if requirements.requires_json_mode and not caps.get("supports_json_mode", False):
            return False
        if requirements.requires_streaming and not caps.get("supports_streaming", False):
            return False

        if int(caps.get("context_window", 0)) < requirements.estimated_prompt_tokens:
            return False

        return True

    def get_default_route(self, provider_id: str) -> Optional[RouteTarget]:
        provider = self.get_provider(provider_id)
        if not provider:
            return None

        model = self.get_model(provider_id, provider.default_model_id)
        if not model:
            return None

        preferred = model.preferred_api_dialects[0] if model.preferred_api_dialects else model.allowed_api_dialects[0]
        route_key = self.make_route_key(provider_id, model.model_id, preferred)
        route = self.get_route(route_key)
        if route:
            return route

        # fallback to any route for default model
        for dialect in model.allowed_api_dialects:
            key = self.make_route_key(provider_id, model.model_id, dialect)
            fallback = self.get_route(key)
            if fallback:
                return fallback

        return None

    def get_routes_for_model(self, provider_id: str, model_id_or_alias: str) -> List[RouteTarget]:
        model = self.get_model(provider_id, model_id_or_alias)
        if not model:
            return []
        result: List[RouteTarget] = []
        for dialect in model.allowed_api_dialects:
            key = self.make_route_key(provider_id, model.model_id, dialect)
            route = self.get_route(key)
            if route:
                result.append(route)
        return result

    def with_runtime_base_url(self, route_key: str, base_url: str) -> Optional[RouteTarget]:
        route = self.get_route(route_key)
        if not route:
            return None
        return replace(route, base_url=base_url)
