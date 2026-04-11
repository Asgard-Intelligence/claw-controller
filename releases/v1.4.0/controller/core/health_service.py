"""Route-level health service with TTL cache, separate from validation."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import httpx

from .provider_contracts import HealthState, RouteHealthSnapshot, ValidationKind
from .provider_registry_v140 import ProviderRegistryV140
from .validation_service import RouteValidationService


class RouteHealthService:
    def __init__(
        self,
        registry: ProviderRegistryV140,
        validation_service: RouteValidationService,
        cache_ttl_seconds: int = 30,
        timeout_seconds: float = 5.0,
    ):
        self.registry = registry
        self.validation_service = validation_service
        self.cache_ttl_seconds = cache_ttl_seconds
        self.timeout_seconds = timeout_seconds
        self._cache: Dict[str, Tuple[float, RouteHealthSnapshot]] = {}

    async def get_route_health(self, route_key: str, force_refresh: bool = False) -> RouteHealthSnapshot:
        now = time.time()
        if route_key in self._cache and not force_refresh:
            ts, snapshot = self._cache[route_key]
            age = now - ts
            if age < self.cache_ttl_seconds:
                return snapshot
            # Keep stale snapshot available with explicit state.
            stale = RouteHealthSnapshot(
                route_key=snapshot.route_key,
                status=HealthState.STALE,
                last_checked_at=datetime.utcnow(),
                last_success_at=snapshot.last_success_at,
                failure_kind=snapshot.failure_kind,
                failure_message="Health snapshot stale",
                host_reachable=snapshot.host_reachable,
                runtime_reachable=snapshot.runtime_reachable,
                validated_model_catalog=snapshot.validated_model_catalog,
                preferred_api_dialect=snapshot.preferred_api_dialect,
            )
            self.registry.set_route_health(stale)
            return stale

        snapshot = await self._probe_route(route_key)
        self._cache[route_key] = (now, snapshot)
        self.registry.set_route_health(snapshot)
        return snapshot

    async def _probe_route(self, route_key: str) -> RouteHealthSnapshot:
        route = self.registry.get_route(route_key)
        if not route:
            return RouteHealthSnapshot(
                route_key=route_key,
                status=HealthState.UNKNOWN,
                last_checked_at=datetime.utcnow(),
                failure_kind=ValidationKind.UNKNOWN,
                failure_message="Route not found",
            )

        provider = self.registry.get_provider(route.provider_id)
        if not provider:
            return RouteHealthSnapshot(
                route_key=route_key,
                status=HealthState.UNKNOWN,
                last_checked_at=datetime.utcnow(),
                failure_kind=ValidationKind.UNKNOWN,
                failure_message="Provider not found",
            )

        validation = await self.validation_service.get_route_validation(route_key)

        if validation.kind == ValidationKind.CREDENTIAL:
            return RouteHealthSnapshot(
                route_key=route_key,
                status=HealthState.UNAUTHORIZED,
                last_checked_at=datetime.utcnow(),
                failure_kind=validation.kind,
                failure_message=validation.message,
                validated_model_catalog=validation.catalog_checked,
                preferred_api_dialect=validation.preferred_api_dialect,
                host_reachable=True,
                runtime_reachable=True,
            )

        if validation.kind == ValidationKind.MODEL:
            return RouteHealthSnapshot(
                route_key=route_key,
                status=HealthState.INVALID_MODEL,
                last_checked_at=datetime.utcnow(),
                failure_kind=validation.kind,
                failure_message=validation.message,
                validated_model_catalog=validation.catalog_checked,
                preferred_api_dialect=validation.preferred_api_dialect,
                host_reachable=True,
                runtime_reachable=True,
            )

        if validation.kind in {ValidationKind.TRANSPORT, ValidationKind.ENDPOINT} and not validation.ok:
            unreachable = validation.kind == ValidationKind.TRANSPORT
            return RouteHealthSnapshot(
                route_key=route_key,
                status=HealthState.UNREACHABLE if unreachable else HealthState.DEGRADED,
                last_checked_at=datetime.utcnow(),
                failure_kind=validation.kind,
                failure_message=validation.message,
                validated_model_catalog=validation.catalog_checked,
                preferred_api_dialect=validation.preferred_api_dialect,
                host_reachable=False if unreachable else True,
                runtime_reachable=False if unreachable else True,
            )

        # Health is separate from validation: probe health endpoint now.
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.get(provider.health_url)

            if resp.status_code in {200, 201}:
                return RouteHealthSnapshot(
                    route_key=route_key,
                    status=HealthState.HEALTHY,
                    last_checked_at=datetime.utcnow(),
                    last_success_at=datetime.utcnow(),
                    validated_model_catalog=validation.catalog_checked,
                    preferred_api_dialect=validation.preferred_api_dialect,
                    host_reachable=True,
                    runtime_reachable=True,
                )

            if resp.status_code in {401, 403}:
                return RouteHealthSnapshot(
                    route_key=route_key,
                    status=HealthState.UNAUTHORIZED,
                    last_checked_at=datetime.utcnow(),
                    failure_kind=ValidationKind.CREDENTIAL,
                    failure_message="Health endpoint unauthorized",
                    validated_model_catalog=validation.catalog_checked,
                    preferred_api_dialect=validation.preferred_api_dialect,
                    host_reachable=True,
                    runtime_reachable=True,
                )

            if resp.status_code in {404, 405} and validation.ok:
                return RouteHealthSnapshot(
                    route_key=route_key,
                    status=HealthState.DEGRADED,
                    last_checked_at=datetime.utcnow(),
                    failure_kind=ValidationKind.ENDPOINT,
                    failure_message="Health endpoint unavailable; route may still work",
                    validated_model_catalog=validation.catalog_checked,
                    preferred_api_dialect=validation.preferred_api_dialect,
                    host_reachable=True,
                    runtime_reachable=True,
                )

            return RouteHealthSnapshot(
                route_key=route_key,
                status=HealthState.DEGRADED,
                last_checked_at=datetime.utcnow(),
                failure_kind=ValidationKind.UNKNOWN,
                failure_message=f"Health endpoint returned {resp.status_code}",
                validated_model_catalog=validation.catalog_checked,
                preferred_api_dialect=validation.preferred_api_dialect,
                host_reachable=True,
                runtime_reachable=True,
            )

        except httpx.RequestError as exc:
            return RouteHealthSnapshot(
                route_key=route_key,
                status=HealthState.UNREACHABLE,
                last_checked_at=datetime.utcnow(),
                failure_kind=ValidationKind.TRANSPORT,
                failure_message=str(exc),
                validated_model_catalog=validation.catalog_checked,
                preferred_api_dialect=validation.preferred_api_dialect,
                host_reachable=False,
                runtime_reachable=False,
            )

    def provider_health_summary(self, provider_id: str) -> Dict[str, object]:
        return self.registry.get_aggregate_provider_health(provider_id)
