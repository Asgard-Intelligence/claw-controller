"""Hybrid router selecting executable route targets and fallback chains."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from .provider_contracts import RoutingDecision, RoutingRequirements
from .provider_registry_v140 import ProviderRegistryV140
from .routing_policy_v140 import RoutingPolicyV140
from .session_manager import SessionState
from .validation_service import RouteValidationService
from .health_service import RouteHealthService


class HybridRouter:
    def __init__(
        self,
        registry: ProviderRegistryV140,
        validation_service: RouteValidationService,
        health_service: RouteHealthService,
        policy: RoutingPolicyV140,
        default_provider: str,
    ):
        self.registry = registry
        self.validation_service = validation_service
        self.health_service = health_service
        self.policy = policy
        self.default_provider = default_provider

    async def select_route(
        self,
        session: SessionState,
        requirements: RoutingRequirements,
    ) -> RoutingDecision:
        candidates = self.registry.find_candidate_routes(requirements)

        if not candidates:
            default_route = self.registry.get_default_route(self.default_provider)
            if not default_route:
                raise ValueError("No candidate routes and no default route available")
            candidates = [default_route]

        validation_by_route, health_by_route = await self._collect_route_state(candidates)

        primary, fallbacks, reason_code, reason_str, transfer_policy = self.policy.choose(
            session=session,
            requirements=requirements,
            candidates=candidates,
            health_by_route=health_by_route,
            validation_by_route=validation_by_route,
        )

        # Preferred dialect from validation may override selected dialect if available.
        selected_validation = validation_by_route.get(primary.route_key)
        if selected_validation and selected_validation.preferred_api_dialect:
            preferred_dialect = selected_validation.preferred_api_dialect
            if preferred_dialect != primary.api_dialect:
                alt_key = self.registry.make_route_key(primary.provider_id, primary.model_id, preferred_dialect)
                alt_route = self.registry.get_route(alt_key)
                if alt_route:
                    primary = alt_route

        fallback_keys = [route.route_key for route in fallbacks]

        return RoutingDecision(
            provider_id=primary.provider_id,
            model_id=primary.model_id,
            api_dialect=primary.api_dialect,
            route_key=primary.route_key,
            reason_code=reason_code,
            reason_str=reason_str,
            context_transfer_required=session.current_route != primary.route_key,
            transfer_policy=transfer_policy,
            estimated_tokens=requirements.estimated_prompt_tokens,
            alternatives=fallback_keys[:2],
            fallback_chain=fallback_keys,
            metadata={
                "candidate_count": len(candidates),
                "validation": {
                    key: {
                        "ok": val.ok,
                        "kind": val.kind.value,
                        "message": val.message,
                    }
                    for key, val in validation_by_route.items()
                },
                "health": {key: health.status.value for key, health in health_by_route.items()},
            },
        )

    async def _collect_route_state(self, routes) -> tuple[Dict[str, object], Dict[str, object]]:
        validation_tasks = [self.validation_service.get_route_validation(route.route_key) for route in routes]
        health_tasks = [self.health_service.get_route_health(route.route_key) for route in routes]

        validations = await asyncio.gather(*validation_tasks)
        healths = await asyncio.gather(*health_tasks)

        validation_by_route = {route.route_key: validation for route, validation in zip(routes, validations)}
        health_by_route = {route.route_key: health for route, health in zip(routes, healths)}
        return validation_by_route, health_by_route

    def build_route_chain(self, decision: RoutingDecision) -> List[str]:
        chain = [decision.route_key]
        chain.extend([key for key in decision.fallback_chain if key != decision.route_key])
        return chain
