"""Deterministic route selection policy for Controller v1.4."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .provider_contracts import (
    HealthState,
    RouteHealthSnapshot,
    RouteTarget,
    RoutingRequirements,
    TransferMode,
    TransferPolicy,
    ValidationKind,
    ValidationResult,
)
from .session_manager import SessionState


class RoutingPolicyV140:
    def __init__(
        self,
        context_window_threshold: float = 0.9,
        fallback_max_hops: int = 3,
        route_stickiness_enabled: bool = True,
    ):
        self.context_window_threshold = context_window_threshold
        self.fallback_max_hops = fallback_max_hops
        self.route_stickiness_enabled = route_stickiness_enabled

    def choose(
        self,
        session: SessionState,
        requirements: RoutingRequirements,
        candidates: List[RouteTarget],
        health_by_route: Dict[str, RouteHealthSnapshot],
        validation_by_route: Dict[str, ValidationResult],
    ) -> Tuple[RouteTarget, List[RouteTarget], str, str, TransferPolicy]:
        if not candidates:
            raise ValueError("No candidate routes available")

        # 1) Hard pin has top priority.
        pinned_route = session.pinned_route
        if pinned_route:
            for route in candidates:
                if route.route_key == pinned_route and self._is_route_usable(route, health_by_route, validation_by_route):
                    transfer = self._compute_transfer_policy(session, route, requirements)
                    return route, self._fallback_chain(route, candidates), "pinned_route", "Pinned route selected", transfer

        pinned_provider = session.pinned_provider
        if pinned_provider:
            pinned_candidates = [r for r in candidates if r.provider_id == pinned_provider]
            usable = [r for r in pinned_candidates if self._is_route_usable(r, health_by_route, validation_by_route)]
            if usable:
                primary = self._sort_routes(usable, requirements, session, health_by_route)[0]
                transfer = self._compute_transfer_policy(session, primary, requirements)
                return primary, self._fallback_chain(primary, candidates), "pinned_provider", "Pinned provider selected", transfer

        # 2) Sticky current route when still usable and capable.
        if self.route_stickiness_enabled and session.current_route:
            for route in candidates:
                if route.route_key == session.current_route and self._is_route_usable(route, health_by_route, validation_by_route):
                    if self._fits_context(route, requirements):
                        transfer = TransferPolicy(mode=TransferMode.NO_TRANSFER, truncation_strategy="none")
                        return route, self._fallback_chain(route, candidates), "sticky_current_route", "Current route is healthy", transfer

        # 3) Score usable candidates and choose highest.
        usable_candidates = [r for r in candidates if self._is_route_usable(r, health_by_route, validation_by_route)]
        if not usable_candidates:
            usable_candidates = list(candidates)

        ranked = self._sort_routes(usable_candidates, requirements, session, health_by_route)
        primary = ranked[0]

        reason_code = "capability_health_match"
        reason_str = "Selected best route by capabilities and health"
        if not self._fits_context(primary, requirements):
            reason_code = "context_pressure"
            reason_str = "Selected route despite context pressure"

        transfer = self._compute_transfer_policy(session, primary, requirements)
        return primary, self._fallback_chain(primary, ranked), reason_code, reason_str, transfer

    def _sort_routes(
        self,
        routes: List[RouteTarget],
        requirements: RoutingRequirements,
        session: SessionState,
        health_by_route: Dict[str, RouteHealthSnapshot],
    ) -> List[RouteTarget]:
        def health_weight(route_key: str) -> int:
            health = health_by_route.get(route_key)
            if not health:
                return 10
            if health.status == HealthState.HEALTHY:
                return 30
            if health.status == HealthState.DEGRADED:
                return 20
            if health.status == HealthState.STALE:
                return 15
            if health.status == HealthState.UNKNOWN:
                return 12
            return 0

        def mode_weight(route: RouteTarget) -> int:
            mode = requirements.routing_mode
            caps = route.capabilities_snapshot
            if mode == "quality":
                return 20 if caps.get("quality_tier") == "premium" else 0
            if mode == "latency":
                return 20 if caps.get("latency_tier") == "low" else 0
            if mode == "cost":
                return 20 if caps.get("cost_tier") == "low" else 0
            return 0

        def hint_weight(route: RouteTarget) -> int:
            weight = 0
            if requirements.provider_hint and route.provider_id == requirements.provider_hint:
                weight += 20
            if requirements.model_hint and route.model_id == requirements.model_hint:
                weight += 20
            if requirements.locality_preference and route.kind == requirements.locality_preference:
                weight += 10
            if session.last_successful_route and route.route_key == session.last_successful_route:
                weight += 8
            return weight

        def context_weight(route: RouteTarget) -> int:
            context = int(route.capabilities_snapshot.get("context_window", 0))
            if context <= 0:
                return 0
            usage_ratio = requirements.estimated_prompt_tokens / max(context, 1)
            if usage_ratio < 0.5:
                return 10
            if usage_ratio < self.context_window_threshold:
                return 6
            if usage_ratio <= 1.0:
                return 2
            return -5

        def score(route: RouteTarget) -> Tuple[int, str]:
            total = health_weight(route.route_key) + mode_weight(route) + hint_weight(route) + context_weight(route)
            return (-total, route.route_key)

        return sorted(routes, key=score)

    def _fallback_chain(self, primary: RouteTarget, routes: List[RouteTarget]) -> List[RouteTarget]:
        ordered = [route for route in routes if route.route_key != primary.route_key]
        return ordered[: self.fallback_max_hops]

    def _is_route_usable(
        self,
        route: RouteTarget,
        health_by_route: Dict[str, RouteHealthSnapshot],
        validation_by_route: Dict[str, ValidationResult],
    ) -> bool:
        validation = validation_by_route.get(route.route_key)
        if validation and validation.kind in {ValidationKind.CREDENTIAL, ValidationKind.MODEL}:
            return False
        if validation and validation.kind == ValidationKind.ENDPOINT and not validation.ok:
            return False

        health = health_by_route.get(route.route_key)
        if not health:
            return True

        return health.status not in {
            HealthState.UNREACHABLE,
            HealthState.UNAUTHORIZED,
            HealthState.INVALID_MODEL,
        }

    def _fits_context(self, route: RouteTarget, requirements: RoutingRequirements) -> bool:
        return int(route.capabilities_snapshot.get("context_window", 0)) >= requirements.estimated_prompt_tokens

    def _compute_transfer_policy(
        self,
        session: SessionState,
        target_route: RouteTarget,
        requirements: RoutingRequirements,
    ) -> TransferPolicy:
        if session.current_route is None:
            return TransferPolicy(mode=TransferMode.REPLAY_FULL, truncation_strategy="none")

        if session.current_route == target_route.route_key:
            return TransferPolicy(mode=TransferMode.NO_TRANSFER, truncation_strategy="none")

        context_window = int(target_route.capabilities_snapshot.get("context_window", 0))
        if requirements.estimated_prompt_tokens <= context_window:
            return TransferPolicy(mode=TransferMode.REPLAY_FULL, truncation_strategy="none")

        return TransferPolicy(
            mode=TransferMode.REPLAY_TRUNCATED,
            truncation_strategy="middle",
            truncation_applied=True,
        )
