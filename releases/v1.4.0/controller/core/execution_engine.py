"""Execution engine for route-chain execution with typed fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .provider_contracts import (
    AdapterError,
    CanonicalMessage,
    CanonicalRequest,
    ExecutionResult,
    FallbackKind,
    RoutingDecision,
    ValidationKind,
)
from .provider_registry_v140 import ProviderRegistryV140
from .provider_adapters import AnthropicMessagesAdapter, OllamaNativeAdapter, OpenAICompatibleAdapter


@dataclass
class RouteAttempt:
    route_key: str
    success: bool
    error_kind: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class EngineExecutionReport:
    used_route_key: str
    attempts: List[RouteAttempt] = field(default_factory=list)


class ExecutionEngineError(Exception):
    def __init__(self, message: str, attempts: List[RouteAttempt]):
        super().__init__(message)
        self.attempts = attempts


class ExecutionEngine:
    def __init__(
        self,
        registry: ProviderRegistryV140,
        timeout_for_profile,
        max_retries_same_route: int = 1,
    ):
        self.registry = registry
        self.timeout_for_profile = timeout_for_profile
        self.max_retries_same_route = max_retries_same_route

        adapters = [
            OpenAICompatibleAdapter(),
            AnthropicMessagesAdapter(),
            OllamaNativeAdapter(),
        ]
        self._adapters = adapters

    async def execute(
        self,
        canonical_request: CanonicalRequest,
        decision: RoutingDecision,
        route_chain: List[str],
    ) -> tuple[ExecutionResult, EngineExecutionReport]:
        attempts: List[RouteAttempt] = []

        for route_key in route_chain:
            route = self.registry.get_route(route_key)
            if not route:
                attempts.append(RouteAttempt(route_key=route_key, success=False, error_kind="unknown", error_message="Route missing"))
                continue

            adapter = self._find_adapter(route.api_dialect)
            if not adapter:
                attempts.append(
                    RouteAttempt(
                        route_key=route_key,
                        success=False,
                        error_kind="unknown",
                        error_message=f"No adapter for dialect {route.api_dialect.value}",
                    )
                )
                continue

            messages = self._prepare_messages_for_route(canonical_request.messages, route)
            timeout_seconds = float(self.timeout_for_profile(route.timeout_profile))

            retries = 0
            while True:
                try:
                    result = await adapter.execute(
                        request=canonical_request,
                        route=route,
                        messages=messages,
                        timeout_seconds=timeout_seconds,
                    )
                    attempts.append(RouteAttempt(route_key=route_key, success=True))
                    return result, EngineExecutionReport(used_route_key=route_key, attempts=attempts)

                except AdapterError as exc:
                    attempts.append(
                        RouteAttempt(
                            route_key=route_key,
                            success=False,
                            error_kind=exc.kind.value,
                            error_message=str(exc),
                        )
                    )

                    action = self._fallback_action(exc, retries)
                    if action == FallbackKind.RETRY_SAME_ROUTE:
                        retries += 1
                        continue
                    if action == FallbackKind.FALLBACK_ROUTE:
                        break
                    raise ExecutionEngineError(str(exc), attempts) from exc

        raise ExecutionEngineError("All routes failed", attempts)

    def _find_adapter(self, dialect):
        for adapter in self._adapters:
            if adapter.can_handle(dialect):
                return adapter
        return None

    def _fallback_action(self, error: AdapterError, retries_done: int) -> FallbackKind:
        if error.kind == ValidationKind.TRANSPORT and error.retryable and retries_done < self.max_retries_same_route:
            return FallbackKind.RETRY_SAME_ROUTE

        if error.kind in {
            ValidationKind.TRANSPORT,
            ValidationKind.CREDENTIAL,
            ValidationKind.MODEL,
            ValidationKind.ENDPOINT,
            ValidationKind.UNKNOWN,
        }:
            return FallbackKind.FALLBACK_ROUTE

        return FallbackKind.NO_FALLBACK

    def _prepare_messages_for_route(
        self,
        messages: List[CanonicalMessage],
        route,
    ) -> List[CanonicalMessage]:
        context_window = int(route.capabilities_snapshot.get("context_window", 0))
        if context_window <= 0:
            return list(messages)

        estimated = sum(max(1, len(m.content) // 4) for m in messages)
        if estimated <= context_window:
            return list(messages)

        # Explicit truncation policy: keep system, oldest context anchor, newest turns.
        system = [m for m in messages if m.role == "system"][:1]
        non_system = [m for m in messages if m.role != "system"]

        keep_first = non_system[:2]
        keep_last = non_system[-10:] if len(non_system) > 10 else non_system

        truncated: List[CanonicalMessage] = []
        truncated.extend(system)
        truncated.extend(keep_first)
        if len(non_system) > len(keep_first) + len(keep_last):
            truncated.append(
                CanonicalMessage(
                    role="system",
                    content="[... conversation truncated due to route context limits ...]",
                    metadata={"truncated": True},
                )
            )
        truncated.extend(keep_last)
        return truncated
