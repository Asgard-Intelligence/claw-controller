"""Safe router compatibility facade over v1.4 route-oriented orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .hybrid_router import HybridRouter
from .provider_contracts import (
    ApiDialect,
    ModelDescriptor,
    ProviderDescriptor,
    ProviderKind,
    TransferMode,
    TransferPolicy,
)
from .provider_registry_v140 import ProviderRegistryV140
from .routing_policy_v140 import RoutingPolicyV140


class RoutingReason(Enum):
    PINNED = "user_pinned_provider"
    HEALTHY = "current_route_healthy"
    FALLBACK = "fallback_from_failed_route"
    USER_REQUESTED = "user_requested_switch"
    CONTEXT_WINDOW = "context_window_exceeded"
    FIRST_MESSAGE = "first_message_no_session"
    CAPABILITY_MISMATCH = "capability_mismatch"


@dataclass
class RoutingDecision:
    provider_id: str
    model_id: str
    api_dialect: ApiDialect
    route_key: str
    reason_code: str
    reason_str: str
    context_transfer_required: bool
    transfer_policy: TransferPolicy
    estimated_tokens: int
    confidence: float
    alternatives: List[str]
    metadata: Dict[str, Any]

    @property
    def provider(self) -> str:  # v1.3 compatibility
        return self.provider_id

    @property
    def model(self) -> Optional[str]:  # v1.3 compatibility
        return self.model_id


@dataclass
class ModelCapabilities:
    provider: str
    model: str
    context_window: int
    supports_system_prompt: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    supports_json_mode: bool = True
    max_output_tokens: int = 4096


class ProviderRegistry:
    """Compatibility wrapper preserving v1.3 register API."""

    def __init__(self):
        self._registry = ProviderRegistryV140()
        self._provider_health: Dict[str, str] = {}

    def register(self, capabilities: ModelCapabilities) -> None:
        provider = self._registry.get_provider(capabilities.provider)
        if not provider:
            provider = ProviderDescriptor(
                provider_id=capabilities.provider,
                label=capabilities.provider,
                kind=ProviderKind.CLOUD,
                dialect_family="openai_compatible",
                base_url="",
                validation_base_url="",
                health_url="",
                credential_env=None,
                default_model_id=capabilities.model,
                enabled=True,
                timeout_profile="default",
            )
            self._registry.register_provider(provider)

        model = ModelDescriptor(
            provider_id=capabilities.provider,
            model_id=capabilities.model,
            aliases=[],
            family="legacy",
            context_window=capabilities.context_window,
            max_output_tokens=capabilities.max_output_tokens,
            supports_tools=capabilities.supports_tools,
            supports_vision=capabilities.supports_vision,
            supports_json_mode=capabilities.supports_json_mode,
            supports_streaming=True,
            supports_system_prompt=capabilities.supports_system_prompt,
            system_prompt_mode="native",
            preferred_api_dialects=[ApiDialect.OPENAI_CHAT_COMPLETIONS],
            allowed_api_dialects=[ApiDialect.OPENAI_CHAT_COMPLETIONS],
        )
        self._registry.register_model(model)

    def get_capabilities(self, provider: str) -> Optional[ModelCapabilities]:
        routes = self._registry.get_routes_for_provider(provider)
        if not routes:
            return None
        route = routes[0]
        caps = route.capabilities_snapshot
        return ModelCapabilities(
            provider=provider,
            model=route.model_id,
            context_window=int(caps.get("context_window", 0)),
            supports_system_prompt=bool(caps.get("supports_system_prompt", True)),
            supports_tools=bool(caps.get("supports_tools", False)),
            supports_vision=bool(caps.get("supports_vision", False)),
            supports_json_mode=bool(caps.get("supports_json_mode", False)),
            max_output_tokens=int(caps.get("max_output_tokens", 0)),
        )

    def set_health(self, provider: str, status: str) -> None:
        self._provider_health[provider] = status

    def get_health(self, provider: str) -> str:
        return self._provider_health.get(provider, "unknown")

    def is_healthy(self, provider: str) -> bool:
        return self._provider_health.get(provider) == "healthy"

    def find_providers_with_min_context(self, min_context: int) -> List[str]:
        providers: List[str] = []
        for provider_id in self._registry._provider_routes.keys():
            caps = self.get_capabilities(provider_id)
            if caps and caps.context_window >= min_context:
                providers.append(provider_id)
        return providers


class RoutingPolicy:
    CONTEXT_WINDOW_THRESHOLD = 0.9

    def __init__(self, registry: ProviderRegistry):
        self.registry = registry


class SafeRoutingEngine:
    """Compatibility facade. New code should use HybridRouter directly."""

    def __init__(
        self,
        registry: ProviderRegistry,
        policy: RoutingPolicy,
        default_provider: str = "openai",
        hybrid_router: Optional[HybridRouter] = None,
    ):
        self.registry = registry
        self.policy = policy
        self.default_provider = default_provider
        self.hybrid_router = hybrid_router

    async def select_provider(self, session, request) -> RoutingDecision:
        # If full v1.4 router is present, delegate through it with a minimal requirements shim.
        if self.hybrid_router:
            raw_request = request if isinstance(request, dict) else getattr(request, "__dict__", {})
            requirements = SimpleNamespace(
                estimated_prompt_tokens=getattr(session, "total_tokens", 0),
                requires_tools=bool(raw_request.get("tools")),
                requires_vision=False,
                requires_json_mode=False,
                requires_streaming=bool(raw_request.get("stream", False)),
                provider_hint=getattr(session, "pinned_provider", None),
                model_hint=None,
                routing_mode="balanced",
                identity_sensitive=False,
                locality_preference=None,
            )
            decision = await self.hybrid_router.select_route(session, requirements)
            return RoutingDecision(
                provider_id=decision.provider_id,
                model_id=decision.model_id,
                api_dialect=decision.api_dialect,
                route_key=decision.route_key,
                reason_code=decision.reason_code,
                reason_str=decision.reason_str,
                context_transfer_required=decision.context_transfer_required,
                transfer_policy=decision.transfer_policy,
                estimated_tokens=decision.estimated_tokens,
                confidence=1.0,
                alternatives=decision.alternatives,
                metadata=decision.metadata,
            )

        # Minimal fallback behavior.
        provider = session.pinned_provider or session.current_provider or self.default_provider
        return RoutingDecision(
            provider_id=provider,
            model_id="unknown",
            api_dialect=ApiDialect.OPENAI_CHAT_COMPLETIONS,
            route_key=f"{provider}:unknown:{ApiDialect.OPENAI_CHAT_COMPLETIONS.value}",
            reason_code=RoutingReason.HEALTHY.value,
            reason_str="Compatibility fallback selection",
            context_transfer_required=False,
            transfer_policy=TransferPolicy(mode=TransferMode.NO_TRANSFER),
            estimated_tokens=getattr(session, "total_tokens", 0),
            confidence=0.5,
            alternatives=[],
            metadata={"compat_mode": True},
        )

    async def execute_fallback(self, session, failed_provider: str, error: Exception):
        return None


class ContextPreservationEngine:
    def __init__(self, registry: ProviderRegistry):
        self.registry = registry

    async def prepare_context_for_routing(self, session, target_provider: str) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = []
        if session.identity:
            messages.append(
                {
                    "role": "system",
                    "content": session.identity.system_prompt,
                    "metadata": {
                        "identity_hash": session.identity.hash,
                        "identity_version": session.identity.version,
                    },
                }
            )

        for msg in session.messages:
            messages.append({"role": msg.role, "content": msg.content, "metadata": dict(msg.metadata)})

        return {
            "messages": messages,
            "original_count": len(messages),
            "final_count": len(messages),
            "truncation_applied": False,
            "identity_preserved": bool(messages and messages[0].get("role") == "system"),
        }
