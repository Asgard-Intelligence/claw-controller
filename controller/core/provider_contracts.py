"""Typed contracts for Controller v1.4 route-oriented orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ProviderKind(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    MANAGED_LOCAL = "managed_local"


class ApiDialect(str, Enum):
    OPENAI_RESPONSES = "openai_responses"
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    OLLAMA_CHAT = "ollama_chat"
    CUSTOM_HTTP = "custom_http"


class HealthState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    UNAUTHORIZED = "unauthorized"
    INVALID_MODEL = "invalid_model"
    UNKNOWN = "unknown"
    STALE = "stale"


class ValidationKind(str, Enum):
    OK = "ok"
    TRANSPORT = "transport"
    CREDENTIAL = "credential"
    MODEL = "model"
    ENDPOINT = "endpoint"
    UNKNOWN = "unknown"


class FallbackKind(str, Enum):
    RETRY_SAME_ROUTE = "retry_same_route"
    FALLBACK_ROUTE = "fallback_route"
    NO_FALLBACK = "no_fallback"


class TransferMode(str, Enum):
    NO_TRANSFER = "no_transfer"
    REPLAY_FULL = "replay_full"
    REPLAY_TRUNCATED = "replay_truncated"
    SUMMARY_HANDOFF = "summary_handoff"


@dataclass
class ProviderDescriptor:
    provider_id: str
    label: str
    kind: ProviderKind
    dialect_family: str
    base_url: str
    validation_base_url: str
    health_url: str
    credential_env: Optional[str]
    default_model_id: str
    enabled: bool = True
    timeout_profile: str = "default"
    supports_model_catalog: bool = True
    supports_runtime_probe: bool = True
    supports_container_reachability_probe: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass
class ModelDescriptor:
    provider_id: str
    model_id: str
    aliases: List[str]
    family: str
    context_window: int
    max_output_tokens: int
    supports_tools: bool
    supports_vision: bool
    supports_json_mode: bool
    supports_streaming: bool
    supports_system_prompt: bool
    system_prompt_mode: str
    preferred_api_dialects: List[ApiDialect]
    allowed_api_dialects: List[ApiDialect]
    quality_tier: str = "standard"
    latency_tier: str = "standard"
    cost_tier: str = "standard"
    identity_stability: str = "standard"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["preferred_api_dialects"] = [x.value for x in self.preferred_api_dialects]
        data["allowed_api_dialects"] = [x.value for x in self.allowed_api_dialects]
        return data


@dataclass
class RouteTarget:
    route_key: str
    provider_id: str
    model_id: str
    api_dialect: ApiDialect
    base_url: str
    credential_env: Optional[str]
    kind: ProviderKind
    timeout_profile: str
    capabilities_snapshot: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["api_dialect"] = self.api_dialect.value
        data["kind"] = self.kind.value
        return data


@dataclass
class RouteHealthSnapshot:
    route_key: str
    status: HealthState
    last_checked_at: datetime
    last_success_at: Optional[datetime] = None
    failure_kind: Optional[ValidationKind] = None
    failure_message: Optional[str] = None
    host_reachable: Optional[bool] = None
    runtime_reachable: Optional[bool] = None
    validated_model_catalog: bool = False
    preferred_api_dialect: Optional[ApiDialect] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "route_key": self.route_key,
            "status": self.status.value,
            "last_checked_at": self.last_checked_at.isoformat(),
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "failure_kind": self.failure_kind.value if self.failure_kind else None,
            "failure_message": self.failure_message,
            "host_reachable": self.host_reachable,
            "runtime_reachable": self.runtime_reachable,
            "validated_model_catalog": self.validated_model_catalog,
            "preferred_api_dialect": self.preferred_api_dialect.value if self.preferred_api_dialect else None,
        }
        return payload


@dataclass
class ValidationResult:
    ok: bool
    validated: bool
    kind: ValidationKind
    message: str
    http_status: Optional[int] = None
    transport_code: Optional[str] = None
    preferred_api_dialect: Optional[ApiDialect] = None
    catalog_checked: bool = False


@dataclass
class RoutingRequirements:
    estimated_prompt_tokens: int
    requires_tools: bool = False
    requires_vision: bool = False
    requires_json_mode: bool = False
    requires_streaming: bool = False
    provider_hint: Optional[str] = None
    model_hint: Optional[str] = None
    routing_mode: str = "balanced"
    identity_sensitive: bool = False
    locality_preference: Optional[ProviderKind] = None


@dataclass
class TransferPolicy:
    mode: TransferMode
    preserve_identity: bool = True
    preserve_system_prompt: bool = True
    preserve_tool_history: bool = True
    truncation_strategy: str = "middle"
    truncation_applied: bool = False


@dataclass
class CanonicalMessage:
    role: str
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalRequest:
    request_id: str
    session_id: str
    agent_id: str
    model: str
    messages: List[CanonicalMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


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
    alternatives: List[str]
    fallback_chain: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ExecutionResult:
    content: str
    provider_id: str
    model_id: str
    api_dialect: ApiDialect
    usage: ExecutionUsage
    finish_reason: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0


@dataclass
class AdapterError(Exception):
    message: str
    kind: ValidationKind
    http_status: Optional[int] = None
    retryable: bool = False
    response_payload: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return self.message
