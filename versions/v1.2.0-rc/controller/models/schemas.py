"""
Pydantic models for the Controller API.
OpenAI-compatible transport schemas with an explicit normalization layer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class IntentType(str, Enum):
    """Classification of user intent."""

    CODE = "code"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Safety risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Role(str, Enum):
    """Message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TextContentPart(BaseModel):
    """Supported OpenAI content part for text-only normalization."""

    model_config = ConfigDict(extra="allow")

    type: Literal["text"]
    text: str


MessageContent = Union[str, List[TextContentPart], None]


class Message(BaseModel):
    """Transport-level chat message."""

    model_config = ConfigDict(extra="allow")

    role: Role
    content: MessageContent = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    refusal: Optional[str] = None

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value: Any) -> MessageContent:
        if value is None:
            return None

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            normalized_parts: List[Dict[str, Any]] = []
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"messages[].content[{index}] must be an object with type='text'"
                    )
                part_type = item.get("type")
                if part_type != "text":
                    raise ValueError(
                        f"unsupported content part type '{part_type}' at index {index}; only 'text' parts are supported"
                    )
                text_value = item.get("text")
                if not isinstance(text_value, str):
                    raise ValueError(
                        f"messages[].content[{index}].text must be a string for type='text'"
                    )
                normalized_parts.append(item)
            return normalized_parts

        raise ValueError(
            "messages[].content must be a string, null, or an array of text content parts"
        )

    def normalized_content(self) -> str:
        """Normalize supported content variants to plain text."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content)

    def structured_part_types(self) -> List[str]:
        """Return content part types observed on the transport boundary."""
        if not isinstance(self.content, list):
            return []
        return [part.type for part in self.content]


class NormalizedMessage(BaseModel):
    """Internal message representation consumed by controller logic."""

    model_config = ConfigDict(extra="allow")

    role: Role
    content: str = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    refusal: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request transport model."""

    model_config = ConfigDict(extra="allow")

    model: str = "controller"
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None
    seed: Optional[int] = None

    # OpenAI-compatible extended fields observed in real runtime traffic.
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    response_format: Optional[Dict[str, Any]] = None
    stream_options: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    # Controller-specific extensions
    require_explanation: Optional[bool] = False
    force_provider: Optional[str] = None
    min_confidence: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: List[Message]) -> List[Message]:
        if not value:
            raise ValueError("messages cannot be empty")
        return value

    @field_validator("force_provider")
    @classmethod
    def normalize_force_provider(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        return normalized or None

    def normalize(self) -> "NormalizedChatCompletionRequest":
        """Normalize transport-specific message shapes into controller internals."""
        extra_body = dict(self.model_extra or {})
        normalized_messages = [
            NormalizedMessage(
                role=message.role,
                content=message.normalized_content(),
                name=message.name,
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
                refusal=message.refusal,
            )
            for message in self.messages
        ]

        return NormalizedChatCompletionRequest(
            model=self.model,
            messages=normalized_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            n=self.n,
            stream=self.stream,
            stop=self.stop,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            user=self.user,
            seed=self.seed,
            tools=self.tools,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            response_format=self.response_format,
            stream_options=self.stream_options,
            metadata=self.metadata,
            require_explanation=self.require_explanation,
            force_provider=self.force_provider,
            min_confidence=self.min_confidence,
            extra_body=extra_body,
        )

    def structured_content_message_count(self) -> int:
        """Count messages that arrived as structured text-part arrays."""
        return sum(1 for message in self.messages if isinstance(message.content, list))


class NormalizedChatCompletionRequest(BaseModel):
    """Internal controller request representation with normalized message text."""

    model_config = ConfigDict(extra="ignore")

    model: str = "controller"
    messages: List[NormalizedMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None
    seed: Optional[int] = None

    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    response_format: Optional[Dict[str, Any]] = None
    stream_options: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    require_explanation: Optional[bool] = False
    force_provider: Optional[str] = None
    min_confidence: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)

    extra_body: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_messages(self) -> "NormalizedChatCompletionRequest":
        if not self.messages:
            raise ValueError("messages cannot be empty")
        return self

    def has_tooling(self) -> bool:
        """Return whether this request carries tool definitions or tool transcript state."""
        return bool(
            self.tools
            or self.tool_choice is not None
            or self.parallel_tool_calls is not None
            or any(message.tool_calls for message in self.messages)
            or any(message.role == Role.TOOL for message in self.messages)
        )

    def requires_openai_passthrough(self) -> bool:
        """Return whether this request uses OpenAI-specific passthrough metadata."""
        return bool(
            self.has_tooling()
            or self.response_format is not None
            or self.stream_options is not None
            or self.extra_body
        )


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """Completion choice."""

    index: int
    message: Message
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None
    controller_metadata: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Dict[str, Any]] = Field(default_factory=list)
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    """List models response."""

    object: str = "list"
    data: List[ModelInfo]


class ConfidenceFactor(BaseModel):
    """Individual confidence factor."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    description: str


class ConfidenceScore(BaseModel):
    """Multi-factor confidence score."""

    overall: float = Field(ge=0.0, le=1.0)
    factors: List[ConfidenceFactor]
    tier_recommendation: str
    reasoning: str


class PIIFinding(BaseModel):
    """PII detection finding."""

    type: str
    position: int
    length: int
    confidence: float = Field(ge=0.0, le=1.0)
    masked: Optional[str] = None


class SafetyCheck(BaseModel):
    """Individual safety check result."""

    check_name: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    details: Optional[str] = None


class SafetyScore(BaseModel):
    """Comprehensive safety score."""

    overall_risk: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)
    pii_detected: List[PIIFinding]
    jailbreak_attempts: List[str]
    content_violations: List[str]
    checks: List[SafetyCheck]
    masked_content: Optional[str] = None
    recommendations: List[str]


class RouteDecision(BaseModel):
    """Routing decision with reasoning."""

    selected_provider: str
    selected_model: str
    selected_tier: str
    confidence: ConfidenceScore
    safety: SafetyScore
    intent: IntentType
    reasoning: str
    estimated_cost: Optional[float] = None
    estimated_latency_ms: Optional[int] = None
    fallback_chain: List[str]
    timestamp: datetime = Field(default_factory=utc_now)


class ExplainabilityReport(BaseModel):
    """Detailed explanation of Controller routing."""

    request_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    original_request: Dict[str, Any]
    route_decision: RouteDecision
    confidence_breakdown: Dict[str, Any]
    safety_analysis: Dict[str, Any]
    intent_classification: Dict[str, Any]
    provider_selection_logic: str
    alternatives_considered: List[Dict[str, Any]]
    user_recommendations: List[str]


class ProviderHealth(BaseModel):
    """Provider health status."""

    provider: str
    status: Literal["healthy", "degraded", "unhealthy", "unknown"]
    latency_ms: Optional[int] = None
    error_rate: float = Field(ge=0.0, le=1.0)
    last_check: datetime = Field(default_factory=utc_now)
    consecutive_failures: int = 0


class ControllerStatus(BaseModel):
    """Controller service status."""

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime_seconds: int
    timestamp: datetime = Field(default_factory=utc_now)
    providers: List[ProviderHealth]
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float
    active_connections: int
    config: Dict[str, Any]


class HealthResponse(BaseModel):
    """Simple health check response."""

    status: str
    version: str
    timestamp: datetime = Field(default_factory=utc_now)
