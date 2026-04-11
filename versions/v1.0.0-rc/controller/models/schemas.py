"""
Pydantic models for Controller API.
OpenAI-compatible schemas with Controller extensions.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


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


class Message(BaseModel):
    """Chat message."""
    role: Role
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = "controller"
    messages: List[Message]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None
    # Controller-specific extensions
    require_explanation: Optional[bool] = False
    force_provider: Optional[str] = None
    min_confidence: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        return v


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
    # Controller-specific extensions
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
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExplainabilityReport(BaseModel):
    """Detailed explanation of Controller's decision."""
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
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
    last_check: datetime = Field(default_factory=datetime.utcnow)
    consecutive_failures: int = 0


class ControllerStatus(BaseModel):
    """Controller service status."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime_seconds: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
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
    timestamp: datetime = Field(default_factory=datetime.utcnow)
