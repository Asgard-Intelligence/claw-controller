"""
API routes for Controller.
OpenAI-compatible endpoints with Controller extensions.
"""

import time
import uuid
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..models.schemas import (
    ChatCompletionRequest, ChatCompletionResponse, ModelsResponse, ModelInfo,
    HealthResponse, ControllerStatus, ExplainabilityReport, RouteDecision,
    ConfidenceScore, SafetyScore, IntentType, RiskLevel, ProviderHealth
)
from ..core.config import Settings, get_settings
from ..core.intelligence import (
    ConfidenceCalculator, SafetyChecker, IntentClassifier,
    RouteSelector, FallbackManager, RateLimiter
)
from ..core.providers import OpenAIProvider, AnthropicProvider, FallbackProvider

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)

# Global state (initialized in startup)
providers: dict = {}
fallback_provider: Optional[FallbackProvider] = None
confidence_calc: Optional[ConfidenceCalculator] = None
safety_checker: Optional[SafetyChecker] = None
intent_classifier: Optional[IntentClassifier] = None
route_selector: Optional[RouteSelector] = None
rate_limiter: Optional[RateLimiter] = None

# Request tracking
request_stats = {
    "total": 0,
    "successful": 0,
    "failed": 0,
    "start_time": time.time(),
}


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_settings)
) -> bool:
    """Verify API key."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    token = credentials.credentials
    if token != settings.CONTROLLER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True


def init_intelligence(settings: Settings):
    """Initialize intelligence components."""
    global confidence_calc, safety_checker, intent_classifier, route_selector, rate_limiter
    
    confidence_calc = ConfidenceCalculator()
    safety_checker = SafetyChecker(
        enable_pii_detection=settings.ENABLE_PII_DETECTION,
        enable_jailbreak_detection=settings.ENABLE_JAILBREAK_DETECTION,
        risk_threshold=settings.SAFETY_RISK_THRESHOLD
    )
    intent_classifier = IntentClassifier()
    route_selector = RouteSelector()
    rate_limiter = RateLimiter(
        requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE
    ) if settings.RATE_LIMIT_ENABLED else None


def init_providers(settings: Settings):
    """Initialize providers."""
    global providers, fallback_provider
    
    providers = {}
    
    # Initialize OpenAI if key available
    if settings.OPENAI_API_KEY:
        try:
            providers["openai"] = OpenAIProvider(settings.get_provider_config("openai"))
            logger.info("OpenAI provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider: {e}")
    
    # Initialize Anthropic if key available
    if settings.ANTHROPIC_API_KEY:
        try:
            providers["anthropic"] = AnthropicProvider(settings.get_provider_config("anthropic"))
            logger.info("Anthropic provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic provider: {e}")
    
    if not providers:
        logger.warning("No providers initialized - running in mock mode")
    
    fallback_provider = FallbackProvider(providers)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0-rc",
        timestamp=datetime.utcnow()
    )


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(
    authorized: bool = Depends(verify_api_key),
    settings: Settings = Depends(get_settings)
):
    """List available models."""
    models = []
    
    # Controller model
    models.append(ModelInfo(
        id="controller",
        created=int(time.time()),
        owned_by="openclaw"
    ))
    
    # OpenAI models
    if settings.OPENAI_API_KEY:
        models.extend([
            ModelInfo(id="gpt-4o", created=1715367049, owned_by="openai"),
            ModelInfo(id="gpt-4o-mini", created=1721172741, owned_by="openai"),
        ])
    
    # Anthropic models
    if settings.ANTHROPIC_API_KEY:
        models.extend([
            ModelInfo(id="claude-3-opus-20240229", created=1710288000, owned_by="anthropic"),
            ModelInfo(id="claude-3-sonnet-20240229", created=1710288000, owned_by="anthropic"),
            ModelInfo(id="claude-3-haiku-20240307", created=1710288000, owned_by="anthropic"),
        ])
    
    return ModelsResponse(data=models)


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    authorized: bool = Depends(verify_api_key),
    settings: Settings = Depends(get_settings)
):
    """OpenAI-compatible chat completions endpoint."""
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Rate limiting
    if rate_limiter:
        client_ip = http_request.client.host if http_request.client else "unknown"
        if not rate_limiter.is_allowed(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    request_stats["total"] += 1
    
    try:
        # If no providers available, return mock response
        if not providers:
            return _create_mock_response(request, request_id)
        
        # Intelligence mode: analyze and route
        if request.model == "controller" or request.model.startswith("controller-"):
            # Classify intent
            intent = intent_classifier.classify(request)
            
            # Check safety
            safety = safety_checker.check(request)
            
            # Block critical safety risks
            if safety.overall_risk == RiskLevel.CRITICAL:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Request blocked due to safety concerns",
                        "safety_score": safety.model_dump(),
                        "recommendations": safety.recommendations
                    }
                )
            
            # Get provider health
            provider_health = {}
            for name, provider in providers.items():
                health = await provider.health_check()
                provider_health[name] = ProviderHealth(
                    provider=name,
                    status=health.get("status", "unknown"),
                    latency_ms=health.get("latency_ms"),
                    error_rate=0.0
                )
            
            # Calculate confidence
            confidence = confidence_calc.calculate(
                request=request,
                provider_health=provider_health,
                safety_score=safety,
                intent=intent
            )
            
            # Select route
            route = route_selector.select(
                request=request,
                confidence=confidence,
                safety=safety,
                intent=intent,
                provider_health=provider_health,
                available_providers=list(providers.keys())
            )
            
            logger.info(f"[{request_id}] Routed to {route.selected_provider}/{route.selected_model} "
                       f"(confidence: {confidence.overall:.2f}, intent: {intent.value})")
            
            # Execute via fallback provider
            if request.stream:
                return StreamingResponse(
                    _stream_completion(request, route, request_id),
                    media_type="text/event-stream"
                )
            else:
                response, used_provider = await fallback_provider.chat_completion(
                    request, preferred_provider=route.selected_provider
                )
                
                # Add controller metadata
                response.controller_metadata = {
                    "request_id": request_id,
                    "routed_provider": route.selected_provider,
                    "actual_provider": used_provider,
                    "confidence": confidence.overall,
                    "intent": intent.value,
                    "safety_risk": safety.overall_risk.value,
                    "latency_ms": int((time.time() - start_time) * 1000),
                }
                
                # Record success for confidence calculation
                confidence_calc.record_result(request, success=True)
                request_stats["successful"] += 1
                
                return response
        
        else:
            # Direct provider routing (non-controller model)
            provider_name = request.model.split("/")[0] if "/" in request.model else settings.DEFAULT_PROVIDER
            
            if provider_name not in providers:
                raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")
            
            provider = providers[provider_name]
            
            if request.stream:
                return StreamingResponse(
                    _stream_direct(request, provider, request_id),
                    media_type="text/event-stream"
                )
            else:
                response = await provider.chat_completion(request)
                request_stats["successful"] += 1
                return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Request failed: {e}")
        request_stats["failed"] += 1
        if confidence_calc:
            confidence_calc.record_result(request, success=False)
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_completion(request: ChatCompletionRequest, route: RouteDecision, request_id: str):
    """Stream completion with controller metadata."""
    try:
        async for chunk in fallback_provider.chat_completion_stream(
            request, preferred_provider=route.selected_provider
        ):
            yield chunk
    except Exception as e:
        logger.error(f"[{request_id}] Streaming error: {e}")
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"


async def _stream_direct(request: ChatCompletionRequest, provider, request_id: str):
    """Stream completion directly from provider."""
    try:
        async for chunk in provider.chat_completion_stream(request):
            yield chunk
    except Exception as e:
        logger.error(f"[{request_id}] Streaming error: {e}")
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"


def _create_mock_response(request: ChatCompletionRequest, request_id: str) -> ChatCompletionResponse:
    """Create mock response when no providers available."""
    from ..models.schemas import Message, Choice, Usage, Role
    
    last_message = request.messages[-1].content if request.messages else ""
    
    return ChatCompletionResponse(
        id=f"mock-{request_id}",
        created=int(time.time()),
        model="controller-mock",
        choices=[Choice(
            index=0,
            message=Message(
                role=Role.ASSISTANT,
                content=("[MOCK MODE] Controller is running but no LLM providers are configured. "
                        f"Your message: '{last_message[:100]}{'...' if len(last_message) > 100 else ''}'")
            ),
            finish_reason="stop"
        )],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        controller_metadata={
            "request_id": request_id,
            "mode": "mock",
            "message": "Configure OPENAI_API_KEY or ANTHROPIC_API_KEY for real responses"
        }
    )


@router.get("/v1/controller/status", response_model=ControllerStatus)
async def controller_status(
    authorized: bool = Depends(verify_api_key),
    settings: Settings = Depends(get_settings)
):
    """Get detailed controller status."""
    
    # Get provider health
    provider_health = []
    for name, provider in providers.items():
        health = await provider.health_check()
        provider_health.append(ProviderHealth(
            provider=name,
            status=health.get("status", "unknown"),
            latency_ms=health.get("latency_ms"),
            error_rate=0.0
        ))
    
    # Calculate uptime
    uptime = int(time.time() - request_stats["start_time"])
    
    # Calculate average latency (simplified)
    avg_latency = 0.0
    if provider_health:
        latencies = [p.latency_ms for p in provider_health if p.latency_ms]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
    
    # Determine overall status
    healthy_count = sum(1 for p in provider_health if p.status == "healthy")
    if healthy_count == len(provider_health) and provider_health:
        status = "healthy"
    elif healthy_count > 0:
        status = "degraded"
    else:
        status = "unhealthy" if provider_health else "healthy"
    
    return ControllerStatus(
        status=status,
        version="1.0.0-rc",
        uptime_seconds=uptime,
        providers=provider_health,
        total_requests=request_stats["total"],
        successful_requests=request_stats["successful"],
        failed_requests=request_stats["failed"],
        average_latency_ms=avg_latency,
        active_connections=0,  # Would track in production
        config={
            "default_provider": settings.DEFAULT_PROVIDER,
            "fallback_providers": settings.FALLBACK_PROVIDERS,
            "rate_limit_enabled": settings.RATE_LIMIT_ENABLED,
            "pii_detection": settings.ENABLE_PII_DETECTION,
            "jailbreak_detection": settings.ENABLE_JAILBREAK_DETECTION,
        }
    )


@router.post("/v1/controller/explain", response_model=ExplainabilityReport)
async def explain_request(
    request: ChatCompletionRequest,
    authorized: bool = Depends(verify_api_key),
    settings: Settings = Depends(get_settings)
):
    """Explain how Controller would route a request without executing it."""
    
    request_id = str(uuid.uuid4())
    
    # Classify intent
    intent = intent_classifier.classify(request)
    
    # Check safety
    safety = safety_checker.check(request)
    
    # Get provider health
    provider_health = {}
    for name, provider in providers.items():
        health = await provider.health_check()
        provider_health[name] = ProviderHealth(
            provider=name,
            status=health.get("status", "unknown"),
            latency_ms=health.get("latency_ms"),
            error_rate=0.0
        )
    
    # Calculate confidence
    confidence = confidence_calc.calculate(
        request=request,
        provider_health=provider_health,
        safety_score=safety,
        intent=intent
    )
    
    # Select route
    route = route_selector.select(
        request=request,
        confidence=confidence,
        safety=safety,
        intent=intent,
        provider_health=provider_health,
        available_providers=list(providers.keys())
    )
    
    # Build alternatives
    alternatives = []
    for provider_name in providers.keys():
        if provider_name != route.selected_provider:
            alternatives.append({
                "provider": provider_name,
                "reason": f"Lower priority in {route.selected_tier} tier"
            })
    
    return ExplainabilityReport(
        request_id=request_id,
        original_request=request.model_dump(),
        route_decision=route,
        confidence_breakdown={
            "overall": confidence.overall,
            "factors": [f.model_dump() for f in confidence.factors],
            "reasoning": confidence.reasoning
        },
        safety_analysis={
            "risk_level": safety.overall_risk.value,
            "risk_score": safety.risk_score,
            "pii_count": len(safety.pii_detected),
            "jailbreak_attempts": len(safety.jailbreak_attempts),
            "recommendations": safety.recommendations
        },
        intent_classification={
            "detected_intent": intent.value,
            "keywords_matched": _get_intent_keywords(intent)
        },
        provider_selection_logic=route.reasoning,
        alternatives_considered=alternatives,
        user_recommendations=_generate_recommendations(route, safety, confidence)
    )


def _get_intent_keywords(intent: IntentType) -> list:
    """Get keywords that matched for intent classification."""
    from ..core.intelligence import INTENT_KEYWORDS
    return INTENT_KEYWORDS.get(intent, [])


def _generate_recommendations(route: RouteDecision, safety: SafetyScore, confidence: ConfidenceScore) -> list:
    """Generate user recommendations."""
    recs = []
    
    if safety.overall_risk != RiskLevel.LOW:
        recs.extend(safety.recommendations)
    
    if confidence.overall < 0.5:
        recs.append("Consider simplifying your request for better results")
    
    if route.selected_tier == "premium":
        recs.append("This request is routed to premium tier - higher cost expected")
    
    if not recs:
        recs.append("Request looks good - no specific recommendations")
    
    return recs
