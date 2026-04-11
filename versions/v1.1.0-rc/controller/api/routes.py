"""
API routes for Controller.
OpenAI-compatible boundary layer hardened for real OpenClaw traffic.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .. import __version__
from ..core.config import Settings, get_settings
from ..core.intelligence import (
    ConfidenceCalculator,
    IntentClassifier,
    RateLimiter,
    RouteSelector,
    SafetyChecker,
)
from ..core.providers import AnthropicProvider, FallbackProvider, OpenAIProvider, Provider
from ..models.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ConfidenceScore,
    ControllerStatus,
    ExplainabilityReport,
    HealthResponse,
    IntentType,
    Message,
    ModelInfo,
    ModelsResponse,
    NormalizedChatCompletionRequest,
    ProviderHealth,
    RiskLevel,
    Role,
    RouteDecision,
    SafetyScore,
    Usage,
)

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)

# Global runtime state (initialized during startup)
providers: Dict[str, Provider] = {}
fallback_provider: Optional[FallbackProvider] = None
confidence_calc: Optional[ConfidenceCalculator] = None
safety_checker: Optional[SafetyChecker] = None
intent_classifier: Optional[IntentClassifier] = None
route_selector: Optional[RouteSelector] = None
rate_limiter: Optional[RateLimiter] = None

request_stats = {
    "total": 0,
    "successful": 0,
    "failed": 0,
    "start_time": time.time(),
}


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    settings: Settings = Depends(get_settings),
) -> bool:
    """Verify API key for protected routes."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    if credentials.credentials != settings.CONTROLLER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


def init_intelligence(settings: Settings) -> None:
    """Initialize request analysis and routing components."""
    global confidence_calc, safety_checker, intent_classifier, route_selector, rate_limiter

    confidence_calc = ConfidenceCalculator()
    safety_checker = SafetyChecker(
        enable_pii_detection=settings.ENABLE_PII_DETECTION,
        enable_jailbreak_detection=settings.ENABLE_JAILBREAK_DETECTION,
        risk_threshold=settings.SAFETY_RISK_THRESHOLD,
    )
    intent_classifier = IntentClassifier()
    route_selector = RouteSelector(settings=settings)
    rate_limiter = (
        RateLimiter(requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE)
        if settings.RATE_LIMIT_ENABLED
        else None
    )


def init_providers(settings: Settings) -> None:
    """Initialize configured providers and fallback wrapper."""
    global providers, fallback_provider

    providers = {}

    provider_order: List[str] = []
    for name in settings.FALLBACK_PROVIDERS + settings.get_enabled_provider_names():
        normalized = name.strip().lower()
        if normalized and normalized not in provider_order:
            provider_order.append(normalized)

    if not provider_order:
        provider_order = ["openai", "anthropic"]

    for provider_name in provider_order:
        if provider_name == "openai" and settings.OPENAI_API_KEY:
            try:
                config = settings.get_provider_config("openai")
                providers["openai"] = OpenAIProvider(config)
                logger.info(
                    "Provider active: openai base_url=%s default_model=%s",
                    config.get("base_url"),
                    config.get("default_model"),
                )
            except Exception as exc:
                logger.warning("Failed to initialize provider openai: %s", exc)
        elif provider_name == "anthropic" and settings.ANTHROPIC_API_KEY:
            try:
                config = settings.get_provider_config("anthropic")
                providers["anthropic"] = AnthropicProvider(config)
                logger.info(
                    "Provider active: anthropic base_url=%s default_model=%s",
                    config.get("base_url"),
                    config.get("default_model"),
                )
            except Exception as exc:
                logger.warning("Failed to initialize provider anthropic: %s", exc)

    fallback_provider = FallbackProvider(providers)

    if providers:
        logger.info("Runtime provider state: active=%s", list(providers.keys()))
    else:
        logger.warning(
            "Runtime provider state: no active providers configured; mock responses will be served"
        )


def _request_id_from_http(http_request: Request) -> str:
    request_id = getattr(http_request.state, "request_id", None)
    if request_id:
        return str(request_id)
    return str(uuid.uuid4())


def _client_ip(http_request: Request) -> str:
    if http_request.client and http_request.client.host:
        return http_request.client.host
    forwarded_for = http_request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return "unknown"


def _ensure_runtime_ready() -> None:
    missing: List[str] = []
    if confidence_calc is None:
        missing.append("confidence_calc")
    if safety_checker is None:
        missing.append("safety_checker")
    if intent_classifier is None:
        missing.append("intent_classifier")
    if route_selector is None:
        missing.append("route_selector")

    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Controller runtime not initialized: {', '.join(missing)}",
        )


def _normalized_model_label(request: NormalizedChatCompletionRequest) -> str:
    return request.model or "controller"


def _controller_mode(model: str) -> bool:
    return model == "controller" or model.startswith("controller-")


def _resolve_direct_provider(
    request: NormalizedChatCompletionRequest,
    settings: Settings,
) -> Tuple[str, str]:
    model = _normalized_model_label(request)
    force_provider = (request.force_provider or "").strip().lower()

    if "/" in model:
        provider_name, target_model = model.split("/", 1)
        provider_name = provider_name.strip().lower()
        target_model = target_model.strip()
        if not provider_name:
            raise HTTPException(status_code=400, detail="Provider prefix cannot be empty")
        if not target_model:
            target_model = settings.get_provider_config(provider_name).get("default_model") or ""
        if not target_model:
            raise HTTPException(
                status_code=400,
                detail=f"No target model could be resolved for provider '{provider_name}'",
            )
        return provider_name, target_model

    provider_name = force_provider or settings.DEFAULT_PROVIDER.strip().lower()
    target_model = model
    return provider_name, target_model


def _build_provider_health(provider_name: str, health: Dict[str, Any]) -> ProviderHealth:
    return ProviderHealth(
        provider=provider_name,
        status=health.get("status", "unknown"),
        latency_ms=health.get("latency_ms"),
        error_rate=float(health.get("error_rate", 0.0) or 0.0),
        consecutive_failures=0,
    )


async def _provider_health_snapshot(provider_names: Optional[List[str]] = None) -> Dict[str, ProviderHealth]:
    names = provider_names or list(providers.keys())
    result: Dict[str, ProviderHealth] = {}
    for provider_name in names:
        provider = providers.get(provider_name)
        if not provider:
            continue
        try:
            health = await provider.health_check()
        except Exception as exc:
            logger.warning("Health check failed for provider %s: %s", provider_name, exc)
            health = {
                "status": "unhealthy",
                "provider": provider_name,
                "error": str(exc),
            }
        result[provider_name] = _build_provider_health(provider_name, health)
    return result


def _resolve_compatible_providers(
    request: NormalizedChatCompletionRequest,
    candidate_names: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    names = candidate_names or list(providers.keys())
    compatible: List[str] = []
    incompatibilities: Dict[str, str] = {}

    for provider_name in names:
        provider = providers.get(provider_name)
        if not provider:
            continue
        try:
            provider.validate_request(request)
            compatible.append(provider_name)
        except ValueError as exc:
            incompatibilities[provider_name] = str(exc)

    return compatible, incompatibilities


def _raise_no_compatible_provider(
    request_id: str,
    request: NormalizedChatCompletionRequest,
    incompatibilities: Dict[str, str],
) -> None:
    detail = {
        "error": "No compatible providers available for this request shape",
        "request_id": request_id,
        "requested_model": request.model,
        "provider_diagnostics": incompatibilities,
    }
    raise HTTPException(status_code=400, detail=detail)


def _metadata_base(
    *,
    request_id: str,
    requested_model: str,
    structured_content_messages: int,
    latency_ms: int,
) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "requested_model": requested_model,
        "structured_content_messages": structured_content_messages,
        "latency_ms": latency_ms,
    }


def _attach_controller_metadata(
    response: ChatCompletionResponse,
    metadata: Dict[str, Any],
) -> ChatCompletionResponse:
    merged = dict(response.controller_metadata or {})
    merged.update(metadata)
    response.controller_metadata = merged
    return response


def _create_mock_response(
    transport_request: ChatCompletionRequest,
    request_id: str,
) -> ChatCompletionResponse:
    normalized = transport_request.normalize()
    last_message = normalized.messages[-1].content if normalized.messages else ""

    return ChatCompletionResponse(
        id=f"mock-{request_id}",
        created=int(time.time()),
        model="controller-mock",
        choices=[
            Choice(
                index=0,
                message=Message(
                    role=Role.ASSISTANT,
                    content=(
                        "[MOCK MODE] Controller is reachable, but no LLM providers are configured. "
                        f"Last message preview: '{last_message[:160]}{'...' if len(last_message) > 160 else ''}'"
                    ),
                ),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        controller_metadata={
            "request_id": request_id,
            "mode": "mock",
            "requested_model": transport_request.model,
            "structured_content_messages": transport_request.structured_content_message_count(),
            "message": "Configure at least one provider for live responses",
        },
    )


def _stream_error_chunk(message: str, request_id: str) -> str:
    payload = {
        "error": {
            "message": message,
            "type": "stream_error",
            "request_id": request_id,
        }
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_controller_completion(
    request: NormalizedChatCompletionRequest,
    route: RouteDecision,
    *,
    request_id: str,
    allowed_providers: List[str],
) -> Any:
    assert fallback_provider is not None
    try:
        async for chunk in fallback_provider.chat_completion_stream(
            request,
            preferred_provider=route.selected_provider,
            allowed_providers=allowed_providers,
            target_model=route.selected_model,
        ):
            yield chunk
    except Exception as exc:
        logger.error(
            "[%s] Streaming controller request failed provider=%s model=%s error=%s",
            request_id,
            route.selected_provider,
            route.selected_model,
            exc,
        )
        yield _stream_error_chunk(str(exc), request_id)
        yield "data: [DONE]\n\n"


async def _stream_direct_completion(
    request: NormalizedChatCompletionRequest,
    provider_name: str,
    provider: Provider,
    *,
    target_model: str,
    request_id: str,
) -> Any:
    try:
        async for chunk in provider.chat_completion_stream(request, target_model=target_model):
            yield chunk
    except Exception as exc:
        logger.error(
            "[%s] Streaming direct request failed provider=%s model=%s error=%s",
            request_id,
            provider_name,
            target_model,
            exc,
        )
        yield _stream_error_chunk(str(exc), request_id)
        yield "data: [DONE]\n\n"


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Simple liveness endpoint."""
    return HealthResponse(status="healthy", version=__version__, timestamp=datetime.now(timezone.utc))


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(
    authorized: bool = Depends(verify_api_key),
    settings: Settings = Depends(get_settings),
) -> ModelsResponse:
    """List controller and backend models visible to OpenClaw."""
    del authorized

    models: List[ModelInfo] = [
        ModelInfo(id="controller", created=int(time.time()), owned_by="openclaw")
    ]

    seen: set[str] = {"controller"}
    for provider_name in providers.keys():
        for model_id in settings.get_provider_models(provider_name):
            if model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                ModelInfo(
                    id=model_id,
                    created=int(time.time()),
                    owned_by=provider_name,
                    root=model_id,
                )
            )

    return ModelsResponse(data=models)


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    authorized: bool = Depends(verify_api_key),
    settings: Settings = Depends(get_settings),
):
    """OpenAI-compatible chat completions endpoint."""
    del authorized

    request_id = _request_id_from_http(http_request)
    start_time = time.time()

    _ensure_runtime_ready()

    if rate_limiter:
        if not rate_limiter.is_allowed(_client_ip(http_request)):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    request_stats["total"] += 1
    normalized = request.normalize()

    try:
        if not providers:
            request_stats["successful"] += 1
            return _create_mock_response(request, request_id)

        if _controller_mode(normalized.model):
            compatibility_names, incompatibilities = _resolve_compatible_providers(normalized)
            if not compatibility_names:
                request_stats["failed"] += 1
                _raise_no_compatible_provider(request_id, normalized, incompatibilities)

            intent = intent_classifier.classify(normalized)
            safety = safety_checker.check(normalized)
            if safety.overall_risk == RiskLevel.CRITICAL:
                request_stats["failed"] += 1
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Request blocked due to safety concerns",
                        "request_id": request_id,
                        "risk_level": safety.overall_risk.value,
                        "risk_score": safety.risk_score,
                        "recommendations": safety.recommendations,
                    },
                )

            provider_health = await _provider_health_snapshot(compatibility_names)
            confidence = confidence_calc.calculate(
                request=normalized,
                provider_health=provider_health,
                safety_score=safety,
                intent=intent,
            )
            route = route_selector.select(
                request=normalized,
                confidence=confidence,
                safety=safety,
                intent=intent,
                provider_health=provider_health,
                available_providers=compatibility_names,
            )

            logger.info(
                "[%s] controller route requested_model=%s selected_provider=%s selected_model=%s compatible=%s confidence=%.3f intent=%s",
                request_id,
                normalized.model,
                route.selected_provider,
                route.selected_model,
                compatibility_names,
                confidence.overall,
                intent.value,
            )

            if normalized.stream:
                return StreamingResponse(
                    _stream_controller_completion(
                        normalized,
                        route,
                        request_id=request_id,
                        allowed_providers=compatibility_names,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

            response, used_provider = await fallback_provider.chat_completion(
                normalized,
                preferred_provider=route.selected_provider,
                allowed_providers=compatibility_names,
                target_model=route.selected_model,
            )
            latency_ms = int((time.time() - start_time) * 1000)
            _attach_controller_metadata(
                response,
                {
                    **_metadata_base(
                        request_id=request_id,
                        requested_model=request.model,
                        structured_content_messages=request.structured_content_message_count(),
                        latency_ms=latency_ms,
                    ),
                    "mode": "controller",
                    "routed_provider": route.selected_provider,
                    "routed_model": route.selected_model,
                    "actual_provider": used_provider,
                    "provider_response_model": response.model,
                    "confidence": confidence.overall,
                    "intent": intent.value,
                    "safety_risk": safety.overall_risk.value,
                    "compatible_providers": compatibility_names,
                    "provider_diagnostics": incompatibilities,
                },
            )

            confidence_calc.record_result(normalized, success=True)
            request_stats["successful"] += 1
            return response

        provider_name, target_model = _resolve_direct_provider(normalized, settings)
        provider = providers.get(provider_name)
        if provider is None:
            request_stats["failed"] += 1
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")

        try:
            provider.validate_request(normalized)
        except ValueError as exc:
            request_stats["failed"] += 1
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Request is incompatible with selected provider",
                    "request_id": request_id,
                    "provider": provider_name,
                    "model": target_model,
                    "reason": str(exc),
                },
            ) from exc

        logger.info(
            "[%s] direct route requested_model=%s provider=%s target_model=%s stream=%s",
            request_id,
            normalized.model,
            provider_name,
            target_model,
            normalized.stream,
        )

        if normalized.stream:
            return StreamingResponse(
                _stream_direct_completion(
                    normalized,
                    provider_name,
                    provider,
                    target_model=target_model,
                    request_id=request_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        response = await provider.chat_completion(normalized, target_model=target_model)
        latency_ms = int((time.time() - start_time) * 1000)
        _attach_controller_metadata(
            response,
            {
                **_metadata_base(
                    request_id=request_id,
                    requested_model=request.model,
                    structured_content_messages=request.structured_content_message_count(),
                    latency_ms=latency_ms,
                ),
                "mode": "direct",
                "routed_provider": provider_name,
                "routed_model": target_model,
                "actual_provider": provider_name,
                "provider_response_model": response.model,
            },
        )
        request_stats["successful"] += 1
        return response

    except HTTPException:
        if confidence_calc and _controller_mode(normalized.model):
            confidence_calc.record_result(normalized, success=False)
        raise
    except Exception as exc:
        logger.error("[%s] Request failed: %s", request_id, exc, exc_info=True)
        request_stats["failed"] += 1
        if confidence_calc and _controller_mode(normalized.model):
            confidence_calc.record_result(normalized, success=False)
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Upstream provider request failed",
                "request_id": request_id,
                "message": str(exc),
            },
        ) from exc


@router.get("/v1/controller/status", response_model=ControllerStatus)
async def controller_status(
    authorized: bool = Depends(verify_api_key),
    settings: Settings = Depends(get_settings),
) -> ControllerStatus:
    """Get detailed controller runtime status."""
    del authorized

    health_map = await _provider_health_snapshot()
    provider_health = list(health_map.values())

    uptime = int(time.time() - request_stats["start_time"])
    latencies = [item.latency_ms for item in provider_health if item.latency_ms is not None]
    average_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

    if not provider_health:
        status = "healthy"
    else:
        healthy_count = sum(1 for item in provider_health if item.status == "healthy")
        degraded_count = sum(1 for item in provider_health if item.status == "degraded")
        if healthy_count == len(provider_health):
            status = "healthy"
        elif healthy_count or degraded_count:
            status = "degraded"
        else:
            status = "unhealthy"

    return ControllerStatus(
        status=status,
        version=__version__,
        uptime_seconds=uptime,
        providers=provider_health,
        total_requests=request_stats["total"],
        successful_requests=request_stats["successful"],
        failed_requests=request_stats["failed"],
        average_latency_ms=average_latency,
        active_connections=0,
        config={
            "default_provider": settings.DEFAULT_PROVIDER,
            "fallback_providers": settings.FALLBACK_PROVIDERS,
            "rate_limit_enabled": settings.RATE_LIMIT_ENABLED,
            "pii_detection": settings.ENABLE_PII_DETECTION,
            "jailbreak_detection": settings.ENABLE_JAILBREAK_DETECTION,
            "active_providers": list(providers.keys()),
        },
    )


@router.post("/v1/controller/explain", response_model=ExplainabilityReport)
async def explain_request(
    request: ChatCompletionRequest,
    http_request: Request,
    authorized: bool = Depends(verify_api_key),
):
    """Explain how the controller would route a request without executing it."""
    del authorized

    request_id = _request_id_from_http(http_request)
    _ensure_runtime_ready()

    normalized = request.normalize()
    intent = intent_classifier.classify(normalized)
    safety = safety_checker.check(normalized)

    compatible_names, incompatibilities = _resolve_compatible_providers(normalized)
    provider_health = await _provider_health_snapshot(compatible_names)
    confidence = confidence_calc.calculate(
        request=normalized,
        provider_health=provider_health,
        safety_score=safety,
        intent=intent,
    )

    if not providers:
        route = RouteDecision(
            selected_provider="mock",
            selected_model="controller-mock",
            selected_tier=confidence.tier_recommendation,
            confidence=confidence,
            safety=safety,
            intent=intent,
            reasoning="No providers are currently configured; controller would return a mock response.",
            fallback_chain=[],
        )
    elif compatible_names:
        route = route_selector.select(
            request=normalized,
            confidence=confidence,
            safety=safety,
            intent=intent,
            provider_health=provider_health,
            available_providers=compatible_names,
        )
    else:
        route = RouteDecision(
            selected_provider="none",
            selected_model="none",
            selected_tier=confidence.tier_recommendation,
            confidence=confidence,
            safety=safety,
            intent=intent,
            reasoning="No compatible providers are available for this request shape.",
            fallback_chain=[],
        )

    alternatives: List[Dict[str, Any]] = []
    for provider_name in providers.keys():
        if provider_name == route.selected_provider:
            continue
        alternative_reason = incompatibilities.get(
            provider_name,
            f"Lower priority than selected provider for {route.selected_tier} tier",
        )
        alternatives.append({"provider": provider_name, "reason": alternative_reason})

    return ExplainabilityReport(
        request_id=request_id,
        original_request=request.model_dump(),
        route_decision=route,
        confidence_breakdown={
            "overall": confidence.overall,
            "factors": [factor.model_dump() for factor in confidence.factors],
            "reasoning": confidence.reasoning,
        },
        safety_analysis={
            "risk_level": safety.overall_risk.value,
            "risk_score": safety.risk_score,
            "pii_count": len(safety.pii_detected),
            "jailbreak_attempts": len(safety.jailbreak_attempts),
            "recommendations": safety.recommendations,
        },
        intent_classification={
            "detected_intent": intent.value,
            "keywords_matched": _get_intent_keywords(intent),
        },
        provider_selection_logic=route.reasoning,
        alternatives_considered=alternatives,
        user_recommendations=_generate_recommendations(route, safety, confidence),
    )


def _get_intent_keywords(intent: IntentType) -> List[str]:
    from ..core.intelligence import INTENT_KEYWORDS

    return INTENT_KEYWORDS.get(intent, [])


def _generate_recommendations(
    route: RouteDecision,
    safety: SafetyScore,
    confidence: ConfidenceScore,
) -> List[str]:
    recommendations: List[str] = []

    if safety.overall_risk != RiskLevel.LOW:
        recommendations.extend(safety.recommendations)

    if confidence.overall < 0.5:
        recommendations.append("Consider simplifying or narrowing the request for higher confidence routing")

    if route.selected_tier == "premium":
        recommendations.append("This request is routed to the premium tier and may incur higher backend cost")

    if route.selected_provider == "none":
        recommendations.append("Add or enable a provider that supports the request payload shape")

    if route.selected_provider == "mock":
        recommendations.append("Configure at least one provider to receive live model responses")

    if not recommendations:
        recommendations.append("No specific recommendations")

    return recommendations
