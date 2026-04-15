"""API routes kept under v130 name for backward compatibility, backed by v1.4 core."""

from __future__ import annotations

import logging
import traceback
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Request

from controller.core.config_v130 import SettingsV13
from controller.core.execution_engine import ExecutionEngine, ExecutionEngineError
from controller.core.health_service import RouteHealthService
from controller.core.hybrid_router import HybridRouter
from controller.core.provider_bootstrap import bootstrap_provider_registry
from controller.core.request_normalizer import RequestNormalizer
from controller.core.response_normalizer import ResponseNormalizer
from controller.core.routing_policy_v140 import RoutingPolicyV140
from controller.core.session_manager import (
    FileSessionStore,
    IdentityManager,
    RedisSessionStore,
    SessionManager,
)
from controller.core.token_estimator import TokenEstimator
from controller.core.validation_service import RouteValidationService

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances initialized during app startup.
session_manager: Optional[SessionManager] = None
registry = None
validation_service: Optional[RouteValidationService] = None
health_service: Optional[RouteHealthService] = None
hybrid_router: Optional[HybridRouter] = None
execution_engine: Optional[ExecutionEngine] = None
request_normalizer: Optional[RequestNormalizer] = None
response_normalizer: Optional[ResponseNormalizer] = None
settings_obj: Optional[SettingsV13] = None

DEFAULT_FEATURE_FLAGS = {
    "session_persistence": True,
    "identity_caching": True,
    "context_preservation": True,
    "safe_routing": True,
    "validation_pipeline": True,
    "routing_logging": True,
}


def init_controller_v130(settings: SettingsV13) -> None:
    """Initialize Controller v1.4 components while preserving old bootstrap signature."""
    global session_manager, registry, validation_service, health_service
    global hybrid_router, execution_engine, request_normalizer, response_normalizer, settings_obj

    settings_obj = settings
    registry = bootstrap_provider_registry(settings)

    if settings.SESSION_STORE_TYPE == "redis":
        store = RedisSessionStore(redis_url=settings.SESSION_REDIS_URL, default_ttl=settings.SESSION_TTL_SECONDS)
    else:
        store = FileSessionStore(base_path=settings.SESSION_FILE_PATH)

    identity_manager = IdentityManager(identity_dir=settings.IDENTITY_DIR)
    session_manager = SessionManager(store=store, identity_manager=identity_manager, session_ttl=settings.SESSION_TTL_SECONDS)

    estimator = TokenEstimator(
        chars_per_token=settings.TOKEN_ESTIMATION_CHARS_PER_TOKEN,
        safety_margin=settings.TOKEN_ESTIMATION_SAFETY_MARGIN,
    )
    request_normalizer = RequestNormalizer(token_estimator=estimator)
    response_normalizer = ResponseNormalizer()

    validation_service = RouteValidationService(
        registry=registry,
        timeout_seconds=settings.VALIDATION_PROBE_TIMEOUT_SECONDS,
        cache_ttl_seconds=settings.VALIDATION_CACHE_TTL_SECONDS,
    )
    health_service = RouteHealthService(
        registry=registry,
        validation_service=validation_service,
        cache_ttl_seconds=settings.HEALTH_CACHE_TTL_SECONDS,
        timeout_seconds=settings.HEALTH_PROBE_TIMEOUT_SECONDS,
    )
    policy = RoutingPolicyV140(
        context_window_threshold=settings.CONTEXT_WINDOW_THRESHOLD,
        fallback_max_hops=settings.ROUTE_FALLBACK_MAX_HOPS,
        route_stickiness_enabled=settings.ROUTE_STICKINESS_ENABLED,
    )
    hybrid_router = HybridRouter(
        registry=registry,
        validation_service=validation_service,
        health_service=health_service,
        policy=policy,
        default_provider=settings.DEFAULT_PROVIDER,
    )
    execution_engine = ExecutionEngine(
        registry=registry,
        timeout_for_profile=settings.timeout_for_profile,
        max_retries_same_route=1,
    )

    logger.info("Controller v1.4 initialized successfully")


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_session_id: Optional[str] = Header(None),
    x_agent_id: Optional[str] = Header(None),
    x_pin_provider: Optional[str] = Header(None),
    x_pin_model: Optional[str] = Header(None),
    x_routing_mode: Optional[str] = Header(None),
):
    """Backward-compatible chat completions endpoint powered by route-aware v1.4 pipeline."""
    if not all([session_manager, hybrid_router, execution_engine, request_normalizer, response_normalizer]):
        raise HTTPException(status_code=503, detail="Controller not initialized")

    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        body = await request.json()
        incoming_messages = body.get("messages", [])

        # Session bootstrap / lookup.
        session = None
        if x_session_id:
            session = await session_manager.get_session(x_session_id)

        if session is None:
            agent_id = x_agent_id or "default_agent"
            session = await session_manager.create_session(
                agent_id=agent_id,
                pinned_provider=x_pin_provider,
                feature_flags=DEFAULT_FEATURE_FLAGS.copy(),
            )
            logger.info("[%s] Created new session: %s", request_id, session.session_id)
        else:
            logger.info("[%s] Using existing session: %s", request_id, session.session_id)

        # Optional compatibility pin update on existing sessions.
        if x_pin_provider:
            session.pinned_provider = x_pin_provider
            await session_manager.store.set(session.session_id, session, ttl=session_manager.session_ttl)

        routing_mode = x_routing_mode or (settings_obj.ROUTING_MODE if settings_obj else "balanced")

        canonical_request = request_normalizer.normalize(
            body=body,
            session=session,
            agent_id=session.metadata.get("agent_id", x_agent_id or "default_agent"),
            session_id=session.session_id,
            provider_hint=x_pin_provider or session.pinned_provider,
            model_hint=x_pin_model,
            routing_mode=routing_mode,
        )

        requirements = request_normalizer.build_requirements(
            canonical_request=canonical_request,
            provider_hint=x_pin_provider or session.pinned_provider or (settings_obj.DEFAULT_PROVIDER if settings_obj else None),
            model_hint=x_pin_model,
            routing_mode=routing_mode,
        )

        previous_route = session.current_route
        previous_provider = session.current_provider

        decision = await hybrid_router.select_route(session, requirements)
        route_chain = hybrid_router.build_route_chain(decision)
        try:
            print(f"ROUTES_V130_PRE_EXEC request_id={request_id} route_chain={route_chain!r}", flush=True)
        except Exception:
            pass

        result, execution_report = await execution_engine.execute(
            canonical_request=canonical_request,
            decision=decision,
            route_chain=route_chain,
        )

        # Persist session state updates.
        await session_manager.update_route(
            session_id=session.session_id,
            route_key=execution_report.used_route_key,
            provider_id=result.provider_id,
            mark_successful=True,
        )

        if decision.context_transfer_required:
            await session_manager.record_routing_event(
                session_id=session.session_id,
                from_provider=previous_provider,
                to_provider=result.provider_id,
                reason=decision.reason_str,
                context_preserved=True,
                token_count=decision.estimated_tokens,
                from_route=previous_route,
                to_route=execution_report.used_route_key,
                api_dialect=result.api_dialect.value,
                metadata={"reason_code": decision.reason_code},
            )

        await session_manager.set_context_transfer_meta(
            session_id=session.session_id,
            transfer_meta={
                "mode": decision.transfer_policy.mode.value,
                "preserve_identity": decision.transfer_policy.preserve_identity,
                "truncation_applied": decision.transfer_policy.truncation_applied,
            },
        )

        # Persist only canonical conversation state (never provider payloads/secrets).
        for msg in incoming_messages:
            if msg.get("role") == "system":
                continue
            content = str(msg.get("content", ""))
            await session_manager.add_message(
                session_id=session.session_id,
                role=msg.get("role", "user"),
                content=content,
                provider=None,
                model=None,
                tokens=max(1, len(content) // 4),
            )

        await session_manager.add_message(
            session_id=session.session_id,
            role="assistant",
            content=result.content,
            provider=result.provider_id,
            model=result.model_id,
            tokens=max(1, result.usage.completion_tokens),
            metadata={
                "api_dialect": result.api_dialect.value,
                "route_key": execution_report.used_route_key,
            },
        )

        response = response_normalizer.to_chat_completions_response(
            request_id=request_id,
            session_id=session.session_id,
            requested_model=body.get("model", "controller"),
            decision=decision,
            execution_result=result,
        )

        elapsed = time.time() - start_time
        logger.info(
            "[%s] Request completed in %.2fs via %s",
            request_id,
            elapsed,
            execution_report.used_route_key,
        )
        return response

    except ExecutionEngineError as exc:
        logger.error("[%s] Route execution failed: %s", request_id, str(exc))
        raise HTTPException(
            status_code=502,
            detail={
                "error": "All route attempts failed",
                "attempts": [
                    {
                        "route_key": attempt.route_key,
                        "success": attempt.success,
                        "error_kind": attempt.error_kind,
                        "error_message": attempt.error_message,
                    }
                    for attempt in exc.attempts
                ],
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[%s] Unhandled error", request_id)
        try:
            print(f"ROUTES_V130_UNHANDLED request_id={request_id} exc={exc!r}", flush=True)
            traceback.print_exc()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/health")
async def health_check():
    if not registry:
        return {"status": "not_initialized", "version": "1.4.0"}

    providers = {}
    for provider_id in ["openai", "anthropic", "ollama"]:
        provider = registry.get_provider(provider_id)
        if not provider:
            continue
        summary = registry.get_aggregate_provider_health(provider_id)
        providers[provider_id] = summary["status"]

    return {
        "status": "healthy",
        "version": "1.4.0",
        "components": {
            "session_manager": "healthy" if session_manager else "not_initialized",
            "hybrid_router": "healthy" if hybrid_router else "not_initialized",
            "execution_engine": "healthy" if execution_engine else "not_initialized",
        },
        "providers": providers,
    }


@router.get("/v1/controller/status")
async def controller_status():
    if not registry:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    providers: Dict[str, Any] = {}
    for provider_id in ["openai", "anthropic", "ollama"]:
        provider = registry.get_provider(provider_id)
        if provider:
            providers[provider_id] = registry.get_aggregate_provider_health(provider_id)

    routes = [route.to_dict() for route in registry.list_routes(enabled_only=True)]

    return {
        "version": "1.4.0",
        "features": DEFAULT_FEATURE_FLAGS,
        "providers": providers,
        "routes_registered": len(routes),
        "routes": routes,
        "session_store": {
            "type": "redis" if isinstance(session_manager.store, RedisSessionStore) else "file"
            if session_manager
            else "unknown"
        },
    }


@router.post("/v1/controller/session")
async def create_session_endpoint(agent_id: str, pin_provider: Optional[str] = None):
    if not session_manager:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    session = await session_manager.create_session(
        agent_id=agent_id,
        pinned_provider=pin_provider,
        feature_flags=DEFAULT_FEATURE_FLAGS.copy(),
    )

    return {
        "session_id": session.session_id,
        "agent_id": agent_id,
        "identity": {
            "name": session.identity.name,
            "hash": session.identity.hash,
        },
        "pinned_provider": pin_provider,
        "created_at": session.created_at.isoformat(),
    }


@router.get("/v1/controller/session/{session_id}")
async def get_session_endpoint(session_id: str):
    if not session_manager:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "identity": {
            "name": session.identity.name,
            "version": session.identity.version,
        },
        "message_count": session.message_count,
        "total_tokens": session.total_tokens,
        "current_provider": session.current_provider,
        "pinned_provider": session.pinned_provider,
        "current_route": session.current_route,
        "pinned_route": session.pinned_route,
        "routing_history": len(session.routing_history),
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }
