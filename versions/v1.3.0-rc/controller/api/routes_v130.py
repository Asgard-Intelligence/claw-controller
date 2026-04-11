"""
API Routes for Controller v1.3.0
Session-aware, identity-preserving, safe routing
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from controller.core.session_manager import (
    SessionManager,
    RedisSessionStore,
    FileSessionStore,
    IdentityManager,
    SessionState,
    Message,
)
from controller.core.safe_router import (
    SafeRoutingEngine,
    ProviderRegistry,
    RoutingPolicy,
    ModelCapabilities,
    ContextPreservationEngine,
)

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

# Global instances (initialized in lifespan)
session_manager: Optional[SessionManager] = None
routing_engine: Optional[SafeRoutingEngine] = None
context_engine: Optional[ContextPreservationEngine] = None
provider_registry: Optional[ProviderRegistry] = None

# Feature flags (can be overridden per session)
DEFAULT_FEATURE_FLAGS = {
    "session_persistence": True,
    "identity_caching": True,
    "context_preservation": True,
    "safe_routing": True,
    "validation_pipeline": True,
    "routing_logging": True,
}


def init_controller_v130(settings):
    """Initialize Controller v1.3.0 components"""
    global session_manager, routing_engine, context_engine, provider_registry
    
    # Initialize provider registry
    provider_registry = ProviderRegistry()
    
    # Register known providers with capabilities
    provider_registry.register(ModelCapabilities(
        provider="openai",
        model="gpt-4",
        context_window=8192,
        supports_tools=True,
    ))
    provider_registry.register(ModelCapabilities(
        provider="openai",
        model="gpt-4-turbo",
        context_window=128000,
        supports_tools=True,
    ))
    provider_registry.register(ModelCapabilities(
        provider="anthropic",
        model="claude-3-opus",
        context_window=200000,
        supports_tools=True,
    ))
    provider_registry.register(ModelCapabilities(
        provider="anthropic",
        model="claude-3-sonnet",
        context_window=200000,
        supports_tools=True,
    ))
    provider_registry.register(ModelCapabilities(
        provider="ollama",
        model="gemma4-e4b-q4-local",
        context_window=8192,
        supports_tools=False,
    ))
    provider_registry.register(ModelCapabilities(
        provider="ollama",
        model="gemma4-31b-q4-local",
        context_window=32768,
        supports_tools=False,
    ))
    
    # Set initial health status
    provider_registry.set_health("openai", "healthy")
    provider_registry.set_health("anthropic", "healthy")
    provider_registry.set_health("ollama", "healthy")
    
    # Initialize session store
    if settings.SESSION_STORE_TYPE == "redis":
        store = RedisSessionStore(
            redis_url=settings.SESSION_REDIS_URL,
            default_ttl=settings.SESSION_TTL_SECONDS
        )
    else:
        store = FileSessionStore(
            base_path=settings.SESSION_FILE_PATH
        )
    
    # Initialize identity manager
    identity_manager = IdentityManager(
        identity_dir=settings.IDENTITY_DIR
    )
    
    # Initialize session manager
    session_manager = SessionManager(
        store=store,
        identity_manager=identity_manager,
        session_ttl=settings.SESSION_TTL_SECONDS
    )
    
    # Initialize routing policy and engine
    routing_policy = RoutingPolicy(registry=provider_registry)
    routing_engine = SafeRoutingEngine(
        registry=provider_registry,
        policy=routing_policy,
        default_provider=settings.DEFAULT_PROVIDER
    )
    
    # Initialize context preservation engine
    context_engine = ContextPreservationEngine(registry=provider_registry)
    
    logger.info("Controller v1.3.0 initialized successfully")


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str = Header(None),
    x_session_id: Optional[str] = Header(None),
    x_agent_id: Optional[str] = Header(None),
    x_pin_provider: Optional[str] = Header(None),
):
    """
    Chat completions with session management and safe routing
    
    Headers:
        - x-session-id: Existing session ID (optional)
        - x-agent-id: Agent ID for identity loading (required for new session)
        - x-pin-provider: Pin specific provider for session (optional)
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Parse request body
        body = await request.json()
        messages = body.get("messages", [])
        model = body.get("model", "controller")
        stream = body.get("stream", False)
        
        # Get or create session
        session = None
        if x_session_id:
            session = await session_manager.get_session(x_session_id)
        
        is_new_session = session is None
        
        if is_new_session:
            # Create new session
            agent_id = x_agent_id or "default_agent"
            session = await session_manager.create_session(
                agent_id=agent_id,
                pinned_provider=x_pin_provider,
                feature_flags=DEFAULT_FEATURE_FLAGS.copy()
            )
            logger.info(f"[{request_id}] Created new session: {session.session_id}")
        else:
            logger.info(f"[{request_id}] Using existing session: {session.session_id}")
        
        # Select provider using safe routing
        routing_decision = await routing_engine.select_provider(session, body)
        
        logger.info(
            f"[{request_id}] Routing decision: {routing_decision.provider} "
            f"(reason: {routing_decision.reason_str}, "
            f"context_transfer: {routing_decision.context_transfer_required})"
        )
        
        # Prepare context for provider
        if routing_decision.context_transfer_required:
            context_result = await context_engine.prepare_context_for_routing(
                session,
                routing_decision.provider
            )
            messages_to_send = context_result["messages"]
            logger.info(
                f"[{request_id}] Context prepared: "
                f"{context_result['original_count']} -> {context_result['final_count']} messages, "
                f"identity_preserved: {context_result['identity_preserved']}"
            )
        else:
            # Use session messages + new message
            messages_to_send = []
            
            # Add system prompt with identity
            if session.identity:
                messages_to_send.append({
                    "role": "system",
                    "content": session.identity.system_prompt
                })
            
            # Add conversation history
            for msg in session.messages:
                messages_to_send.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Add new messages
            messages_to_send.extend(messages)
        
        # Update session with current provider
        await session_manager.update_provider(session.session_id, routing_decision.provider)
        
        # Record routing event if this was a transfer
        if routing_decision.context_transfer_required:
            await session_manager.record_routing_event(
                session_id=session.session_id,
                from_provider=session.current_provider,
                to_provider=routing_decision.provider,
                reason=routing_decision.reason_str,
                context_preserved=True,
                token_count=routing_decision.estimated_tokens
            )
        
        # Call provider (simplified - would call actual provider in production)
        provider_response = await _call_provider(
            provider=routing_decision.provider,
            messages=messages_to_send,
            model=body.get("model"),
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
        )
        
        # Add user message to session
        if messages:
            await session_manager.add_message(
                session_id=session.session_id,
                role="user",
                content=messages[-1].get("content", ""),
                provider=None,
                model=None,
                tokens=len(messages[-1].get("content", "")) // 4
            )
        
        # Add assistant response to session
        response_content = provider_response.get("content", "")
        await session_manager.add_message(
            session_id=session.session_id,
            role="assistant",
            content=response_content,
            provider=routing_decision.provider,
            model=provider_response.get("model"),
            tokens=len(response_content) // 4
        )
        
        # Build response
        response = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "session_id": session.session_id,
            "routing": {
                "provider": routing_decision.provider,
                "reason": routing_decision.reason_str,
                "context_transfer": routing_decision.context_transfer_required,
            },
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content,
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": routing_decision.estimated_tokens,
                "completion_tokens": len(response_content) // 4,
                "total_tokens": routing_decision.estimated_tokens + len(response_content) // 4,
            }
        }
        
        elapsed = time.time() - start_time
        logger.info(f"[{request_id}] Request completed in {elapsed:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _call_provider(
    provider: str,
    messages: List[Dict],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict:
    """
    Call LLM provider (simplified mock implementation)
    In production, this would call actual provider APIs
    """
    # Mock response for demonstration
    last_message = messages[-1].get("content", "") if messages else ""
    
    # Simple mock logic for testing
    if "как тебя зовут" in last_message.lower() or "what is your name" in last_message.lower():
        # Extract name from system prompt
        name = "Assistant"
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                import re
                match = re.search(r'[Nn]ame is (\w+)', content)
                if match:
                    name = match.group(1)
                    break
        content = f"My name is {name}."
    elif "как меня зовут" in last_message.lower() or "what is my name" in last_message.lower():
        # Look for user's name in conversation
        name = None
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                import re
                match = re.search(r'[Мм]еня зовут (\w+)|[Mm]y name is (\w+)', content)
                if match:
                    name = match.group(1) or match.group(2)
                    break
        if name:
            content = f"Your name is {name}."
        else:
            content = "I don't know your name yet."
    else:
        content = f"Response from {provider}: Acknowledged."
    
    return {
        "content": content,
        "model": model or f"{provider}-default",
        "provider": provider,
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    health = {
        "status": "healthy",
        "version": "1.3.0",
        "components": {}
    }
    
    # Check session manager
    if session_manager:
        health["components"]["session_manager"] = "healthy"
    else:
        health["components"]["session_manager"] = "not_initialized"
    
    # Check routing engine
    if routing_engine:
        health["components"]["routing_engine"] = "healthy"
    else:
        health["components"]["routing_engine"] = "not_initialized"
    
    # Provider health
    if provider_registry:
        health["providers"] = {
            name: provider_registry.get_health(name)
            for name in ["openai", "anthropic", "ollama"]
        }
    
    return health


@router.get("/v1/controller/status")
async def controller_status():
    """Detailed controller status"""
    return {
        "version": "1.3.0",
        "features": DEFAULT_FEATURE_FLAGS,
        "providers": {
            "registered": list(provider_registry._capabilities.keys()) if provider_registry else [],
            "health": {
                name: provider_registry.get_health(name)
                for name in ["openai", "anthropic", "ollama"]
            } if provider_registry else {}
        },
        "session_store": {
            "type": "redis" if isinstance(session_manager.store, RedisSessionStore) else "file"
                if session_manager else "unknown",
        },
    }


@router.post("/v1/controller/session")
async def create_session_endpoint(
    agent_id: str,
    pin_provider: Optional[str] = None,
):
    """Create new session explicitly"""
    session = await session_manager.create_session(
        agent_id=agent_id,
        pinned_provider=pin_provider,
        feature_flags=DEFAULT_FEATURE_FLAGS.copy()
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
    """Get session details"""
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
        "routing_history": len(session.routing_history),
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }
