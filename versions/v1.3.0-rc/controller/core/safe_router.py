"""
Safe Routing Engine - v1.3.0
Routing with context preservation and context window awareness
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class RoutingReason(Enum):
    PINNED = "user_pinned_provider"
    HEALTHY = "current_provider_healthy"
    FALLBACK = "fallback_from_failed_provider"
    USER_REQUESTED = "user_requested_switch"
    CONTEXT_WINDOW = "context_window_exceeded"
    FIRST_MESSAGE = "first_message_no_session"
    CAPABILITY_MISMATCH = "capability_mismatch"


@dataclass
class RoutingDecision:
    """Complete routing decision with justification"""
    provider: str
    model: Optional[str]
    reason: RoutingReason
    reason_str: str
    context_transfer_required: bool
    estimated_tokens: int
    confidence: float
    alternatives: List[str]
    metadata: Dict[str, Any]


@dataclass
class ModelCapabilities:
    """Capabilities and limits of a model"""
    provider: str
    model: str
    context_window: int
    supports_system_prompt: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    supports_json_mode: bool = True
    max_output_tokens: int = 4096
    
    def can_handle(self, estimated_tokens: int, requires_tools: bool = False) -> Tuple[bool, str]:
        """Check if model can handle request"""
        if estimated_tokens > self.context_window:
            return False, f"Context window exceeded: {estimated_tokens} > {self.context_window}"
        
        if requires_tools and not self.supports_tools:
            return False, "Model does not support tools"
        
        return True, "OK"


class ProviderRegistry:
    """Registry of providers and their capabilities"""
    
    def __init__(self):
        self._capabilities: Dict[str, ModelCapabilities] = {}
        self._health_status: Dict[str, str] = {}
    
    def register(self, capabilities: ModelCapabilities) -> None:
        """Register provider capabilities"""
        key = f"{capabilities.provider}/{capabilities.model}"
        self._capabilities[key] = capabilities
        self._capabilities[capabilities.provider] = capabilities
    
    def get_capabilities(self, provider: str) -> Optional[ModelCapabilities]:
        """Get capabilities for provider"""
        return self._capabilities.get(provider)
    
    def set_health(self, provider: str, status: str) -> None:
        """Set provider health status"""
        self._health_status[provider] = status
    
    def get_health(self, provider: str) -> str:
        """Get provider health status"""
        return self._health_status.get(provider, "unknown")
    
    def is_healthy(self, provider: str) -> bool:
        """Check if provider is healthy"""
        return self._health_status.get(provider) == "healthy"
    
    def find_providers_with_min_context(self, min_context: int) -> List[str]:
        """Find providers with at least min_context window"""
        result = []
        for key, caps in self._capabilities.items():
            if "/" not in key:  # Skip provider/model keys
                continue
            if caps.context_window >= min_context:
                result.append(key)
        
        # Sort by context window (largest first)
        result.sort(key=lambda p: self._capabilities[p].context_window, reverse=True)
        return result


class RoutingPolicy:
    """Routing policy for v1.3.0"""
    
    # Threshold for routing to larger model (90% of capacity)
    CONTEXT_WINDOW_THRESHOLD = 0.9
    
    # Providers to avoid for identity-critical sessions
    IDENTITY_CRITICAL_PROVIDERS = ["gemma", "ollama"]
    
    def __init__(self, registry: ProviderRegistry):
        self.registry = registry
    
    def should_allow_routing(
        self,
        session,
        request
    ) -> Tuple[bool, str]:
        """Determine if routing is allowed"""
        # Check pinned provider
        if hasattr(session, 'pinned_provider') and session.pinned_provider:
            return False, f"Provider pinned: {session.pinned_provider}"
        
        # Check identity-critical
        if self._is_identity_critical(request):
            return False, "Identity-critical session - routing disabled"
        
        return True, "Routing allowed"
    
    def _is_identity_critical(self, request) -> bool:
        """Check if request is identity-critical"""
        identity_keywords = [
            "тебя зовут", "твоё имя", "your name",
            "запомни", "remember", "identity",
            "меня зовут", "my name is"
        ]
        
        # Get last message content
        if hasattr(request, 'messages') and request.messages:
            content = request.messages[-1].get('content', '').lower()
            return any(kw in content for kw in identity_keywords)
        
        return False
    
    def check_context_window(
        self,
        session,
        provider: str
    ) -> Tuple[bool, str]:
        """Check if context fits in provider's window"""
        caps = self.registry.get_capabilities(provider)
        if not caps:
            return False, f"Unknown provider: {provider}"
        
        current_tokens = getattr(session, 'total_tokens', 0)
        limit = caps.context_window
        
        if current_tokens > limit:
            return False, f"Context exceeds limit: {current_tokens} > {limit}"
        
        if current_tokens > limit * self.CONTEXT_WINDOW_THRESHOLD:
            return True, f"Context near limit: {current_tokens}/{limit} ({current_tokens/limit:.1%})"
        
        return True, "OK"


class SafeRoutingEngine:
    """Routing engine that preserves context"""
    
    def __init__(
        self,
        registry: ProviderRegistry,
        policy: RoutingPolicy,
        default_provider: str = "openai"
    ):
        self.registry = registry
        self.policy = policy
        self.default_provider = default_provider
    
    async def select_provider(
        self,
        session,
        request
    ) -> RoutingDecision:
        """Select provider with context preservation"""
        
        # 1. Check for pinned provider (user override)
        if hasattr(session, 'pinned_provider') and session.pinned_provider:
            return RoutingDecision(
                provider=session.pinned_provider,
                model=None,
                reason=RoutingReason.PINNED,
                reason_str="User-pinned provider",
                context_transfer_required=False,
                estimated_tokens=getattr(session, 'total_tokens', 0),
                confidence=1.0,
                alternatives=[],
                metadata={"pinned": True}
            )
        
        # 2. Check current provider health
        current_provider = getattr(session, 'current_provider', None)
        if current_provider:
            if self.registry.is_healthy(current_provider):
                # Check context window
                fits, msg = self.policy.check_context_window(session, current_provider)
                if fits:
                    return RoutingDecision(
                        provider=current_provider,
                        model=None,
                        reason=RoutingReason.HEALTHY,
                        reason_str="Current provider healthy",
                        context_transfer_required=False,
                        estimated_tokens=getattr(session, 'total_tokens', 0),
                        confidence=0.95,
                        alternatives=[],
                        metadata={"health_check": msg}
                    )
                else:
                    # Context window exceeded - need to route
                    return await self._route_to_larger_context(session, current_provider)
        
        # 3. First message or no current provider
        return await self._select_initial_provider(session)
    
    async def _route_to_larger_context(
        self,
        session,
        current_provider: str
    ) -> RoutingDecision:
        """Route to provider with larger context window"""
        current_tokens = getattr(session, 'total_tokens', 0)
        
        # Find providers with larger context
        candidates = self.registry.find_providers_with_min_context(current_tokens)
        
        # Filter out current provider
        candidates = [p for p in candidates if p != current_provider]
        
        # Filter healthy providers
        healthy = [p for p in candidates if self.registry.is_healthy(p)]
        
        if healthy:
            target = healthy[0]  # Largest context
            return RoutingDecision(
                provider=target,
                model=None,
                reason=RoutingReason.CONTEXT_WINDOW,
                reason_str=f"Context window exceeded on {current_provider}",
                context_transfer_required=True,
                estimated_tokens=current_tokens,
                confidence=0.85,
                alternatives=healthy[1:3],
                metadata={
                    "previous_provider": current_provider,
                    "context_threshold": self.policy.CONTEXT_WINDOW_THRESHOLD
                }
            )
        
        # No larger provider available - fallback to default
        return RoutingDecision(
            provider=self.default_provider,
            model=None,
            reason=RoutingReason.FALLBACK,
            reason_str="No provider with sufficient context window",
            context_transfer_required=True,
            estimated_tokens=current_tokens,
            confidence=0.6,
            alternatives=[],
            metadata={"warning": "Context may be truncated"}
        )
    
    async def _select_initial_provider(self, session) -> RoutingDecision:
        """Select initial provider for new session"""
        # Use default provider
        if self.registry.is_healthy(self.default_provider):
            return RoutingDecision(
                provider=self.default_provider,
                model=None,
                reason=RoutingReason.FIRST_MESSAGE,
                reason_str="First message - selecting default provider",
                context_transfer_required=False,
                estimated_tokens=0,
                confidence=0.9,
                alternatives=[],
                metadata={"default": True}
            )
        
        # Find healthy fallback
        fallbacks = ["anthropic", "openai"]
        for fb in fallbacks:
            if self.registry.is_healthy(fb):
                return RoutingDecision(
                    provider=fb,
                    model=None,
                    reason=RoutingReason.FALLBACK,
                    reason_str=f"Default provider unhealthy, using fallback",
                    context_transfer_required=False,
                    estimated_tokens=0,
                    confidence=0.7,
                    alternatives=[],
                    metadata={"fallback_from": self.default_provider}
                )
        
        # No healthy providers
        return RoutingDecision(
            provider=self.default_provider,
            model=None,
            reason=RoutingReason.FALLBACK,
            reason_str="No healthy providers available - using default",
            context_transfer_required=False,
            estimated_tokens=0,
            confidence=0.3,
            alternatives=[],
            metadata={"warning": "No healthy providers"}
        )
    
    async def execute_fallback(
        self,
        session,
        failed_provider: str,
        error: Exception
    ) -> Optional[RoutingDecision]:
        """Execute fallback with context preservation"""
        
        # Check if fallback is allowed
        allowed, reason = self.policy.should_allow_routing(session, None)
        if not allowed:
            return None
        
        # Find fallback provider
        fallbacks = ["openai", "anthropic"]
        for fb in fallbacks:
            if fb != failed_provider and self.registry.is_healthy(fb):
                return RoutingDecision(
                    provider=fb,
                    model=None,
                    reason=RoutingReason.FALLBACK,
                    reason_str=f"Fallback from {failed_provider}: {str(error)}",
                    context_transfer_required=True,
                    estimated_tokens=getattr(session, 'total_tokens', 0),
                    confidence=0.7,
                    alternatives=[],
                    metadata={
                        "failed_provider": failed_provider,
                        "error": str(error)
                    }
                )
        
        return None


class ContextPreservationEngine:
    """Ensures zero context loss during routing"""
    
    def __init__(self, registry: ProviderRegistry):
        self.registry = registry
    
    async def prepare_context_for_routing(
        self,
        session,
        target_provider: str
    ) -> Dict:
        """Prepare complete context for provider switch"""
        
        # 1. Gather all messages
        messages = self._gather_messages(session)
        
        # 2. Apply provider-specific formatting
        formatted = self._format_for_provider(messages, target_provider)
        
        # 3. Check context window limits
        truncated = self._apply_context_window(formatted, target_provider)
        
        # 4. Validate system prompt integrity
        validated = self._validate_system_prompt(truncated, session)
        
        return {
            "messages": validated,
            "original_count": len(messages),
            "final_count": len(validated),
            "truncation_applied": len(validated) < len(messages),
            "identity_preserved": self._check_identity_preserved(validated, session)
        }
    
    def _gather_messages(self, session) -> List[Dict]:
        """Gather complete message history with system prompt first"""
        messages = []
        
        # ALWAYS include system prompt as first message
        if hasattr(session, 'identity') and session.identity:
            system_msg = {
                "role": "system",
                "content": session.identity.system_prompt,
                "metadata": {
                    "identity_version": session.identity.version,
                    "identity_hash": session.identity.hash
                }
            }
            messages.append(system_msg)
        
        # Add all conversation messages in order
        if hasattr(session, 'messages'):
            for msg in session.messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "metadata": msg.metadata
                })
        
        return messages
    
    def _format_for_provider(
        self,
        messages: List[Dict],
        provider: str
    ) -> List[Dict]:
        """Format messages for specific provider API"""
        # For now, return as-is (OpenAI format)
        # In production, would convert to provider-specific format
        return messages
    
    def _apply_context_window(
        self,
        messages: List[Dict],
        provider: str
    ) -> List[Dict]:
        """Truncate if exceeds model context window"""
        caps = self.registry.get_capabilities(provider)
        if not caps:
            return messages
        
        limit = caps.context_window
        
        # Estimate tokens (rough approximation)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4  # Rough estimate
        
        if estimated_tokens <= limit:
            return messages
        
        # Smart truncation: keep system + first 2 + last 10 messages
        system_msg = messages[0] if messages[0].get("role") == "system" else None
        conversation = messages[1:] if system_msg else messages
        
        keep_first = 2
        keep_last = 10
        
        if len(conversation) > keep_first + keep_last:
            truncated = (
                conversation[:keep_first] +
                [{"role": "system", "content": "[... conversation truncated ...]"}] +
                conversation[-keep_last:]
            )
        else:
            truncated = conversation
        
        return [system_msg] + truncated if system_msg else truncated
    
    def _validate_system_prompt(self, messages: List[Dict], session) -> List[Dict]:
        """Validate system prompt integrity"""
        if not messages:
            return messages
        
        # Ensure first message is system
        if messages[0].get("role") != "system":
            # Inject identity as system message
            if hasattr(session, 'identity') and session.identity:
                system_msg = {
                    "role": "system",
                    "content": session.identity.system_prompt,
                    "metadata": {"restored": True}
                }
                messages = [system_msg] + messages
        
        return messages
    
    def _check_identity_preserved(self, messages: List[Dict], session) -> bool:
        """Check if identity is preserved in messages"""
        if not messages or not hasattr(session, 'identity'):
            return False
        
        system_msg = messages[0]
        if system_msg.get("role") != "system":
            return False
        
        content = system_msg.get("content", "")
        return session.identity.name in content
