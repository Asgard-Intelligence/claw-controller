"""
Session Manager - Core of Controller v1.3.0
Manages session state with persistence and identity anchoring
"""

import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass, field
from pathlib import Path
import json

# Optional Redis import
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


@dataclass
class IdentityState:
    """Identity configuration - LOADED ONCE, NEVER DRIFTS"""
    name: str
    loaded_from: str
    loaded_at: datetime
    hash: str
    system_prompt: str
    personality_traits: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    version: str = "1.0"
    _frozen: bool = True


@dataclass
class Message:
    """Message with full metadata"""
    id: str
    role: str  # "system", "user", "assistant", "tool"
    content: str
    timestamp: datetime
    provider: Optional[str] = None
    model: Optional[str] = None
    tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "tokens": self.tokens,
            "metadata": self.metadata,
        }


@dataclass
class RoutingEvent:
    """Record of routing decision"""
    timestamp: datetime
    from_provider: Optional[str]
    to_provider: str
    reason: str
    context_preserved: bool
    token_count: int


@dataclass
class SessionState:
    """Complete session state - SINGLE SOURCE OF TRUTH"""
    session_id: str
    created_at: datetime
    updated_at: datetime
    identity: IdentityState
    messages: List[Message] = field(default_factory=list)
    message_count: int = 0
    total_tokens: int = 0
    current_provider: Optional[str] = None
    pinned_provider: Optional[str] = None
    routing_history: List[RoutingEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "identity": {
                "name": self.identity.name,
                "hash": self.identity.hash,
                "version": self.identity.version,
            },
            "messages": [m.to_dict() for m in self.messages],
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "current_provider": self.current_provider,
            "pinned_provider": self.pinned_provider,
            "routing_history": len(self.routing_history),
        }


class ISessionStore(Protocol):
    """Abstract session storage interface"""
    async def get(self, session_id: str) -> Optional[SessionState]: ...
    async def set(self, session_id: str, state: SessionState, ttl: int = None) -> None: ...
    async def delete(self, session_id: str) -> None: ...


class RedisSessionStore:
    """Production session storage with Redis"""
    
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        self.redis = redis.from_url(redis_url)
        self.default_ttl = default_ttl
        self._local_cache: Dict[str, SessionState] = {}
    
    async def get(self, session_id: str) -> Optional[SessionState]:
        # L1 cache check (in-process)
        if session_id in self._local_cache:
            return self._local_cache[session_id]
        
        # L2 cache check (Redis)
        data = await self.redis.get(f"session:{session_id}")
        if data:
            state = self._deserialize(data)
            self._local_cache[session_id] = state
            return state
        return None
    
    async def set(self, session_id: str, state: SessionState, ttl: int = None) -> None:
        # Update both caches atomically
        self._local_cache[session_id] = state
        await self.redis.setex(
            f"session:{session_id}",
            ttl or self.default_ttl,
            self._serialize(state)
        )
    
    async def delete(self, session_id: str) -> None:
        self._local_cache.pop(session_id, None)
        await self.redis.delete(f"session:{session_id}")
    
    def _serialize(self, state: SessionState) -> bytes:
        return json.dumps(state.to_dict(), default=str).encode()
    
    def _deserialize(self, data: bytes) -> SessionState:
        # Simplified deserialization - full implementation would reconstruct all objects
        dict_data = json.loads(data)
        return SessionState(
            session_id=dict_data["session_id"],
            created_at=datetime.fromisoformat(dict_data["created_at"]),
            updated_at=datetime.fromisoformat(dict_data["updated_at"]),
            identity=IdentityState(
                name=dict_data["identity"]["name"],
                loaded_from="",
                loaded_at=datetime.utcnow(),
                hash=dict_data["identity"]["hash"],
                system_prompt="",
                version=dict_data["identity"].get("version", "1.0"),
            ),
            messages=[],
            message_count=dict_data.get("message_count", 0),
            total_tokens=dict_data.get("total_tokens", 0),
            current_provider=dict_data.get("current_provider"),
            pinned_provider=dict_data.get("pinned_provider"),
        )


class FileSessionStore:
    """Development/testing storage"""
    
    def __init__(self, base_path: str = ".sessions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    async def get(self, session_id: str) -> Optional[SessionState]:
        file_path = self.base_path / f"{session_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
            return self._dict_to_state(data)
        return None
    
    async def set(self, session_id: str, state: SessionState, ttl: int = None) -> None:
        file_path = self.base_path / f"{session_id}.json"
        with open(file_path, 'w') as f:
            json.dump(state.to_dict(), f, indent=2, default=str)
    
    async def delete(self, session_id: str) -> None:
        file_path = self.base_path / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
    
    def _dict_to_state(self, data: Dict) -> SessionState:
        return SessionState(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            identity=IdentityState(
                name=data["identity"]["name"],
                loaded_from="",
                loaded_at=datetime.utcnow(),
                hash=data["identity"]["hash"],
                system_prompt="",
            ),
            messages=[],
            message_count=data.get("message_count", 0),
            total_tokens=data.get("total_tokens", 0),
        )


class IdentityManager:
    """Manages agent identity - LOADED ONCE, NEVER DRIFTS"""
    
    def __init__(self, identity_dir: str = ".identity"):
        self.identity_dir = Path(identity_dir)
        self._cache: Dict[str, IdentityState] = {}
    
    async def load_identity(self, agent_id: str) -> IdentityState:
        """Load identity - cached forever"""
        if agent_id in self._cache:
            return self._cache[agent_id]
        
        identity_file = self.identity_dir / agent_id / "IDENTITY.md"
        
        # Default identity if file doesn't exist
        if not identity_file.exists():
            identity = self._create_default_identity(agent_id)
        else:
            content = identity_file.read_text()
            identity = self._parse_identity(content, str(identity_file))
        
        # Cache forever
        self._cache[agent_id] = identity
        return identity
    
    def _create_default_identity(self, agent_id: str) -> IdentityState:
        """Create default identity"""
        system_prompt = f"""You are an AI assistant. Your name is {agent_id}.
You help users with their tasks while being helpful, harmless, and honest.
Always remember your name is {agent_id}."""
        
        return IdentityState(
            name=agent_id,
            loaded_from="default",
            loaded_at=datetime.utcnow(),
            hash=hashlib.sha256(system_prompt.encode()).hexdigest(),
            system_prompt=system_prompt,
            capabilities=["conversation", "task_assistance"],
        )
    
    def _parse_identity(self, content: str, source: str) -> IdentityState:
        """Parse identity from IDENTITY.md content"""
        # Extract name from content
        name = self._extract_name(content) or "Assistant"
        
        return IdentityState(
            name=name,
            loaded_from=source,
            loaded_at=datetime.utcnow(),
            hash=hashlib.sha256(content.encode()).hexdigest(),
            system_prompt=content,
            personality_traits=self._extract_traits(content),
            constraints=self._extract_constraints(content),
            capabilities=self._extract_capabilities(content),
        )
    
    def _extract_name(self, content: str) -> Optional[str]:
        """Extract name from identity content"""
        import re
        patterns = [
            r'[Nn]ame is["\']?\s+([^"\'\.\n]+)',
            r'[Nn]ame[:\s]+["\']?([^"\'\n]+)["\']?',
            r'[Тт]ебя зовут[:\s]+["\']?([^"\'\n]+)["\']?',
            r'[Тт]вое имя[:\s]+["\']?([^"\'\n]+)["\']?',
            r'[Yy]our name is["\']?\s+([^"\'\.\n]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip().rstrip('.')
        return None
    
    def _extract_traits(self, content: str) -> List[str]:
        return []
    
    def _extract_constraints(self, content: str) -> List[str]:
        return []
    
    def _extract_capabilities(self, content: str) -> List[str]:
        return []
    
    def inject_identity_into_messages(
        self,
        messages: List[Dict],
        identity: IdentityState
    ) -> List[Dict]:
        """Inject identity into messages as system prompt"""
        # Remove existing system messages
        filtered = [m for m in messages if m.get("role") != "system"]
        
        # Add identity as first system message
        identity_message = {
            "role": "system",
            "content": identity.system_prompt,
            "metadata": {
                "identity_version": identity.version,
                "identity_hash": identity.hash,
                "injected_by": "IdentityManager"
            }
        }
        
        return [identity_message] + filtered
    
    def verify_identity_integrity(
        self,
        messages: List[Dict],
        expected_identity: IdentityState
    ) -> bool:
        """Verify identity is present in messages"""
        if not messages:
            return False
        
        system_msg = messages[0]
        if system_msg.get("role") != "system":
            return False
        
        content = system_msg.get("content", "")
        
        # Check name present
        if expected_identity.name not in content:
            return False
        
        return True


class SessionManager:
    """Centralized session management with persistence"""
    
    def __init__(
        self,
        store: ISessionStore,
        identity_manager: IdentityManager,
        session_ttl: int = 86400
    ):
        self.store = store
        self.identity_manager = identity_manager
        self.session_ttl = session_ttl
    
    async def create_session(
        self,
        agent_id: str,
        pinned_provider: Optional[str] = None,
        feature_flags: Optional[Dict[str, bool]] = None
    ) -> SessionState:
        """Create new session with identity loaded"""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Load identity (cached forever)
        identity = await self.identity_manager.load_identity(agent_id)
        
        session = SessionState(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            identity=identity,
            messages=[],
            message_count=0,
            total_tokens=0,
            current_provider=None,
            pinned_provider=pinned_provider,
            routing_history=[],
            metadata={"agent_id": agent_id},
            feature_flags=feature_flags or {}
        )
        
        # Persist immediately
        await self.store.set(session_id, session, ttl=self.session_ttl)
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session from store"""
        return await self.store.get(session_id)
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tokens: Optional[int] = None
    ) -> Optional[SessionState]:
        """Add message to session and persist"""
        session = await self.store.get(session_id)
        if not session:
            return None
        
        # Create message
        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            provider=provider,
            model=model,
            tokens=tokens
        )
        
        # Add message
        session.messages.append(message)
        session.message_count += 1
        session.total_tokens += tokens or 0
        session.updated_at = datetime.utcnow()
        
        # Persist
        await self.store.set(session_id, session, ttl=self.session_ttl)
        
        return session
    
    async def update_provider(
        self,
        session_id: str,
        provider: str
    ) -> Optional[SessionState]:
        """Update current provider"""
        session = await self.store.get(session_id)
        if not session:
            return None
        
        session.current_provider = provider
        session.updated_at = datetime.utcnow()
        
        await self.store.set(session_id, session, ttl=self.session_ttl)
        return session
    
    async def record_routing_event(
        self,
        session_id: str,
        from_provider: Optional[str],
        to_provider: str,
        reason: str,
        context_preserved: bool,
        token_count: int
    ) -> Optional[SessionState]:
        """Record routing event"""
        session = await self.store.get(session_id)
        if not session:
            return None
        
        event = RoutingEvent(
            timestamp=datetime.utcnow(),
            from_provider=from_provider,
            to_provider=to_provider,
            reason=reason,
            context_preserved=context_preserved,
            token_count=token_count
        )
        
        session.routing_history.append(event)
        session.updated_at = datetime.utcnow()
        
        await self.store.set(session_id, session, ttl=self.session_ttl)
        return session
