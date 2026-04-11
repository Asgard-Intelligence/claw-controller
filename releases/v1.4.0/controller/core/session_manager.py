"""Session manager for Controller v1.4 with lossless persistence and migration."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


@dataclass
class IdentityState:
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "loaded_from": self.loaded_from,
            "loaded_at": self.loaded_at.isoformat(),
            "hash": self.hash,
            "system_prompt": self.system_prompt,
            "personality_traits": list(self.personality_traits),
            "constraints": list(self.constraints),
            "capabilities": list(self.capabilities),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityState":
        loaded_at_raw = data.get("loaded_at")
        loaded_at = datetime.fromisoformat(loaded_at_raw) if loaded_at_raw else datetime.utcnow()
        system_prompt = data.get("system_prompt", "")
        return cls(
            name=data.get("name", "Assistant"),
            loaded_from=data.get("loaded_from", "legacy"),
            loaded_at=loaded_at,
            hash=data.get("hash") or hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            system_prompt=system_prompt,
            personality_traits=list(data.get("personality_traits", [])),
            constraints=list(data.get("constraints", [])),
            capabilities=list(data.get("capabilities", [])),
            version=data.get("version", "1.0"),
        )


@dataclass
class Message:
    id: str
    role: str
    content: str
    timestamp: datetime
    provider: Optional[str] = None
    model: Optional[str] = None
    tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        timestamp_raw = data.get("timestamp")
        timestamp = datetime.fromisoformat(timestamp_raw) if timestamp_raw else datetime.utcnow()
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=timestamp,
            provider=data.get("provider"),
            model=data.get("model"),
            tokens=data.get("tokens"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RoutingEvent:
    timestamp: datetime
    from_provider: Optional[str]
    to_provider: Optional[str]
    reason: str
    context_preserved: bool
    token_count: int
    from_route: Optional[str] = None
    to_route: Optional[str] = None
    api_dialect: Optional[str] = None
    failure_kind: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "from_provider": self.from_provider,
            "to_provider": self.to_provider,
            "reason": self.reason,
            "context_preserved": self.context_preserved,
            "token_count": self.token_count,
            "from_route": self.from_route,
            "to_route": self.to_route,
            "api_dialect": self.api_dialect,
            "failure_kind": self.failure_kind,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingEvent":
        timestamp_raw = data.get("timestamp")
        timestamp = datetime.fromisoformat(timestamp_raw) if timestamp_raw else datetime.utcnow()
        return cls(
            timestamp=timestamp,
            from_provider=data.get("from_provider"),
            to_provider=data.get("to_provider"),
            reason=data.get("reason", ""),
            context_preserved=bool(data.get("context_preserved", True)),
            token_count=int(data.get("token_count", 0)),
            from_route=data.get("from_route"),
            to_route=data.get("to_route"),
            api_dialect=data.get("api_dialect"),
            failure_kind=data.get("failure_kind"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SessionState:
    session_id: str
    created_at: datetime
    updated_at: datetime
    identity: IdentityState
    messages: List[Message] = field(default_factory=list)
    message_count: int = 0
    total_tokens: int = 0
    current_provider: Optional[str] = None
    pinned_provider: Optional[str] = None
    current_route: Optional[str] = None
    pinned_route: Optional[str] = None
    last_successful_route: Optional[str] = None
    routing_history: List[RoutingEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    identity_hash_at_session_start: Optional[str] = None
    context_transfer_meta: Dict[str, Any] = field(default_factory=dict)
    session_version: str = "1.4"

    def _derive_provider_from_route(self, route: Optional[str]) -> Optional[str]:
        if not route:
            return None
        return route.split(":", 1)[0]

    def sync_compat_fields(self) -> None:
        derived_current = self._derive_provider_from_route(self.current_route)
        derived_pinned = self._derive_provider_from_route(self.pinned_route)
        if derived_current:
            self.current_provider = derived_current
        if derived_pinned:
            self.pinned_provider = derived_pinned

    def to_dict(self) -> Dict[str, Any]:
        self.sync_compat_fields()
        return {
            "session_version": self.session_version,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "identity": self.identity.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
            "message_count": self.message_count,
            "total_tokens": self.total_tokens,
            "current_provider": self.current_provider,
            "pinned_provider": self.pinned_provider,
            "current_route": self.current_route,
            "pinned_route": self.pinned_route,
            "last_successful_route": self.last_successful_route,
            "routing_history": [e.to_dict() for e in self.routing_history],
            "metadata": self.metadata,
            "feature_flags": self.feature_flags,
            "identity_hash_at_session_start": self.identity_hash_at_session_start,
            "context_transfer_meta": self.context_transfer_meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        created_at = datetime.fromisoformat(data["created_at"])
        updated_at = datetime.fromisoformat(data.get("updated_at", data["created_at"]))

        identity_raw = dict(data.get("identity", {}))
        if "system_prompt" not in identity_raw:
            # v1.3 compatibility: system prompt was omitted from serialized identity.
            identity_raw["system_prompt"] = data.get("metadata", {}).get("identity_system_prompt", "")
        identity = IdentityState.from_dict(identity_raw)

        messages = [Message.from_dict(item) for item in data.get("messages", [])]

        routing_history_raw = data.get("routing_history", [])
        routing_history: List[RoutingEvent] = []
        if isinstance(routing_history_raw, list):
            routing_history = [RoutingEvent.from_dict(item) for item in routing_history_raw]

        state = cls(
            session_id=data["session_id"],
            created_at=created_at,
            updated_at=updated_at,
            identity=identity,
            messages=messages,
            message_count=int(data.get("message_count", len(messages))),
            total_tokens=int(data.get("total_tokens", 0)),
            current_provider=data.get("current_provider"),
            pinned_provider=data.get("pinned_provider"),
            current_route=data.get("current_route"),
            pinned_route=data.get("pinned_route"),
            last_successful_route=data.get("last_successful_route"),
            routing_history=routing_history,
            metadata=dict(data.get("metadata", {})),
            feature_flags=dict(data.get("feature_flags", {})),
            identity_hash_at_session_start=data.get("identity_hash_at_session_start") or identity.hash,
            context_transfer_meta=dict(data.get("context_transfer_meta", {})),
            session_version=str(data.get("session_version", "1.3")),
        )

        # Migrate legacy routing history count-only representation.
        if isinstance(routing_history_raw, int):
            state.metadata.setdefault("legacy_routing_history_count", routing_history_raw)

        if not state.identity.system_prompt and state.identity.name:
            state.identity.system_prompt = (
                f"You are an AI assistant. Your name is {state.identity.name}. "
                f"Always remember your name is {state.identity.name}."
            )

        state.sync_compat_fields()
        state.session_version = "1.4"
        return state


class ISessionStore(Protocol):
    async def get(self, session_id: str) -> Optional[SessionState]: ...

    async def set(self, session_id: str, state: SessionState, ttl: int = None) -> None: ...

    async def delete(self, session_id: str) -> None: ...


class RedisSessionStore:
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        self.redis = redis.from_url(redis_url)
        self.default_ttl = default_ttl
        self._local_cache: Dict[str, SessionState] = {}

    async def get(self, session_id: str) -> Optional[SessionState]:
        if session_id in self._local_cache:
            return self._local_cache[session_id]

        data = await self.redis.get(f"session:{session_id}")
        if not data:
            return None

        state = self._deserialize(data)
        self._local_cache[session_id] = state
        return state

    async def set(self, session_id: str, state: SessionState, ttl: int = None) -> None:
        self._local_cache[session_id] = state
        await self.redis.setex(
            f"session:{session_id}",
            ttl or self.default_ttl,
            self._serialize(state),
        )

    async def delete(self, session_id: str) -> None:
        self._local_cache.pop(session_id, None)
        await self.redis.delete(f"session:{session_id}")

    def _serialize(self, state: SessionState) -> bytes:
        return json.dumps(state.to_dict(), ensure_ascii=False).encode("utf-8")

    def _deserialize(self, data: bytes) -> SessionState:
        payload = json.loads(data)
        return SessionState.from_dict(payload)


class FileSessionStore:
    def __init__(self, base_path: str = ".sessions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

    async def get(self, session_id: str) -> Optional[SessionState]:
        file_path = self.base_path / f"{session_id}.json"
        if not file_path.exists():
            return None
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return SessionState.from_dict(payload)

    async def set(self, session_id: str, state: SessionState, ttl: int = None) -> None:
        file_path = self.base_path / f"{session_id}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)

    async def delete(self, session_id: str) -> None:
        file_path = self.base_path / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()


class IdentityManager:
    def __init__(self, identity_dir: str = ".identity"):
        self.identity_dir = Path(identity_dir)
        self._cache: Dict[str, IdentityState] = {}

    async def load_identity(self, agent_id: str) -> IdentityState:
        if agent_id in self._cache:
            return self._cache[agent_id]

        identity_file = self.identity_dir / agent_id / "IDENTITY.md"
        if not identity_file.exists():
            identity = self._create_default_identity(agent_id)
        else:
            content = identity_file.read_text(encoding="utf-8")
            identity = self._parse_identity(content, str(identity_file))

        self._cache[agent_id] = identity
        return identity

    def _create_default_identity(self, agent_id: str) -> IdentityState:
        system_prompt = (
            f"You are an AI assistant. Your name is {agent_id}. "
            "You help users with their tasks while being helpful, harmless, and honest. "
            f"Always remember your name is {agent_id}."
        )
        return IdentityState(
            name=agent_id,
            loaded_from="default",
            loaded_at=datetime.utcnow(),
            hash=hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            system_prompt=system_prompt,
            capabilities=["conversation", "task_assistance"],
        )

    def _parse_identity(self, content: str, source: str) -> IdentityState:
        name = self._extract_name(content) or "Assistant"
        return IdentityState(
            name=name,
            loaded_from=source,
            loaded_at=datetime.utcnow(),
            hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            system_prompt=content,
            personality_traits=self._extract_traits(content),
            constraints=self._extract_constraints(content),
            capabilities=self._extract_capabilities(content),
        )

    def _extract_name(self, content: str) -> Optional[str]:
        import re

        patterns = [
            r"[Nn]ame is[\"']?\s+([^\"'\.\n]+)",
            r"[Nn]ame[:\s]+[\"']?([^\"'\n]+)[\"']?",
            r"[Тт]ебя зовут[:\s]+[\"']?([^\"'\n]+)[\"']?",
            r"[Тт]вое имя[:\s]+[\"']?([^\"'\n]+)[\"']?",
            r"[Yy]our name is[\"']?\s+([^\"'\.\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip().rstrip(".")
        return None

    def _extract_traits(self, content: str) -> List[str]:
        return []

    def _extract_constraints(self, content: str) -> List[str]:
        return []

    def _extract_capabilities(self, content: str) -> List[str]:
        return []

    def inject_identity_into_messages(self, messages: List[Dict[str, Any]], identity: IdentityState) -> List[Dict[str, Any]]:
        filtered = [m for m in messages if m.get("role") != "system"]
        identity_message = {
            "role": "system",
            "content": identity.system_prompt,
            "metadata": {
                "identity_version": identity.version,
                "identity_hash": identity.hash,
                "injected_by": "IdentityManager",
            },
        }
        return [identity_message] + filtered

    def verify_identity_integrity(self, messages: List[Dict[str, Any]], expected_identity: IdentityState) -> bool:
        if not messages:
            return False
        system_msg = messages[0]
        if system_msg.get("role") != "system":
            return False
        content = system_msg.get("content", "")
        return expected_identity.name in content


class SessionManager:
    def __init__(self, store: ISessionStore, identity_manager: IdentityManager, session_ttl: int = 86400):
        self.store = store
        self.identity_manager = identity_manager
        self.session_ttl = session_ttl

    async def create_session(
        self,
        agent_id: str,
        pinned_provider: Optional[str] = None,
        pinned_route: Optional[str] = None,
        feature_flags: Optional[Dict[str, bool]] = None,
    ) -> SessionState:
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
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
            current_route=None,
            pinned_route=pinned_route,
            last_successful_route=None,
            routing_history=[],
            metadata={"agent_id": agent_id},
            feature_flags=feature_flags or {},
            identity_hash_at_session_start=identity.hash,
            context_transfer_meta={},
            session_version="1.4",
        )

        await self.store.set(session_id, session, ttl=self.session_ttl)
        return session

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        session = await self.store.get(session_id)
        if not session:
            return None

        # Transparent migration write-back for legacy sessions.
        if session.session_version != "1.4":
            session.session_version = "1.4"
            session.updated_at = datetime.utcnow()
            await self.store.set(session_id, session, ttl=self.session_ttl)

        return session

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SessionState]:
        session = await self.store.get(session_id)
        if not session:
            return None

        message = Message(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            provider=provider,
            model=model,
            tokens=tokens,
            metadata=metadata or {},
        )

        session.messages.append(message)
        session.message_count = len(session.messages)
        session.total_tokens += tokens or 0
        session.updated_at = datetime.utcnow()

        await self.store.set(session_id, session, ttl=self.session_ttl)
        return session

    async def update_route(
        self,
        session_id: str,
        route_key: str,
        provider_id: Optional[str] = None,
        mark_successful: bool = False,
    ) -> Optional[SessionState]:
        session = await self.store.get(session_id)
        if not session:
            return None

        session.current_route = route_key
        session.current_provider = provider_id or route_key.split(":", 1)[0]
        if mark_successful:
            session.last_successful_route = route_key
        session.updated_at = datetime.utcnow()

        await self.store.set(session_id, session, ttl=self.session_ttl)
        return session

    async def update_provider(self, session_id: str, provider: str) -> Optional[SessionState]:
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
        to_provider: Optional[str],
        reason: str,
        context_preserved: bool,
        token_count: int,
        from_route: Optional[str] = None,
        to_route: Optional[str] = None,
        api_dialect: Optional[str] = None,
        failure_kind: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SessionState]:
        session = await self.store.get(session_id)
        if not session:
            return None

        event = RoutingEvent(
            timestamp=datetime.utcnow(),
            from_provider=from_provider,
            to_provider=to_provider,
            reason=reason,
            context_preserved=context_preserved,
            token_count=token_count,
            from_route=from_route,
            to_route=to_route,
            api_dialect=api_dialect,
            failure_kind=failure_kind,
            metadata=metadata or {},
        )

        session.routing_history.append(event)
        session.updated_at = datetime.utcnow()

        await self.store.set(session_id, session, ttl=self.session_ttl)
        return session

    async def set_context_transfer_meta(self, session_id: str, transfer_meta: Dict[str, Any]) -> Optional[SessionState]:
        session = await self.store.get(session_id)
        if not session:
            return None

        session.context_transfer_meta = transfer_meta
        session.updated_at = datetime.utcnow()
        await self.store.set(session_id, session, ttl=self.session_ttl)
        return session
