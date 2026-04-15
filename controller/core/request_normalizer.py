"""Request normalization into canonical internal model for v1.4."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from .provider_contracts import CanonicalMessage, CanonicalRequest, ProviderKind, RoutingRequirements
from .session_manager import SessionState
from .token_estimator import TokenEstimator


class RequestNormalizer:
    def __init__(self, token_estimator: TokenEstimator):
        self.token_estimator = token_estimator

    def normalize(
        self,
        body: Dict[str, Any],
        session: SessionState,
        agent_id: str,
        session_id: str,
        provider_hint: Optional[str] = None,
        model_hint: Optional[str] = None,
        routing_mode: str = "balanced",
        locality_preference: Optional[str] = None,
    ) -> CanonicalRequest:
        request_id = str(uuid.uuid4())

        incoming_messages = self._normalize_messages(body.get("messages", []))
        full_messages = self._build_canonical_history(session, incoming_messages)

        return CanonicalRequest(
            request_id=request_id,
            session_id=session_id,
            agent_id=agent_id,
            model=body.get("model", "controller"),
            messages=full_messages,
            stream=bool(body.get("stream", False)),
            temperature=body.get("temperature"),
            max_tokens=body.get("max_tokens"),
            tools=body.get("tools"),
            response_format=body.get("response_format"),
            raw=body,
        )

    def build_requirements(
        self,
        canonical_request: CanonicalRequest,
        provider_hint: Optional[str] = None,
        model_hint: Optional[str] = None,
        routing_mode: str = "balanced",
        locality_preference: Optional[str] = None,
    ) -> RoutingRequirements:
        requires_json = bool(
            canonical_request.response_format
            and canonical_request.response_format.get("type") in {"json_object", "json_schema"}
        )

        preference = None
        if locality_preference == "local":
            preference = ProviderKind.LOCAL
        elif locality_preference == "cloud":
            preference = ProviderKind.CLOUD

        model_from_request = canonical_request.model if canonical_request.model != "controller" else None

        force_plain_local = provider_hint == "ollama"

        return RoutingRequirements(
            estimated_prompt_tokens=self.token_estimator.estimate_messages(canonical_request.messages),
            requires_tools=False if force_plain_local else bool(canonical_request.tools),
            requires_vision=False if force_plain_local else self._requires_vision(canonical_request.messages),
            requires_json_mode=False if force_plain_local else requires_json,
            requires_streaming=canonical_request.stream,
            provider_hint=provider_hint,
            model_hint=model_hint or model_from_request,
            routing_mode=routing_mode,
            identity_sensitive=self._is_identity_sensitive(canonical_request.messages),
            locality_preference=preference,
        )

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[CanonicalMessage]:
        normalized: List[CanonicalMessage] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                # OpenAI multimodal style blocks -> stringify for canonical storage.
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(str(block.get("text", "")))
                content = "\n".join([x for x in text_parts if x])
            normalized.append(
                CanonicalMessage(
                    role=msg.get("role", "user"),
                    content=str(content),
                    name=msg.get("name"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                    metadata=dict(msg.get("metadata", {})),
                )
            )
        return normalized

    def _build_canonical_history(
        self,
        session: SessionState,
        incoming_messages: List[CanonicalMessage],
    ) -> List[CanonicalMessage]:
        messages: List[CanonicalMessage] = []

        # Canonical identity anchor is always first message.
        messages.append(
            CanonicalMessage(
                role="system",
                content=session.identity.system_prompt,
                metadata={
                    "identity_hash": session.identity.hash,
                    "identity_version": session.identity.version,
                    "source": "session_identity",
                },
            )
        )

        for msg in session.messages:
            messages.append(
                CanonicalMessage(
                    role=msg.role,
                    content=msg.content,
                    metadata=dict(msg.metadata),
                )
            )

        # External system messages are ignored to preserve identity invariants.
        messages.extend([m for m in incoming_messages if m.role != "system"])
        return messages

    def _is_identity_sensitive(self, messages: List[CanonicalMessage]) -> bool:
        if not messages:
            return False
        last = messages[-1].content.lower()
        keywords = [
            "your name",
            "my name",
            "тебя зовут",
            "как тебя зовут",
            "как меня зовут",
            "identity",
        ]
        return any(keyword in last for keyword in keywords)

    def _requires_vision(self, messages: List[CanonicalMessage]) -> bool:
        for msg in messages:
            if msg.metadata.get("has_image"):
                return True
        return False
