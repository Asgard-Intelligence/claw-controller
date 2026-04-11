"""
Provider implementations for different LLM services.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional, Sequence

import httpx

from ..models.schemas import (
    ChatCompletionResponse,
    Choice,
    Message,
    NormalizedChatCompletionRequest,
    Role,
    Usage,
)

logger = logging.getLogger(__name__)

ROUTING_TIERS: tuple[str, str, str] = ("premium", "standard", "basic")

_REASONING_META_PATTERNS = [
    r"\bthe user\b",
    r"\buser asks\b",
    r"\buser wants\b",
    r"\bresponse should\b",
    r"\bfinal answer should\b",
    r"\bi should\b",
    r"\bi need to\b",
    r"\blet me\b",
    r"\bplan\b",
    r"\bconstraints?\b",
    r"\banalysis\b",
    r"\breasoning\b",
    r"\bdraft\b",
    r"\bthink(?:ing)?\b",
    r"\bassistant should\b",
    r"\bsystem prompt\b",
    r"\bprompt\b",
    r"\btool\b",
    r"\bpolicy\b",
    r"\boutput language\b",
    r"\bneed answer\b",
    r"\bпользователь\b",
    r"\bинструкц",
    r"\bограничен",
    r"\bплан\b",
    r"\bанализ\b",
    r"\bрассужден",
    r"\bчерновик\b",
    r"\bя должен\b",
    r"\bмне нужно\b",
    r"\bответ должен\b",
    r"\bнужно ответить\b",
    r"\bтребован",
]

_FINAL_MARKER_PATTERNS = [
    r"(?is)<final(?:_answer)?>\s*(.+?)\s*</final(?:_answer)?>",
    r"(?is)(?:^|\n)\s*(?:final answer|final|assistant response|answer|reply)\s*[:：-]\s*(.+)$",
    r"(?is)(?:^|\n)\s*(?:итоговый ответ|финальный ответ|ответ|краткий ответ|ответ пользователю)\s*[:：-]\s*(.+)$",
    r"(?is)(?:^|\n)\s*(?:assistant|ассистент)\s*[:：-]\s*(.+)$",
]


@dataclass
class ProviderCatalog:
    """Backend inventory and resolved routing view for a provider."""

    provider: str
    discovered_models: List[str] = field(default_factory=list)
    resolved_tier_models: Dict[str, str] = field(default_factory=dict)
    default_model: str = ""
    routing_enabled: bool = False
    inventory_status: str = "unknown"
    inventory_issues: List[str] = field(default_factory=list)
    refreshed_at: Optional[int] = None


def _compact_whitespace(value: str) -> str:
    return re.sub(r"[ \t]+", " ", value).strip()


def _flatten_content(value: Any) -> Optional[str]:
    """Flatten text-like content structures into a single string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                part_type = item.get("type")
                text_value = item.get("text")
                if isinstance(text_value, dict):
                    text_value = text_value.get("value")
                if part_type in {"text", "output_text"} and isinstance(text_value, str):
                    parts.append(text_value)
                    continue
                if isinstance(item.get("content"), str):
                    parts.append(item["content"])
                    continue
        return "".join(parts) if parts else None
    if isinstance(value, dict):
        for key in ("text", "value", "content", "output_text", "answer"):
            candidate = value.get(key)
            flattened = _flatten_content(candidate)
            if flattened:
                return flattened
        return None
    return str(value)


def _parse_model_inventory(payload: Any) -> List[str]:
    """Parse model ids from OpenAI-compatible or Anthropic-like inventory payloads."""
    models: List[str] = []

    def _add(candidate: Any) -> None:
        if isinstance(candidate, str):
            normalized = candidate.strip()
            if normalized and normalized not in models:
                models.append(normalized)
        elif isinstance(candidate, dict):
            _add(candidate.get("id"))
            _add(candidate.get("name"))
            _add(candidate.get("model"))

    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            for item in payload["data"]:
                _add(item)
        if isinstance(payload.get("models"), list):
            for item in payload["models"]:
                _add(item)
        if isinstance(payload.get("items"), list):
            for item in payload["items"]:
                _add(item)
    elif isinstance(payload, list):
        for item in payload:
            _add(item)

    return models


def _guess_language(text: str) -> str:
    lower = text.lower()
    if any(ch in lower for ch in ("ї", "є", "і", "ґ")):
        return "uk"
    if re.search(r"[а-яё]", lower):
        return "ru"
    if re.search(r"[a-z]", lower):
        return "en"
    return "en"


def _safe_empty_response_text(request: Optional[NormalizedChatCompletionRequest], sample_text: str = "") -> str:
    reference = sample_text
    if request:
        for message in reversed(request.messages):
            if message.role == Role.USER and message.content:
                reference = message.content
                break
    language = _guess_language(reference)
    if language == "uk":
        return "Вибачте, я не зміг коректно сформувати фінальну відповідь. Будь ласка, повторіть запит."
    if language == "ru":
        return "Извините, я не смог корректно сформировать финальный ответ. Пожалуйста, повторите запрос."
    return "Sorry, I couldn't safely extract the final answer. Please try again."


def _strip_outer_quotes(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'", "“", "”", "«", "»"}:
        return cleaned[1:-1].strip()
    return cleaned


def _contains_meta_reasoning(text: str) -> bool:
    lower = text.lower()
    if any(re.search(pattern, lower) for pattern in _REASONING_META_PATTERNS):
        return True
    if lower.startswith(("- ", "* ", "1.", "2.", "3.")) and ":" in lower:
        return True
    return False


def _cleanup_candidate_text(text: str) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"```.+?```", "", cleaned, flags=re.S)
    cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", cleaned)
    cleaned = re.sub(
        r"(?is)^\s*(?:final answer|final|answer|reply|итоговый ответ|финальный ответ|ответ|ассистент)\s*[:：-]\s*",
        "",
        cleaned,
    )
    cleaned = _strip_outer_quotes(cleaned)
    cleaned = cleaned.strip()
    return cleaned


def _score_user_facing_candidate(candidate: str, source: str) -> float:
    score = 0.0
    text = candidate.strip()
    lower = text.lower()
    words = text.split()

    if not text:
        return -100.0
    if lower in {"text", "текст", "answer", "ответ", "response", "reply"}:
        return -100.0

    if not _contains_meta_reasoning(text):
        score += 3.0
    else:
        score -= 5.0

    if len(words) <= 80:
        score += 1.0
    else:
        score -= 0.5

    if "\n" not in text:
        score += 0.5

    if re.search(r"[.!?…！？。]$", text):
        score += 0.5

    if re.search(r"\b(меня зовут|я |привет|здравствуйте|hello|hi|i am|i'm)\b", lower):
        score += 1.2

    if text.count(":") >= 3:
        score -= 0.5

    position = source.rfind(text)
    if position >= 0 and source:
        score += position / max(len(source), 1)

    return score


def _extract_marked_final_answer(reasoning: str) -> Optional[str]:
    last_match: Optional[str] = None
    for pattern in _FINAL_MARKER_PATTERNS:
        for match in re.finditer(pattern, reasoning):
            candidate = _cleanup_candidate_text(match.group(1))
            if candidate:
                last_match = candidate
    return last_match


def _extract_final_answer_from_reasoning(reasoning: str) -> Optional[str]:
    if not reasoning:
        return None

    normalized = reasoning.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return None

    explicit = _extract_marked_final_answer(normalized)
    if explicit and not _contains_meta_reasoning(explicit):
        return explicit

    candidates: List[str] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]

    for value in paragraphs + lines:
        cleaned = _cleanup_candidate_text(value)
        if cleaned:
            candidates.append(cleaned)
            sentences = re.split(r"(?<=[.!?…！？。])\s+", cleaned)
            if len(sentences) > 1:
                tail = " ".join(sentences[-2:]).strip()
                tail = _cleanup_candidate_text(tail)
                if tail:
                    candidates.append(tail)

    unique_candidates: List[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)

    if not unique_candidates:
        return None

    scored = [(_score_user_facing_candidate(candidate, normalized), candidate) for candidate in unique_candidates]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_candidate = scored[0]
    if best_score >= 1.0:
        return best_candidate

    if not _contains_meta_reasoning(normalized):
        fallback = _cleanup_candidate_text(normalized)
        return fallback or None

    return None


def _extract_message_answer(
    choice_data: Dict[str, Any],
    message_data: Dict[str, Any],
    response: Dict[str, Any],
    request: Optional[NormalizedChatCompletionRequest] = None,
) -> Optional[str]:
    """Extract only the final user-facing assistant answer."""
    if message_data.get("tool_calls"):
        return _flatten_content(message_data.get("content"))

    direct_candidates = [
        _flatten_content(message_data.get("content")),
        _flatten_content(message_data.get("output")),
        _flatten_content(message_data.get("answer")),
        _flatten_content(message_data.get("final")),
        _flatten_content(message_data.get("completion")),
        _flatten_content(choice_data.get("text")),
        _flatten_content(response.get("output_text")),
    ]
    for candidate in direct_candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    reasoning_candidates = [
        _flatten_content(message_data.get("reasoning")),
        _flatten_content(message_data.get("reasoning_content")),
        _flatten_content(message_data.get("analysis")),
        _flatten_content(message_data.get("thought")),
        _flatten_content(choice_data.get("reasoning")),
        _flatten_content(choice_data.get("analysis")),
    ]
    for reasoning in reasoning_candidates:
        if isinstance(reasoning, str) and reasoning.strip():
            extracted = _extract_final_answer_from_reasoning(reasoning)
            if extracted:
                return extracted

    return _safe_empty_response_text(request, sample_text=_flatten_content(message_data.get("reasoning")) or "")


def _sanitize_openai_stream_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Remove reasoning-only delta fragments and emit only transport-safe user-facing chunks."""
    if not isinstance(payload, dict):
        return None

    sanitized = dict(payload)
    raw_choices = sanitized.get("choices")
    if not isinstance(raw_choices, list):
        return sanitized

    cleaned_choices: List[Dict[str, Any]] = []
    for raw_choice in raw_choices:
        if not isinstance(raw_choice, dict):
            continue
        choice = dict(raw_choice)
        delta = choice.get("delta")
        if isinstance(delta, dict):
            delta = dict(delta)
            for key in ("reasoning", "reasoning_content", "analysis", "thinking", "thought"):
                delta.pop(key, None)
            content_value = delta.get("content")
            if isinstance(content_value, list):
                flattened = _flatten_content(content_value)
                if flattened is not None:
                    delta["content"] = flattened
                else:
                    delta.pop("content", None)
            choice["delta"] = delta
            if not delta and choice.get("finish_reason") is None:
                continue

        message = choice.get("message")
        if isinstance(message, dict):
            message = dict(message)
            for key in ("reasoning", "reasoning_content", "analysis", "thinking", "thought"):
                message.pop(key, None)
            choice["message"] = message

        cleaned_choices.append(choice)

    sanitized["choices"] = cleaned_choices
    if not cleaned_choices and not sanitized.get("usage"):
        return None
    return sanitized


def _transport_safe_chunk(chunk: Any) -> str | bytes:
    """Normalize provider stream chunks to strings/bytes only."""
    if isinstance(chunk, (str, bytes)):
        return chunk
    if isinstance(chunk, tuple) and chunk:
        head = chunk[0]
        if isinstance(head, (str, bytes)):
            return head
    if isinstance(chunk, dict):
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    raise TypeError(f"Unsupported streaming chunk type: {type(chunk)!r}")


def _model_score(model_id: str) -> float:
    """Heuristic routing score derived only from discovered backend model ids."""
    text = model_id.lower()
    score = 0.0

    for multiplier, number in re.findall(r"(\d+)x(\d+(?:\.\d+)?)b", text):
        score = max(score, float(multiplier) * float(number))

    for pattern in (
        r"(?<!q)(?<!fp)(?<!int)(\d+(?:\.\d+)?)b",
        r"(?:^|[-_/])e(\d+(?:\.\d+)?)b(?:$|[-_/])",
    ):
        for match in re.findall(pattern, text):
            try:
                score = max(score, float(match))
            except ValueError:
                continue

    keyword_boosts = {
        "opus": 120.0,
        "max": 100.0,
        "ultra": 95.0,
        "large": 90.0,
        "sonnet": 70.0,
        "medium": 55.0,
        "haiku": 30.0,
        "mini": 20.0,
        "small": 18.0,
        "tiny": 10.0,
        "nano": 5.0,
    }
    for keyword, boost in keyword_boosts.items():
        if keyword in text:
            score = max(score, boost)

    if score == 0.0:
        score = max(1.0, min(float(len(text)), 40.0))

    return score


class Provider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.api_key = config.get("api_key")
        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.default_model = str(config.get("default_model", "") or "")
        self.tier_models = {
            tier: str(value or "").strip()
            for tier, value in dict(config.get("tier_models", {})).items()
            if str(value or "").strip()
        }
        self.discovery_strict = bool(config.get("model_discovery_strict", False))
        self.discovery_refresh_seconds = int(config.get("model_discovery_refresh_seconds", 300) or 0)
        self.routing_require_discovered_models = bool(
            config.get("routing_require_discovered_models", True)
        )
        self.client = httpx.AsyncClient(timeout=60.0)

        self.discovered_models: List[str] = []
        self.resolved_tier_models: Dict[str, str] = {}
        self.inventory_status: str = "unknown"
        self.inventory_issues: List[str] = []
        self.inventory_refreshed_at: Optional[int] = None
        self.routing_default_model: str = self.default_model

    def validate_request(self, request: NormalizedChatCompletionRequest) -> None:
        """Raise ValueError when the request uses unsupported features."""
        return None

    @abstractmethod
    async def chat_completion(
        self,
        request: NormalizedChatCompletionRequest,
        target_model: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Execute chat completion."""

    @abstractmethod
    async def chat_completion_stream(
        self,
        request: NormalizedChatCompletionRequest,
        target_model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion."""

    @abstractmethod
    def format_request(
        self,
        request: NormalizedChatCompletionRequest,
        model: str,
    ) -> Dict[str, Any]:
        """Format request for provider API."""

    @abstractmethod
    def parse_response(
        self,
        response: Dict[str, Any],
        model: str,
        request: Optional[NormalizedChatCompletionRequest] = None,
    ) -> ChatCompletionResponse:
        """Parse provider response to standard format."""

    @abstractmethod
    async def discover_models(self) -> List[str]:
        """Discover backend model ids from the provider inventory endpoint."""

    def routing_catalog(self) -> ProviderCatalog:
        return ProviderCatalog(
            provider=self.name,
            discovered_models=list(self.discovered_models),
            resolved_tier_models=dict(self.resolved_tier_models),
            default_model=self.routing_default_model,
            routing_enabled=bool(self.resolved_tier_models),
            inventory_status=self.inventory_status,
            inventory_issues=list(self.inventory_issues),
            refreshed_at=self.inventory_refreshed_at,
        )

    def _infer_tier_models(self, models: Sequence[str]) -> Dict[str, str]:
        unique = [model for model in models if model]
        if not unique:
            return {}
        ordered = sorted(unique, key=lambda item: (_model_score(item), item.lower()), reverse=True)
        premium = ordered[0]
        basic = ordered[-1]
        if len(ordered) == 1:
            standard = ordered[0]
        elif self.default_model and self.default_model in ordered:
            standard = self.default_model
        else:
            standard = ordered[len(ordered) // 2]
        return {
            "premium": premium,
            "standard": standard,
            "basic": basic,
        }

    def _apply_model_inventory(self, discovered_models: Sequence[str]) -> None:
        unique_models: List[str] = []
        for candidate in discovered_models:
            normalized = str(candidate or "").strip()
            if normalized and normalized not in unique_models:
                unique_models.append(normalized)

        issues: List[str] = []
        resolved: Dict[str, str] = {}

        for tier in ROUTING_TIERS:
            configured = self.tier_models.get(tier, "")
            if configured:
                if configured in unique_models:
                    resolved[tier] = configured
                else:
                    issues.append(
                        f"configured {tier} model '{configured}' is not present in discovered backend inventory"
                    )

        if self.default_model:
            if self.default_model in unique_models:
                resolved.setdefault("standard", self.default_model)
                resolved.setdefault("basic", self.default_model)
            else:
                issues.append(
                    f"configured default model '{self.default_model}' is not present in discovered backend inventory"
                )

        inferred = self._infer_tier_models(unique_models)
        for tier in ROUTING_TIERS:
            inferred_candidate = inferred.get(tier)
            if inferred_candidate:
                resolved.setdefault(tier, inferred_candidate)

        self.discovered_models = unique_models
        self.resolved_tier_models = resolved
        self.routing_default_model = (
            resolved.get("standard")
            or resolved.get("basic")
            or resolved.get("premium")
            or (unique_models[0] if unique_models else "")
        )
        self.inventory_issues = issues
        self.inventory_refreshed_at = int(time.time())

        if not unique_models:
            self.inventory_status = "unhealthy"
            if "no models discovered from backend inventory" not in self.inventory_issues:
                self.inventory_issues.append("no models discovered from backend inventory")
        elif not resolved:
            self.inventory_status = "degraded"
            self.inventory_issues.append("no routable tier mapping could be resolved from discovered models")
        elif issues:
            self.inventory_status = "degraded"
        else:
            self.inventory_status = "healthy"

    async def refresh_model_inventory(self, *, force: bool = False) -> ProviderCatalog:
        now = int(time.time())
        if (
            not force
            and self.inventory_refreshed_at is not None
            and self.discovery_refresh_seconds > 0
            and now - self.inventory_refreshed_at < self.discovery_refresh_seconds
        ):
            return self.routing_catalog()

        previous_models = list(self.discovered_models)
        previous_mapping = dict(self.resolved_tier_models)
        previous_default = self.routing_default_model
        previous_status = self.inventory_status
        previous_issues = list(self.inventory_issues)

        try:
            discovered_models = await self.discover_models()
            self._apply_model_inventory(discovered_models)
        except Exception as exc:
            if previous_models:
                self.discovered_models = previous_models
                self.resolved_tier_models = previous_mapping
                self.routing_default_model = previous_default
                self.inventory_status = "degraded"
                self.inventory_issues = list(previous_issues)
                self.inventory_issues.append(f"model discovery refresh failed; using last known inventory: {exc}")
                self.inventory_refreshed_at = now
            else:
                self.discovered_models = []
                self.resolved_tier_models = {}
                self.routing_default_model = ""
                self.inventory_status = "unhealthy"
                self.inventory_issues = [f"model discovery failed: {exc}"]
                self.inventory_refreshed_at = now

        if self.discovery_strict and self.inventory_status != "healthy":
            raise RuntimeError(
                f"provider {self.name} model discovery validation failed: {'; '.join(self.inventory_issues)}"
            )

        return self.routing_catalog()

    def resolve_target_model(
        self,
        *,
        requested_model: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> str:
        requested = str(requested_model or "").strip()
        if self.routing_require_discovered_models and not self.discovered_models:
            raise ValueError(
                f"provider {self.name} has no discovered backend models available for routing"
            )

        if requested:
            if self.discovered_models and requested not in self.discovered_models:
                raise ValueError(
                    f"model '{requested}' is not present on backend for provider {self.name}"
                )
            return requested

        if tier:
            resolved = self.resolved_tier_models.get(tier)
            if resolved:
                return resolved

        if self.routing_default_model:
            return self.routing_default_model

        if self.discovered_models:
            return self.discovered_models[0]

        raise ValueError(f"provider {self.name} has no routable model")

    def supports_model(self, model: str) -> bool:
        candidate = str(model or "").strip()
        return bool(candidate and self.discovered_models and candidate in self.discovered_models)

    async def health_check(self) -> Dict[str, Any]:
        """Check provider health using model inventory refresh state."""
        catalog = await self.refresh_model_inventory(force=False)
        return {
            "status": catalog.inventory_status,
            "provider": self.name,
            "latency_ms": None,
            "model_count": len(catalog.discovered_models),
            "inventory_issues": catalog.inventory_issues,
        }

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()


class OpenAIProvider(Provider):
    """OpenAI-compatible API provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("openai", config)
        if not self.api_key:
            raise ValueError("OpenAI-compatible API key is required")

    async def discover_models(self) -> List[str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = await self.client.get(f"{self.base_url}/models", headers=headers, timeout=15.0)
        response.raise_for_status()
        return _parse_model_inventory(response.json())

    def format_request(
        self,
        request: NormalizedChatCompletionRequest,
        model: str,
    ) -> Dict[str, Any]:
        """Format request for an OpenAI-compatible API."""
        messages: List[Dict[str, Any]] = []
        for message in request.messages:
            formatted: Dict[str, Any] = {"role": message.role.value}
            content_value: Any = message.content
            if message.role == Role.ASSISTANT and message.tool_calls and content_value == "":
                content_value = None
            formatted["content"] = content_value
            if message.name:
                formatted["name"] = message.name
            if message.tool_calls is not None:
                formatted["tool_calls"] = message.tool_calls
            if message.tool_call_id:
                formatted["tool_call_id"] = message.tool_call_id
            if message.refusal:
                formatted["refusal"] = message.refusal
            messages.append(formatted)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
            "stream": request.stream,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = request.stop
        if request.user:
            payload["user"] = request.user
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.tools is not None:
            payload["tools"] = request.tools
        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        if request.stream_options is not None:
            payload["stream_options"] = request.stream_options
        if request.metadata is not None:
            payload["metadata"] = request.metadata

        for key, value in request.extra_body.items():
            if key not in payload and value is not None:
                payload[key] = value

        return payload

    def parse_response(
        self,
        response: Dict[str, Any],
        model: str,
        request: Optional[NormalizedChatCompletionRequest] = None,
    ) -> ChatCompletionResponse:
        """Parse OpenAI-compatible response without exposing raw reasoning."""
        choices: List[Choice] = []
        for choice_data in response.get("choices", []):
            message_data = choice_data.get("message", {}) if isinstance(choice_data, dict) else {}
            role_value = message_data.get("role", "assistant")
            try:
                role = Role(role_value)
            except ValueError:
                role = Role.ASSISTANT

            content = _extract_message_answer(choice_data, message_data, response, request=request)
            message = Message(
                role=role,
                content=content,
                name=message_data.get("name"),
                tool_calls=message_data.get("tool_calls"),
                tool_call_id=message_data.get("tool_call_id"),
                refusal=message_data.get("refusal"),
            )
            choices.append(
                Choice(
                    index=choice_data.get("index", 0),
                    message=message,
                    finish_reason=choice_data.get("finish_reason"),
                    logprobs=choice_data.get("logprobs"),
                )
            )

        if not choices:
            choices.append(
                Choice(
                    index=0,
                    message=Message(
                        role=Role.ASSISTANT,
                        content=_safe_empty_response_text(request),
                    ),
                    finish_reason="stop",
                )
            )

        usage_data = response.get("usage", {})
        total_tokens = usage_data.get("total_tokens")
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        return ChatCompletionResponse(
            id=response.get("id", ""),
            created=response.get("created", int(time.time())),
            model=response.get("model", model),
            choices=choices,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            system_fingerprint=response.get("system_fingerprint"),
        )

    async def chat_completion(
        self,
        request: NormalizedChatCompletionRequest,
        target_model: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Execute chat completion via an OpenAI-compatible API."""
        model = self.resolve_target_model(requested_model=target_model)
        payload = self.format_request(request, model)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start_time = time.time()
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            logger.info(
                "OpenAI-compatible request completed in %dms using model=%s",
                int((time.time() - start_time) * 1000),
                model,
            )
            return self.parse_response(data, model, request=request)
        except httpx.HTTPStatusError as exc:
            logger.error(
                "OpenAI-compatible API error: %s - %s",
                exc.response.status_code,
                exc.response.text,
            )
            raise
        except Exception as exc:
            logger.error("OpenAI-compatible request failed: %s", exc)
            raise

    async def chat_completion_stream(
        self,
        request: NormalizedChatCompletionRequest,
        target_model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion via an OpenAI-compatible API."""
        model = self.resolve_target_model(requested_model=target_model)
        payload = self.format_request(request, model)
        payload["stream"] = True

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        payload_data = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    sanitized = _sanitize_openai_stream_payload(payload_data)
                    if sanitized is None:
                        continue
                    yield f"data: {json.dumps(sanitized, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.error("OpenAI-compatible streaming error: %s", exc)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI-compatible API health using discovered model inventory."""
        start = time.time()
        catalog = await self.refresh_model_inventory(force=False)
        latency = int((time.time() - start) * 1000)
        return {
            "status": catalog.inventory_status,
            "provider": self.name,
            "latency_ms": latency,
            "model_count": len(catalog.discovered_models),
            "inventory_issues": catalog.inventory_issues,
        }


class AnthropicProvider(Provider):
    """Anthropic API provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("anthropic", config)
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        self.version = config.get("version", "2023-06-01")

    async def discover_models(self) -> List[str]:
        response = await self.client.get(
            f"{self.base_url}/v1/models",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.version,
            },
            timeout=15.0,
        )
        response.raise_for_status()
        return _parse_model_inventory(response.json())

    def validate_request(self, request: NormalizedChatCompletionRequest) -> None:
        """Reject unsupported OpenAI-only request features for Anthropic."""
        if request.tools or request.tool_choice is not None or request.parallel_tool_calls is not None:
            raise ValueError(
                "anthropic provider does not support OpenAI tool payloads in controller 1.2.0-rc"
            )
        if request.response_format is not None or request.stream_options is not None:
            raise ValueError(
                "anthropic provider does not support OpenAI response_format/stream_options in controller 1.2.0-rc"
            )
        if request.extra_body:
            raise ValueError(
                "anthropic provider cannot safely passthrough unknown OpenAI-specific request fields"
            )
        if any(message.role == Role.TOOL or message.tool_calls for message in request.messages):
            raise ValueError(
                "anthropic provider does not support tool call transcripts in controller 1.2.0-rc"
            )

    def format_request(
        self,
        request: NormalizedChatCompletionRequest,
        model: str,
    ) -> Dict[str, Any]:
        """Format request for the Anthropic messages API."""
        self.validate_request(request)

        system_messages: List[str] = []
        messages: List[Dict[str, Any]] = []

        for message in request.messages:
            if message.role == Role.SYSTEM:
                if message.content:
                    system_messages.append(message.content)
                continue
            messages.append(
                {
                    "role": "user" if message.role == Role.USER else "assistant",
                    "content": message.content,
                }
            )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
        }

        if system_messages:
            payload["system"] = "\n\n".join(system_messages)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop is not None:
            payload["stop_sequences"] = (
                [request.stop] if isinstance(request.stop, str) else request.stop
            )
        return payload

    def parse_response(
        self,
        response: Dict[str, Any],
        model: str,
        request: Optional[NormalizedChatCompletionRequest] = None,
    ) -> ChatCompletionResponse:
        """Parse Anthropic response to OpenAI-compatible format."""
        content = ""
        for block in response.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        if not content:
            content = _safe_empty_response_text(request)

        usage_data = response.get("usage", {})
        prompt_tokens = usage_data.get("input_tokens", 0)
        completion_tokens = usage_data.get("output_tokens", 0)

        return ChatCompletionResponse(
            id=response.get("id", ""),
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role=Role.ASSISTANT, content=content),
                    finish_reason=response.get("stop_reason", "stop"),
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def chat_completion(
        self,
        request: NormalizedChatCompletionRequest,
        target_model: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Execute chat completion via Anthropic."""
        model = self.resolve_target_model(requested_model=target_model)
        payload = self.format_request(request, model)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "Content-Type": "application/json",
        }

        start_time = time.time()
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            logger.info(
                "Anthropic request completed in %dms using model=%s",
                int((time.time() - start_time) * 1000),
                model,
            )
            return self.parse_response(data, model, request=request)
        except httpx.HTTPStatusError as exc:
            logger.error("Anthropic API error: %s - %s", exc.response.status_code, exc.response.text)
            raise
        except Exception as exc:
            logger.error("Anthropic request failed: %s", exc)
            raise

    async def chat_completion_stream(
        self,
        request: NormalizedChatCompletionRequest,
        target_model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion via Anthropic."""
        model = self.resolve_target_model(requested_model=target_model)
        payload = self.format_request(request, model)
        payload["stream"] = True

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "Content-Type": "application/json",
        }

        current_event: Optional[str] = None
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event: "):
                        current_event = line[7:]
                        continue
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]
                    if data == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break

                    try:
                        payload_data = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if current_event == "content_block_delta":
                        text = payload_data.get("delta", {}).get("text", "")
                        chunk = {
                            "id": payload_data.get("message", {}).get("id", ""),
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif current_event == "message_stop":
                        yield "data: [DONE]\n\n"
                        break
        except Exception as exc:
            logger.error("Anthropic streaming error: %s", exc)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check Anthropic API health using discovered model inventory."""
        start = time.time()
        catalog = await self.refresh_model_inventory(force=False)
        latency = int((time.time() - start) * 1000)
        return {
            "status": catalog.inventory_status,
            "provider": self.name,
            "latency_ms": latency,
            "model_count": len(catalog.discovered_models),
            "inventory_issues": catalog.inventory_issues,
        }


class FallbackProvider:
    """Provider wrapper that manages fallback order and retry-on-provider-failure."""

    def __init__(self, providers: Dict[str, Provider]):
        self.providers = providers
        self.failure_counts: Dict[str, int] = {name: 0 for name in providers.keys()}
        self.max_failures = 3

    def _provider_order(
        self,
        preferred_provider: Optional[str] = None,
        allowed_providers: Optional[List[str]] = None,
    ) -> List[str]:
        names = list(allowed_providers or self.providers.keys())
        names = [name for name in names if name in self.providers]
        if not names:
            return []

        if preferred_provider in names:
            remaining = [name for name in names if name != preferred_provider]
            remaining.sort(key=lambda name: self.failure_counts.get(name, 0))
            return [preferred_provider] + remaining

        names.sort(key=lambda name: self.failure_counts.get(name, 0))
        return names

    def _target_model_for_provider(
        self,
        provider_name: str,
        *,
        target_model: Optional[str] = None,
        target_models: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        if target_models and provider_name in target_models:
            return target_models[provider_name]
        return target_model

    async def chat_completion(
        self,
        request: NormalizedChatCompletionRequest,
        preferred_provider: Optional[str] = None,
        allowed_providers: Optional[List[str]] = None,
        target_model: Optional[str] = None,
        target_models: Optional[Dict[str, str]] = None,
    ) -> tuple[ChatCompletionResponse, str]:
        """Execute completion with provider fallback."""
        last_error: Optional[Exception] = None
        for provider_name in self._provider_order(preferred_provider, allowed_providers):
            if self.failure_counts.get(provider_name, 0) >= self.max_failures:
                continue
            provider = self.providers[provider_name]
            provider_target_model = self._target_model_for_provider(
                provider_name,
                target_model=target_model,
                target_models=target_models,
            )
            try:
                response = await provider.chat_completion(request, target_model=provider_target_model)
                self.failure_counts[provider_name] = max(
                    0, self.failure_counts.get(provider_name, 0) - 1
                )
                return response, provider_name
            except Exception as exc:
                logger.warning("Provider %s failed: %s", provider_name, exc)
                self.failure_counts[provider_name] = self.failure_counts.get(provider_name, 0) + 1
                last_error = exc
        raise Exception(f"All providers failed. Last error: {last_error}")

    async def chat_completion_stream(
        self,
        request: NormalizedChatCompletionRequest,
        preferred_provider: Optional[str] = None,
        allowed_providers: Optional[List[str]] = None,
        target_model: Optional[str] = None,
        target_models: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Execute streaming completion with provider fallback.

        Contract: only transport-safe strings or bytes are yielded.
        """
        last_error: Optional[Exception] = None
        for provider_name in self._provider_order(preferred_provider, allowed_providers):
            if self.failure_counts.get(provider_name, 0) >= self.max_failures:
                continue
            provider = self.providers[provider_name]
            provider_target_model = self._target_model_for_provider(
                provider_name,
                target_model=target_model,
                target_models=target_models,
            )
            emitted = False
            try:
                logger.info(
                    "Streaming via provider=%s model=%s",
                    provider_name,
                    provider_target_model or provider.routing_default_model or provider.default_model,
                )
                async for chunk in provider.chat_completion_stream(request, target_model=provider_target_model):
                    emitted = True
                    yield _transport_safe_chunk(chunk)
                self.failure_counts[provider_name] = max(
                    0, self.failure_counts.get(provider_name, 0) - 1
                )
                return
            except Exception as exc:
                logger.warning("Provider %s streaming failed: %s", provider_name, exc)
                self.failure_counts[provider_name] = self.failure_counts.get(provider_name, 0) + 1
                last_error = exc
                if emitted:
                    raise Exception(
                        f"Streaming failed after partial response from provider {provider_name}: {exc}"
                    ) from exc
        raise Exception(f"All providers failed for streaming. Last error: {last_error}")

    async def health_check(self) -> Dict[str, Any]:
        """Check all provider health."""
        return {name: await provider.health_check() for name, provider in self.providers.items()}

    async def close(self) -> None:
        """Close all underlying providers."""
        for provider in self.providers.values():
            await provider.close()
