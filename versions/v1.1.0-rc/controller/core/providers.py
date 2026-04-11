"""
Provider implementations for different LLM services.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

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


def _flatten_content(value: Any) -> Optional[str]:
    """Flatten text-only OpenAI-style content structures into a string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(value)


class Provider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.api_key = config.get("api_key")
        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.default_model = config.get("default_model", "")
        self.tier_models = config.get("tier_models", {})
        self.client = httpx.AsyncClient(timeout=60.0)

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
    def parse_response(self, response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        """Parse provider response to standard format."""

    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        return {"status": "unknown", "provider": self.name}

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()


class OpenAIProvider(Provider):
    """OpenAI-compatible API provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("openai", config)
        if not self.api_key:
            raise ValueError("OpenAI-compatible API key is required")

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

    def parse_response(self, response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        """Parse OpenAI-compatible response."""
        choices: List[Choice] = []
        for choice_data in response.get("choices", []):
            message_data = choice_data.get("message", {})
            role_value = message_data.get("role", "assistant")
            try:
                role = Role(role_value)
            except ValueError:
                role = Role.ASSISTANT

            message = Message(
                role=role,
                content=_flatten_content(message_data.get("content")),
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
        model = target_model or request.model or self.default_model
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
            return self.parse_response(data, model)
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
        model = target_model or request.model or self.default_model
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
                        json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    yield f"data: {data}\n\n"
        except Exception as exc:
            logger.error("OpenAI-compatible streaming error: %s", exc)
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI-compatible API health using the configured base URL."""
        try:
            start = time.time()
            headers: Dict[str, str] = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = await self.client.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10.0,
            )
            latency = int((time.time() - start) * 1000)
            if 200 <= response.status_code < 300:
                return {
                    "status": "healthy",
                    "provider": self.name,
                    "latency_ms": latency,
                }
            return {
                "status": "degraded",
                "provider": self.name,
                "latency_ms": latency,
                "error": f"Status {response.status_code}",
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "provider": self.name,
                "error": str(exc),
            }


class AnthropicProvider(Provider):
    """Anthropic API provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("anthropic", config)
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        self.version = config.get("version", "2023-06-01")

    def validate_request(self, request: NormalizedChatCompletionRequest) -> None:
        """Reject unsupported OpenAI-only request features for Anthropic."""
        if request.tools or request.tool_choice is not None or request.parallel_tool_calls is not None:
            raise ValueError(
                "anthropic provider does not support OpenAI tool payloads in controller 1.1.0"
            )
        if request.response_format is not None or request.stream_options is not None:
            raise ValueError(
                "anthropic provider does not support OpenAI response_format/stream_options in controller 1.1.0"
            )
        if request.extra_body:
            raise ValueError(
                "anthropic provider cannot safely passthrough unknown OpenAI-specific request fields"
            )
        if any(message.role == Role.TOOL or message.tool_calls for message in request.messages):
            raise ValueError(
                "anthropic provider does not support tool call transcripts in controller 1.1.0"
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

    def parse_response(self, response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        """Parse Anthropic response to OpenAI-compatible format."""
        content = ""
        for block in response.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

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
        model = target_model or self.default_model
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
            return self.parse_response(data, model)
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
        model = target_model or self.default_model
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
        """Check Anthropic API health."""
        try:
            start = time.time()
            response = await self.client.get(
                f"{self.base_url}/v1/models",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.version,
                },
                timeout=10.0,
            )
            latency = int((time.time() - start) * 1000)
            if response.status_code in {200, 404}:
                return {
                    "status": "healthy",
                    "provider": self.name,
                    "latency_ms": latency,
                }
            return {
                "status": "degraded",
                "provider": self.name,
                "latency_ms": latency,
                "error": f"Status {response.status_code}",
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "provider": self.name,
                "error": str(exc),
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

    async def chat_completion(
        self,
        request: NormalizedChatCompletionRequest,
        preferred_provider: Optional[str] = None,
        allowed_providers: Optional[List[str]] = None,
        target_model: Optional[str] = None,
    ) -> tuple[ChatCompletionResponse, str]:
        """Execute completion with provider fallback."""
        last_error: Optional[Exception] = None
        for provider_name in self._provider_order(preferred_provider, allowed_providers):
            if self.failure_counts.get(provider_name, 0) >= self.max_failures:
                continue
            provider = self.providers[provider_name]
            try:
                response = await provider.chat_completion(request, target_model=target_model)
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
    ) -> AsyncGenerator[str, None]:
        """Execute streaming completion with provider fallback.

        Contract: only transport-safe strings are yielded.
        """
        last_error: Optional[Exception] = None
        for provider_name in self._provider_order(preferred_provider, allowed_providers):
            if self.failure_counts.get(provider_name, 0) >= self.max_failures:
                continue
            provider = self.providers[provider_name]
            emitted = False
            try:
                logger.info("Streaming via provider=%s model=%s", provider_name, target_model or provider.default_model)
                async for chunk in provider.chat_completion_stream(request, target_model=target_model):
                    emitted = True
                    yield chunk
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
