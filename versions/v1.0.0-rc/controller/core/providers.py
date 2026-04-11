"""
Provider implementations for different LLM services.
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator, List
from datetime import datetime

import httpx

from ..models.schemas import (
    ChatCompletionRequest, ChatCompletionResponse, Message, Choice, Usage,
    Role
)

logger = logging.getLogger(__name__)


class Provider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "")
        self.default_model = config.get("default_model", "")
        self.client = httpx.AsyncClient(timeout=60.0)
    
    @abstractmethod
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute chat completion."""
        pass
    
    @abstractmethod
    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion."""
        pass
    
    @abstractmethod
    def format_request(self, request: ChatCompletionRequest, model: str) -> Dict[str, Any]:
        """Format request for provider API."""
        pass
    
    @abstractmethod
    def parse_response(self, response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        """Parse provider response to standard format."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        return {"status": "unknown", "provider": self.name}
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class OpenAIProvider(Provider):
    """OpenAI API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("openai", config)
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
    
    def format_request(self, request: ChatCompletionRequest, model: str) -> Dict[str, Any]:
        """Format request for OpenAI API."""
        messages = []
        for m in request.messages:
            msg = {"role": m.role.value, "content": m.content}
            if m.name:
                msg["name"] = m.name
            messages.append(msg)
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
            "stream": request.stream,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
        }
        
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        if request.user:
            payload["user"] = request.user
        
        return payload
    
    def parse_response(self, response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        """Parse OpenAI response."""
        choices = []
        for c in response.get("choices", []):
            msg_data = c.get("message", {})
            message = Message(
                role=Role(msg_data.get("role", "assistant")),
                content=msg_data.get("content", ""),
            )
            choices.append(Choice(
                index=c.get("index", 0),
                message=message,
                finish_reason=c.get("finish_reason")
            ))
        
        usage_data = response.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        
        return ChatCompletionResponse(
            id=response.get("id", ""),
            created=response.get("created", int(time.time())),
            model=response.get("model", model),
            choices=choices,
            usage=usage,
            system_fingerprint=response.get("system_fingerprint")
        )
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute chat completion via OpenAI API."""
        model = request.force_provider or self.default_model
        payload = self.format_request(request, model)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        start_time = time.time()
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"OpenAI request completed in {(time.time() - start_time)*1000:.0f}ms")
            
            return self.parse_response(data, model)
        
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            raise
    
    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion via OpenAI API."""
        model = request.force_provider or self.default_model
        payload = self.format_request(request, model)
        payload["stream"] = True
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            chunk = json.loads(data)
                            yield f"data: {json.dumps(chunk)}\n\n"
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenAI API health."""
        try:
            start = time.time()
            response = await self.client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0
            )
            latency = int((time.time() - start) * 1000)
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "provider": self.name,
                    "latency_ms": latency
                }
            else:
                return {
                    "status": "degraded",
                    "provider": self.name,
                    "latency_ms": latency,
                    "error": f"Status {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.name,
                "error": str(e)
            }


class AnthropicProvider(Provider):
    """Anthropic API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("anthropic", config)
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        self.version = config.get("version", "2023-06-01")
    
    def format_request(self, request: ChatCompletionRequest, model: str) -> Dict[str, Any]:
        """Format request for Anthropic API."""
        # Extract system message
        system_message = ""
        messages = []
        
        for m in request.messages:
            if m.role == Role.SYSTEM:
                system_message = m.content
            else:
                messages.append({
                    "role": "user" if m.role == Role.USER else "assistant",
                    "content": m.content
                })
        
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
        }
        
        if system_message:
            payload["system"] = system_message
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        
        return payload
    
    def parse_response(self, response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        """Parse Anthropic response to OpenAI format."""
        content = ""
        for block in response.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
        
        message = Message(
            role=Role.ASSISTANT,
            content=content,
        )
        
        choice = Choice(
            index=0,
            message=message,
            finish_reason=response.get("stop_reason", "stop")
        )
        
        usage_data = response.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )
        
        return ChatCompletionResponse(
            id=response.get("id", ""),
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=usage,
        )
    
    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Execute chat completion via Anthropic API."""
        model = request.force_provider or self.default_model
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
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Anthropic request completed in {(time.time() - start_time)*1000:.0f}ms")
            
            return self.parse_response(data, model)
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Anthropic API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            raise
    
    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion via Anthropic API."""
        model = request.force_provider or self.default_model
        payload = self.format_request(request, model)
        payload["stream"] = True
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "Content-Type": "application/json",
        }
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        try:
                            chunk = json.loads(data)
                            # Convert to OpenAI-like format
                            openai_chunk = {
                                "id": chunk.get("message", {}).get("id", ""),
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": chunk.get("delta", {}).get("text", "")},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n"
                        except json.JSONDecodeError:
                            continue
                    elif line == "event: message_stop":
                        yield "data: [DONE]\n\n"
                        break
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Anthropic API health."""
        try:
            start = time.time()
            # Anthropic doesn't have a simple models endpoint, so we do a minimal request
            response = await self.client.get(
                f"{self.base_url}/v1/models",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": self.version,
                },
                timeout=10.0
            )
            latency = int((time.time() - start) * 1000)
            
            if response.status_code in [200, 404]:  # 404 is OK, means API is reachable
                return {
                    "status": "healthy",
                    "provider": self.name,
                    "latency_ms": latency
                }
            else:
                return {
                    "status": "degraded",
                    "provider": self.name,
                    "latency_ms": latency,
                    "error": f"Status {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.name,
                "error": str(e)
            }


class FallbackProvider:
    """Provider that manages fallback chain."""
    
    def __init__(self, providers: Dict[str, Provider]):
        self.providers = providers
        self.failure_counts: Dict[str, int] = {name: 0 for name in providers.keys()}
        self.max_failures = 3
    
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        preferred_provider: Optional[str] = None
    ) -> tuple[ChatCompletionResponse, str]:
        """Execute with fallback."""
        
        # Build provider order
        provider_order = list(self.providers.keys())
        if preferred_provider and preferred_provider in provider_order:
            provider_order.remove(preferred_provider)
            provider_order.insert(0, preferred_provider)
        
        # Sort by failure count
        provider_order.sort(key=lambda p: self.failure_counts.get(p, 0))
        
        last_error = None
        for provider_name in provider_order:
            if self.failure_counts.get(provider_name, 0) >= self.max_failures:
                continue
            
            provider = self.providers[provider_name]
            try:
                response = await provider.chat_completion(request)
                self.failure_counts[provider_name] = max(0, self.failure_counts.get(provider_name, 0) - 1)
                return response, provider_name
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                self.failure_counts[provider_name] = self.failure_counts.get(provider_name, 0) + 1
                last_error = e
        
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        preferred_provider: Optional[str] = None
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Execute streaming with fallback."""
        
        provider_order = list(self.providers.keys())
        if preferred_provider and preferred_provider in provider_order:
            provider_order.remove(preferred_provider)
            provider_order.insert(0, preferred_provider)
        
        provider_order.sort(key=lambda p: self.failure_counts.get(p, 0))
        
        for provider_name in provider_order:
            if self.failure_counts.get(provider_name, 0) >= self.max_failures:
                continue
            
            provider = self.providers[provider_name]
            try:
                async for chunk in provider.chat_completion_stream(request):
                    yield chunk, provider_name
                return
            except Exception as e:
                logger.warning(f"Provider {provider_name} streaming failed: {e}")
                self.failure_counts[provider_name] = self.failure_counts.get(provider_name, 0) + 1
        
        raise Exception("All providers failed for streaming")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check all providers health."""
        results = {}
        for name, provider in self.providers.items():
            results[name] = await provider.health_check()
        return results
    
    async def close(self):
        """Close all providers."""
        for provider in self.providers.values():
            await provider.close()
