"""Registry bootstrap from legacy env and optional route config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .endpoint_normalizer import build_url, normalize_provider_base_url
from .provider_contracts import ApiDialect, ModelDescriptor, ProviderDescriptor, ProviderKind
from .provider_registry_v140 import ProviderRegistryV140


def _enum_dialects(values: Iterable[str]) -> List[ApiDialect]:
    result: List[ApiDialect] = []
    for value in values:
        try:
            result.append(ApiDialect(value))
        except ValueError:
            continue
    return result


def bootstrap_provider_registry(settings: Any) -> ProviderRegistryV140:
    registry = ProviderRegistryV140()

    route_config_path = getattr(settings, "ROUTE_REGISTRY_PATH", None)
    if route_config_path:
        path = Path(route_config_path)
        if path.exists():
            _bootstrap_from_file(registry, path)
            return registry

    _bootstrap_from_legacy_env(registry, settings)
    return registry


def _bootstrap_from_file(registry: ProviderRegistryV140, path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))

    for provider in data.get("providers", []):
        pd = ProviderDescriptor(
            provider_id=provider["provider_id"],
            label=provider.get("label", provider["provider_id"]),
            kind=ProviderKind(provider.get("kind", ProviderKind.CLOUD.value)),
            dialect_family=provider.get("dialect_family", "openai_compatible"),
            base_url=normalize_provider_base_url(provider["base_url"]),
            validation_base_url=normalize_provider_base_url(provider.get("validation_base_url", provider["base_url"])),
            health_url=provider.get("health_url") or build_url(provider["base_url"], "/models"),
            credential_env=provider.get("credential_env"),
            default_model_id=provider["default_model_id"],
            enabled=provider.get("enabled", True),
            timeout_profile=provider.get("timeout_profile", "default"),
            supports_model_catalog=provider.get("supports_model_catalog", True),
            supports_runtime_probe=provider.get("supports_runtime_probe", True),
            supports_container_reachability_probe=provider.get("supports_container_reachability_probe", False),
        )
        registry.register_provider(pd)

    for model in data.get("models", []):
        allowed = _enum_dialects(model.get("allowed_api_dialects", [ApiDialect.OPENAI_CHAT_COMPLETIONS.value]))
        preferred = _enum_dialects(model.get("preferred_api_dialects", [])) or allowed[:1]
        md = ModelDescriptor(
            provider_id=model["provider_id"],
            model_id=model["model_id"],
            aliases=model.get("aliases", []),
            family=model.get("family", "general"),
            context_window=int(model.get("context_window", 8192)),
            max_output_tokens=int(model.get("max_output_tokens", 4096)),
            supports_tools=bool(model.get("supports_tools", False)),
            supports_vision=bool(model.get("supports_vision", False)),
            supports_json_mode=bool(model.get("supports_json_mode", False)),
            supports_streaming=bool(model.get("supports_streaming", True)),
            supports_system_prompt=bool(model.get("supports_system_prompt", True)),
            system_prompt_mode=model.get("system_prompt_mode", "native"),
            preferred_api_dialects=preferred,
            allowed_api_dialects=allowed,
            quality_tier=model.get("quality_tier", "standard"),
            latency_tier=model.get("latency_tier", "standard"),
            cost_tier=model.get("cost_tier", "standard"),
            identity_stability=model.get("identity_stability", "standard"),
            tags=model.get("tags", []),
        )
        registry.register_model(md)


def _bootstrap_from_legacy_env(registry: ProviderRegistryV140, settings: Any) -> None:
    openai_base = normalize_provider_base_url(getattr(settings, "OPENAI_BASE_URL", "https://api.openai.com/v1"))
    anthropic_base = normalize_provider_base_url(getattr(settings, "ANTHROPIC_BASE_URL", "https://api.anthropic.com"))
    ollama_runtime_base = normalize_provider_base_url(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_validation_base = normalize_provider_base_url(
        getattr(settings, "OLLAMA_VALIDATION_BASE_URL", ollama_runtime_base)
    )
    ollama_health_endpoint = getattr(settings, "OLLAMA_HEALTH_ENDPOINT", "/api/tags")

    registry.register_provider(
        ProviderDescriptor(
            provider_id="openai",
            label="OpenAI",
            kind=ProviderKind.CLOUD,
            dialect_family="openai_compatible",
            base_url=openai_base,
            validation_base_url=openai_base,
            health_url=build_url(openai_base, "/models"),
            credential_env="OPENAI_API_KEY",
            default_model_id=getattr(settings, "OPENAI_MODEL_STANDARD", "gpt-4"),
            enabled=True,
            timeout_profile="cloud_default",
            supports_model_catalog=True,
            supports_runtime_probe=True,
            supports_container_reachability_probe=False,
        )
    )

    registry.register_provider(
        ProviderDescriptor(
            provider_id="anthropic",
            label="Anthropic",
            kind=ProviderKind.CLOUD,
            dialect_family="anthropic",
            base_url=anthropic_base,
            validation_base_url=anthropic_base,
            health_url=build_url(anthropic_base, "/v1/models"),
            credential_env="ANTHROPIC_API_KEY",
            default_model_id=getattr(settings, "ANTHROPIC_MODEL_STANDARD", "claude-3-sonnet-20240229"),
            enabled=True,
            timeout_profile="cloud_default",
            supports_model_catalog=True,
            supports_runtime_probe=True,
            supports_container_reachability_probe=False,
        )
    )

    registry.register_provider(
        ProviderDescriptor(
            provider_id="ollama",
            label="Ollama",
            kind=ProviderKind.LOCAL,
            dialect_family="ollama_native",
            base_url=ollama_runtime_base,
            validation_base_url=ollama_validation_base,
            health_url=build_url(ollama_validation_base, ollama_health_endpoint),
            credential_env=None,
            default_model_id=getattr(settings, "OLLAMA_MODEL", "gemma4-e4b-q4-local"),
            enabled=True,
            timeout_profile="local_default",
            supports_model_catalog=True,
            supports_runtime_probe=True,
            supports_container_reachability_probe=True,
        )
    )

    # Model catalog (legacy env-compatible defaults)
    registry.register_model(
        ModelDescriptor(
            provider_id="openai",
            model_id=getattr(settings, "OPENAI_MODEL_STANDARD", "gpt-4"),
            aliases=["openai-standard", "gpt4"],
            family="gpt",
            context_window=8192,
            max_output_tokens=4096,
            supports_tools=True,
            supports_vision=False,
            supports_json_mode=True,
            supports_streaming=True,
            supports_system_prompt=True,
            system_prompt_mode="native",
            preferred_api_dialects=[ApiDialect.OPENAI_CHAT_COMPLETIONS],
            allowed_api_dialects=[ApiDialect.OPENAI_RESPONSES, ApiDialect.OPENAI_CHAT_COMPLETIONS],
            quality_tier="standard",
            latency_tier="standard",
            cost_tier="medium",
            identity_stability="high",
            tags=["cloud", "openai"],
        )
    )

    registry.register_model(
        ModelDescriptor(
            provider_id="openai",
            model_id=getattr(settings, "OPENAI_MODEL_PREMIUM", "gpt-4-turbo"),
            aliases=["openai-premium"],
            family="gpt",
            context_window=128000,
            max_output_tokens=4096,
            supports_tools=True,
            supports_vision=True,
            supports_json_mode=True,
            supports_streaming=True,
            supports_system_prompt=True,
            system_prompt_mode="native",
            preferred_api_dialects=[ApiDialect.OPENAI_RESPONSES],
            allowed_api_dialects=[ApiDialect.OPENAI_RESPONSES, ApiDialect.OPENAI_CHAT_COMPLETIONS],
            quality_tier="premium",
            latency_tier="medium",
            cost_tier="high",
            identity_stability="high",
            tags=["cloud", "openai", "premium"],
        )
    )

    registry.register_model(
        ModelDescriptor(
            provider_id="anthropic",
            model_id=getattr(settings, "ANTHROPIC_MODEL_STANDARD", "claude-3-sonnet-20240229"),
            aliases=["anthropic-standard", "claude-sonnet"],
            family="claude",
            context_window=200000,
            max_output_tokens=4096,
            supports_tools=True,
            supports_vision=True,
            supports_json_mode=True,
            supports_streaming=True,
            supports_system_prompt=True,
            system_prompt_mode="native",
            preferred_api_dialects=[ApiDialect.ANTHROPIC_MESSAGES],
            allowed_api_dialects=[ApiDialect.ANTHROPIC_MESSAGES],
            quality_tier="standard",
            latency_tier="standard",
            cost_tier="medium",
            identity_stability="high",
            tags=["cloud", "anthropic"],
        )
    )

    registry.register_model(
        ModelDescriptor(
            provider_id="anthropic",
            model_id=getattr(settings, "ANTHROPIC_MODEL_PREMIUM", "claude-3-opus-20240229"),
            aliases=["anthropic-premium", "claude-opus"],
            family="claude",
            context_window=200000,
            max_output_tokens=4096,
            supports_tools=True,
            supports_vision=True,
            supports_json_mode=True,
            supports_streaming=True,
            supports_system_prompt=True,
            system_prompt_mode="native",
            preferred_api_dialects=[ApiDialect.ANTHROPIC_MESSAGES],
            allowed_api_dialects=[ApiDialect.ANTHROPIC_MESSAGES],
            quality_tier="premium",
            latency_tier="medium",
            cost_tier="high",
            identity_stability="high",
            tags=["cloud", "anthropic", "premium"],
        )
    )

    registry.register_model(
        ModelDescriptor(
            provider_id="ollama",
            model_id=getattr(settings, "OLLAMA_MODEL", "gemma4-e4b-q4-local"),
            aliases=["local-default", "ollama-default"],
            family="local",
            context_window=32768,
            max_output_tokens=2048,
            supports_tools=False,
            supports_vision=False,
            supports_json_mode=False,
            supports_streaming=True,
            supports_system_prompt=True,
            system_prompt_mode="native",
            preferred_api_dialects=[ApiDialect.OLLAMA_CHAT],
            allowed_api_dialects=[ApiDialect.OLLAMA_CHAT, ApiDialect.OPENAI_CHAT_COMPLETIONS],
            quality_tier="standard",
            latency_tier="low",
            cost_tier="low",
            identity_stability="medium",
            tags=["local", "ollama"],
        )
    )
