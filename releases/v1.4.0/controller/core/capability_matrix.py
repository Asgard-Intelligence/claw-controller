"""Capability snapshot builder for provider/model pairs."""

from __future__ import annotations

from typing import Dict, Any

from .provider_contracts import ProviderDescriptor, ModelDescriptor


class CapabilityMatrix:
    """Combines provider-level and model-level metadata into route capabilities."""

    @staticmethod
    def build_snapshot(provider: ProviderDescriptor, model: ModelDescriptor) -> Dict[str, Any]:
        return {
            "provider_id": provider.provider_id,
            "model_id": model.model_id,
            "provider_kind": provider.kind.value,
            "dialect_family": provider.dialect_family,
            "context_window": model.context_window,
            "max_output_tokens": model.max_output_tokens,
            "supports_tools": model.supports_tools,
            "supports_vision": model.supports_vision,
            "supports_json_mode": model.supports_json_mode,
            "supports_streaming": model.supports_streaming,
            "supports_system_prompt": model.supports_system_prompt,
            "system_prompt_mode": model.system_prompt_mode,
            "identity_stability": model.identity_stability,
            "timeout_profile": provider.timeout_profile,
            "quality_tier": model.quality_tier,
            "latency_tier": model.latency_tier,
            "cost_tier": model.cost_tier,
            "tags": list(model.tags),
        }
