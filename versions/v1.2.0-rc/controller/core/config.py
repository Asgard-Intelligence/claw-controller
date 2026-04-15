"""
Configuration management using pydantic-settings.
Externalized config via environment variables.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Annotated, Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Controller configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server settings
    CONTROLLER_HOST: str = Field(default="0.0.0.0", description="Server bind host")
    CONTROLLER_PORT: int = Field(default=8080, ge=1, le=65535, description="Server port")
    CONTROLLER_API_KEY: str = Field(description="API key for Controller authentication")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format: json or text")

    # Provider API keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI-compatible API key")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")

    # Provider settings
    DEFAULT_PROVIDER: str = Field(default="openai", description="Default provider")
    FALLBACK_PROVIDERS: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["openai", "anthropic"],
        description="Fallback chain. Accepts comma-separated or JSON array syntax.",
    )

    # OpenAI-compatible settings
    OPENAI_BASE_URL: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL",
    )
    OPENAI_DEFAULT_MODEL: Optional[str] = Field(
        default=None,
        description="Preferred default OpenAI-compatible model. Must exist in discovered backend inventory.",
    )
    OPENAI_MODEL_PREMIUM: Optional[str] = Field(
        default=None,
        description="Tier override for premium routing on the OpenAI-compatible provider",
    )
    OPENAI_MODEL_STANDARD: Optional[str] = Field(
        default=None,
        description="Tier override for standard routing on the OpenAI-compatible provider",
    )
    OPENAI_MODEL_BASIC: Optional[str] = Field(
        default=None,
        description="Tier override for basic routing on the OpenAI-compatible provider",
    )

    # Anthropic settings
    ANTHROPIC_BASE_URL: str = Field(
        default="https://api.anthropic.com",
        description="Anthropic API base URL",
    )
    ANTHROPIC_DEFAULT_MODEL: Optional[str] = Field(
        default=None,
        description="Preferred default Anthropic model. Must exist in discovered backend inventory.",
    )
    ANTHROPIC_MODEL_PREMIUM: Optional[str] = Field(
        default=None,
        description="Tier override for premium routing on Anthropic",
    )
    ANTHROPIC_MODEL_STANDARD: Optional[str] = Field(
        default=None,
        description="Tier override for standard routing on Anthropic",
    )
    ANTHROPIC_MODEL_BASIC: Optional[str] = Field(
        default=None,
        description="Tier override for basic routing on Anthropic",
    )
    ANTHROPIC_VERSION: str = Field(default="2023-06-01", description="Anthropic API version")

    # Backend model discovery and routing safety
    ROUTING_REQUIRE_DISCOVERED_MODELS: bool = Field(
        default=True,
        description="Require controller routing to use only models discovered from backend inventory",
    )
    MODEL_DISCOVERY_STRICT: bool = Field(
        default=False,
        description="Fail provider initialization when configured model mappings do not match discovered inventory",
    )
    MODEL_DISCOVERY_REFRESH_SECONDS: int = Field(
        default=300,
        ge=0,
        description="How long to cache discovered backend models before refreshing",
    )

    # Intelligence settings
    CONFIDENCE_THRESHOLD_HIGH: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="High confidence threshold",
    )
    CONFIDENCE_THRESHOLD_MEDIUM: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Medium confidence threshold",
    )
    SAFETY_RISK_THRESHOLD: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Safety risk threshold",
    )
    ENABLE_PII_DETECTION: bool = Field(default=True, description="Enable PII detection")
    ENABLE_JAILBREAK_DETECTION: bool = Field(default=True, description="Enable jailbreak detection")

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=60,
        ge=1,
        description="Requests per minute limit",
    )

    # Circuit breaker
    CIRCUIT_BREAKER_ENABLED: bool = Field(default=True, description="Enable circuit breaker")
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        default=5,
        ge=1,
        description="Failure threshold",
    )
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(
        default=30,
        ge=1,
        description="Recovery timeout in seconds",
    )

    # Cache settings
    CACHE_ENABLED: bool = Field(default=False, description="Enable response caching")
    CACHE_TTL_SECONDS: int = Field(default=300, ge=1, description="Cache TTL in seconds")

    # Redis settings (optional)
    REDIS_URL: Optional[str] = Field(default=None, description="Redis URL for caching")

    # Metrics
    METRICS_ENABLED: bool = Field(default=False, description="Enable Prometheus metrics")
    METRICS_PORT: int = Field(default=9090, ge=1, le=65535, description="Metrics server port")

    # Tracing (optional)
    TRACING_ENABLED: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    JAEGER_ENDPOINT: Optional[str] = Field(default=None, description="Jaeger endpoint")

    # Runtime paths
    RUNTIME_DIR: str = Field(default="./runtime", description="Runtime directory")
    LOGS_DIR: str = Field(default="./runtime/logs", description="Logs directory")
    CACHE_DIR: str = Field(default="./runtime/cache", description="Cache directory")
    PID_FILE: str = Field(default="./runtime/controller.pid", description="PID file path")

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        normalized = value.upper()
        if normalized not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return normalized

    @field_validator("CONTROLLER_API_KEY")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        if not value or len(value) < 16:
            raise ValueError("CONTROLLER_API_KEY must be at least 16 characters")
        return value

    @field_validator("DEFAULT_PROVIDER")
    @classmethod
    def normalize_default_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("DEFAULT_PROVIDER cannot be empty")
        return normalized

    @field_validator(
        "OPENAI_DEFAULT_MODEL",
        "OPENAI_MODEL_PREMIUM",
        "OPENAI_MODEL_STANDARD",
        "OPENAI_MODEL_BASIC",
        "ANTHROPIC_DEFAULT_MODEL",
        "ANTHROPIC_MODEL_PREMIUM",
        "ANTHROPIC_MODEL_STANDARD",
        "ANTHROPIC_MODEL_BASIC",
        mode="before",
    )
    @classmethod
    def normalize_model_name(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("FALLBACK_PROVIDERS", mode="before")
    @classmethod
    def parse_fallback_providers(cls, value: Any) -> List[str]:
        if value is None:
            return ["openai", "anthropic"]

        if isinstance(value, list):
            return [str(item).strip().lower() for item in value if str(item).strip()]

        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "FALLBACK_PROVIDERS must be a JSON array or comma-separated list"
                    ) from exc
                if not isinstance(decoded, list):
                    raise ValueError("FALLBACK_PROVIDERS JSON value must be an array")
                return [str(item).strip().lower() for item in decoded if str(item).strip()]
            return [part.strip().lower() for part in raw.split(",") if part.strip()]

        raise ValueError("FALLBACK_PROVIDERS must be a list or string")

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        provider = provider.lower()
        configs = {
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL.rstrip("/"),
                "default_model": self.OPENAI_DEFAULT_MODEL or "",
                "tier_models": self.get_tier_models("openai"),
                "model_discovery_strict": self.MODEL_DISCOVERY_STRICT,
                "model_discovery_refresh_seconds": self.MODEL_DISCOVERY_REFRESH_SECONDS,
                "routing_require_discovered_models": self.ROUTING_REQUIRE_DISCOVERED_MODELS,
            },
            "anthropic": {
                "api_key": self.ANTHROPIC_API_KEY,
                "base_url": self.ANTHROPIC_BASE_URL.rstrip("/"),
                "default_model": self.ANTHROPIC_DEFAULT_MODEL or "",
                "tier_models": self.get_tier_models("anthropic"),
                "version": self.ANTHROPIC_VERSION,
                "model_discovery_strict": self.MODEL_DISCOVERY_STRICT,
                "model_discovery_refresh_seconds": self.MODEL_DISCOVERY_REFRESH_SECONDS,
                "routing_require_discovered_models": self.ROUTING_REQUIRE_DISCOVERED_MODELS,
            },
        }
        return configs.get(provider, {})

    def get_tier_models(self, provider: str) -> Dict[str, str]:
        """Return configured routing-tier preferences for a provider."""
        provider = provider.lower()

        if provider == "openai":
            return {
                "premium": self.OPENAI_MODEL_PREMIUM or "",
                "standard": self.OPENAI_MODEL_STANDARD or self.OPENAI_DEFAULT_MODEL or "",
                "basic": self.OPENAI_MODEL_BASIC or self.OPENAI_DEFAULT_MODEL or "",
            }

        if provider == "anthropic":
            return {
                "premium": self.ANTHROPIC_MODEL_PREMIUM or "",
                "standard": self.ANTHROPIC_MODEL_STANDARD or self.ANTHROPIC_DEFAULT_MODEL or "",
                "basic": self.ANTHROPIC_MODEL_BASIC or self.ANTHROPIC_DEFAULT_MODEL or "",
            }

        return {}

    def get_tier_model(self, provider: str, tier: str) -> Optional[str]:
        """Return the configured model preference for a provider/tier pair."""
        value = self.get_tier_models(provider).get(tier.lower())
        return value or None

    def get_provider_models(self, provider: str) -> List[str]:
        """Return configured non-empty model ids for a provider without inventing defaults."""
        config = self.get_provider_config(provider)
        candidates = [config.get("default_model")]
        candidates.extend(self.get_tier_models(provider).values())

        result: List[str] = []
        for candidate in candidates:
            normalized = str(candidate or "").strip()
            if normalized and normalized not in result:
                result.append(normalized)
        return result

    def get_enabled_provider_names(self) -> List[str]:
        """Return providers that are configured with credentials."""
        enabled: List[str] = []
        if self.OPENAI_API_KEY:
            enabled.append("openai")
        if self.ANTHROPIC_API_KEY:
            enabled.append("anthropic")
        return enabled

    def ensure_directories(self) -> None:
        """Ensure runtime directories exist."""
        for path in [self.RUNTIME_DIR, self.LOGS_DIR, self.CACHE_DIR]:
            os.makedirs(path, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
