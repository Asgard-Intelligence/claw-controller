"""Configuration for Controller v1.4 (keeps v1.3 env compatibility)."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class SettingsV13(BaseSettings):
    """Controller settings class kept under v130 name for backward compatibility."""

    # Application
    APP_NAME: str = "OpenClaw Controller v1.4.0"
    APP_VERSION: str = "1.4.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    API_WORKERS: int = 1

    # Session
    SESSION_STORE_TYPE: Literal["redis", "file", "memory"] = "file"
    SESSION_TTL_SECONDS: int = 86400
    SESSION_REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_FILE_PATH: str = "./sessions"

    # Identity
    IDENTITY_DIR: str = "./identity"
    IDENTITY_CACHE_FOREVER: bool = True
    IDENTITY_VERIFY_HASH: bool = True
    IDENTITY_AUTO_INJECT: bool = True

    # Context
    CONTEXT_PRESERVATION_ENABLED: bool = True
    CONTEXT_TRANSFER_TIMEOUT_SECONDS: int = 30
    CONTEXT_TRUNCATION_STRATEGY: Literal["middle", "oldest", "summarize"] = "middle"
    CONTEXT_WINDOW_THRESHOLD: float = 0.9
    TOKEN_ESTIMATION_CHARS_PER_TOKEN: int = 4
    TOKEN_ESTIMATION_SAFETY_MARGIN: float = 1.15

    # Routing (legacy + new)
    SAFE_ROUTING_ENABLED: bool = True
    CONTEXT_WINDOW_CHECK_ENABLED: bool = True
    DEFAULT_PROVIDER: str = "openai"
    FALLBACK_PROVIDERS: List[str] = Field(default_factory=lambda: ["anthropic", "openai", "ollama"])
    ROUTING_MODE: Literal["balanced", "quality", "cost", "latency"] = "balanced"
    ROUTE_FALLBACK_MAX_HOPS: int = 3
    ROUTE_STICKINESS_ENABLED: bool = True

    # Registry/bootstrap
    ROUTE_REGISTRY_PATH: Optional[str] = None

    # Health
    HEALTH_ENABLED: bool = True
    HEALTH_CACHE_TTL_SECONDS: int = 30
    HEALTH_PROBE_TIMEOUT_SECONDS: float = 5.0
    HEALTH_PROBE_RETRIES: int = 1

    # Validation
    VALIDATION_PIPELINE_ENABLED: bool = True
    VALIDATION_CACHE_TTL_SECONDS: int = 300
    VALIDATION_PROBE_TIMEOUT_SECONDS: float = 6.0
    INVARIANT_CHECKING: Literal["strict", "warn", "off"] = "warn"

    # Provider config (legacy)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL_PREMIUM: str = "gpt-4-turbo"
    OPENAI_MODEL_STANDARD: str = "gpt-4"
    OPENAI_MODEL_BASIC: str = "gpt-3.5-turbo"

    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com"
    ANTHROPIC_MODEL_PREMIUM: str = "claude-3-opus-20240229"
    ANTHROPIC_MODEL_STANDARD: str = "claude-3-sonnet-20240229"
    ANTHROPIC_MODEL_BASIC: str = "claude-3-haiku-20240307"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_VALIDATION_BASE_URL: Optional[str] = None
    OLLAMA_HEALTH_ENDPOINT: str = "/api/tags"
    OLLAMA_MODEL: str = "gemma4-e4b-q4-local"

    # Timeouts
    TIMEOUT_PROFILE_DEFAULT_SECONDS: float = 60.0
    TIMEOUT_PROFILE_CLOUD_DEFAULT_SECONDS: float = 60.0
    TIMEOUT_PROFILE_LOCAL_DEFAULT_SECONDS: float = 120.0

    # Compatibility and observability
    V120_COMPATIBILITY_MODE: bool = False
    STRICT_MODE: bool = False
    ROUTING_LOG_LEVEL: Literal["debug", "info", "warn", "error"] = "info"
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090
    TRACING_ENABLED: bool = True

    # Security
    API_KEY_REQUIRED: bool = False
    API_KEY: Optional[str] = None
    ALLOWED_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_REQUESTS_PER_HOUR: int = 1000

    # Feature flags
    FEATURE_SESSION_PERSISTENCE: bool = True
    FEATURE_IDENTITY_CACHING: bool = True
    FEATURE_CONTEXT_PRESERVATION: bool = True
    FEATURE_SAFE_ROUTING: bool = True
    FEATURE_PROVIDER_PINNING: bool = True
    FEATURE_CONTEXT_AWARE_FALLBACK: bool = True
    FEATURE_VALIDATION_PIPELINE: bool = True
    FEATURE_ROUTING_LOGGING: bool = True
    FEATURE_METRICS_COLLECTION: bool = True
    FEATURE_SESSION_TRACING: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def get_feature_flags(self) -> Dict[str, bool]:
        return {
            "session_persistence": self.FEATURE_SESSION_PERSISTENCE,
            "identity_caching": self.FEATURE_IDENTITY_CACHING,
            "context_preservation": self.FEATURE_CONTEXT_PRESERVATION,
            "safe_routing": self.FEATURE_SAFE_ROUTING,
            "provider_pinning": self.FEATURE_PROVIDER_PINNING,
            "context_aware_fallback": self.FEATURE_CONTEXT_AWARE_FALLBACK,
            "validation_pipeline": self.FEATURE_VALIDATION_PIPELINE,
            "routing_logging": self.FEATURE_ROUTING_LOGGING,
            "metrics_collection": self.FEATURE_METRICS_COLLECTION,
            "session_tracing": self.FEATURE_SESSION_TRACING,
        }

    def get_provider_config(self, provider: str) -> Dict[str, object]:
        configs: Dict[str, Dict[str, object]] = {
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL,
                "models": {
                    "premium": self.OPENAI_MODEL_PREMIUM,
                    "standard": self.OPENAI_MODEL_STANDARD,
                    "basic": self.OPENAI_MODEL_BASIC,
                },
            },
            "anthropic": {
                "api_key": self.ANTHROPIC_API_KEY,
                "base_url": self.ANTHROPIC_BASE_URL,
                "models": {
                    "premium": self.ANTHROPIC_MODEL_PREMIUM,
                    "standard": self.ANTHROPIC_MODEL_STANDARD,
                    "basic": self.ANTHROPIC_MODEL_BASIC,
                },
            },
            "ollama": {
                "base_url": self.OLLAMA_BASE_URL,
                "validation_base_url": self.OLLAMA_VALIDATION_BASE_URL or self.OLLAMA_BASE_URL,
                "health_endpoint": self.OLLAMA_HEALTH_ENDPOINT,
                "model": self.OLLAMA_MODEL,
            },
        }
        return configs.get(provider, {})

    def timeout_for_profile(self, profile: str) -> float:
        if profile == "cloud_default":
            return self.TIMEOUT_PROFILE_CLOUD_DEFAULT_SECONDS
        if profile == "local_default":
            return self.TIMEOUT_PROFILE_LOCAL_DEFAULT_SECONDS
        return self.TIMEOUT_PROFILE_DEFAULT_SECONDS


settings = SettingsV13()
