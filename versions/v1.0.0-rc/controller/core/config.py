"""
Configuration management using pydantic-settings.
Externalized config via environment variables.
"""

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
    
    # Provider settings
    DEFAULT_PROVIDER: str = Field(default="openai", description="Default provider")
    FALLBACK_PROVIDERS: List[str] = Field(default=["openai", "anthropic"], description="Fallback chain")
    
    # OpenAI settings
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
    OPENAI_DEFAULT_MODEL: str = Field(default="gpt-4o-mini", description="Default OpenAI model")
    
    # Anthropic settings
    ANTHROPIC_BASE_URL: str = Field(default="https://api.anthropic.com", description="Anthropic API base URL")
    ANTHROPIC_DEFAULT_MODEL: str = Field(default="claude-3-haiku-20240307", description="Default Anthropic model")
    ANTHROPIC_VERSION: str = Field(default="2023-06-01", description="Anthropic API version")
    
    # Intelligence settings
    CONFIDENCE_THRESHOLD_HIGH: float = Field(default=0.8, ge=0.0, le=1.0, description="High confidence threshold")
    CONFIDENCE_THRESHOLD_MEDIUM: float = Field(default=0.5, ge=0.0, le=1.0, description="Medium confidence threshold")
    SAFETY_RISK_THRESHOLD: float = Field(default=0.7, ge=0.0, le=1.0, description="Safety risk threshold")
    ENABLE_PII_DETECTION: bool = Field(default=True, description="Enable PII detection")
    ENABLE_JAILBREAK_DETECTION: bool = Field(default=True, description="Enable jailbreak detection")
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=60, ge=1, description="Requests per minute limit")
    
    # Circuit breaker
    CIRCUIT_BREAKER_ENABLED: bool = Field(default=True, description="Enable circuit breaker")
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5, ge=1, description="Failure threshold")
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(default=30, ge=1, description="Recovery timeout in seconds")
    
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
    def validate_log_level(cls, v):
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()
    
    @field_validator("CONTROLLER_API_KEY")
    @classmethod
    def validate_api_key(cls, v):
        if not v or len(v) < 16:
            raise ValueError("CONTROLLER_API_KEY must be at least 16 characters")
        return v
    
    @field_validator("FALLBACK_PROVIDERS", mode="before")
    @classmethod
    def parse_fallback_providers(cls, v):
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get configuration for specific provider."""
        configs = {
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "base_url": self.OPENAI_BASE_URL,
                "default_model": self.OPENAI_DEFAULT_MODEL,
            },
            "anthropic": {
                "api_key": self.ANTHROPIC_API_KEY,
                "base_url": self.ANTHROPIC_BASE_URL,
                "default_model": self.ANTHROPIC_DEFAULT_MODEL,
                "version": self.ANTHROPIC_VERSION,
            },
        }
        return configs.get(provider, {})
    
    def ensure_directories(self):
        """Ensure runtime directories exist."""
        for path in [self.RUNTIME_DIR, self.LOGS_DIR, self.CACHE_DIR]:
            os.makedirs(path, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
