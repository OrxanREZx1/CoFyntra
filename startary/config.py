"""
Configuration for the Startary AI system.

Uses pydantic-settings so that environment variables override defaults.

M1 changes:
  - OPENAI_API_KEY is now required for live AI execution.
    It reads from the standard OPENAI_API_KEY env var (no STARTARY_ prefix)
    so it's compatible with the standard OpenAI toolchain convention.
  - STARTARY_MODEL_NAME controls model selection (default: gpt-4o).
  - Deterministic tests do NOT require OPENAI_API_KEY.
"""

from __future__ import annotations

import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central settings object.

    OPENAI_API_KEY:     Required for AI execution; read from environment directly.
    STARTARY_MODEL_NAME: Model for diagnosis (default: gpt-4o).
    STARTARY_LOG_LEVEL:  Logging verbosity (default: INFO).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Development ---
    # Reads STARTARY_LOG_LEVEL from env
    startary_log_level: str = Field(default="INFO", alias="startary_log_level")
    log_level: str = "INFO"

    # --- LLM (Milestone 1+) ---
    # Reads standard OPENAI_API_KEY (no prefix) — compatible with OpenAI convention
    openai_api_key: str | None = Field(default=None, alias="openai_api_key")
    # Reads STARTARY_MODEL_NAME from env
    model_name: str = Field(default="gpt-4o", alias="startary_model_name")
    fast_model_name: str | None = Field(default=None, alias="startary_fast_model")
    reasoning_model_name: str | None = Field(default=None, alias="startary_reasoning_model")
    fast_model_fallback: str | None = Field(default=None, alias="startary_fast_model_fallback")
    reasoning_model_fallback: str | None = Field(default=None, alias="startary_reasoning_model_fallback")

    def get_model_for_tier(self, tier: str = "reasoning") -> str:
        """Resolve model name for specified task tier: 'fast', 'medium', or 'reasoning'."""
        tier_lower = tier.lower().strip()
        if tier_lower in ("fast", "lightweight"):
            return self.fast_model_name or "gpt-4o-mini"
        if tier_lower in ("medium",):
            return self.fast_model_name or self.model_name
        if tier_lower in ("reasoning", "strategic"):
            return self.reasoning_model_name or self.model_name
        return self.model_name

    def get_fallback_for_tier(self, tier: str = "reasoning") -> str | None:
        """Resolve fallback model name for specified task tier."""
        tier_lower = tier.lower().strip()
        if tier_lower in ("fast", "lightweight"):
            return self.fast_model_fallback
        if tier_lower in ("medium",):
            return self.fast_model_fallback or self.model_name
        return self.reasoning_model_fallback

    # --- Database (Milestone 2+) ---
    database_url: str | None = Field(default=None, alias="startary_database_url")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


def get_settings() -> Settings:
    """Return the application settings singleton."""
    return Settings()


def get_model_for_tier(settings: Settings, tier: str = "reasoning") -> str:
    """Resolve model name for specified task tier: 'fast', 'medium', or 'reasoning'."""
    return settings.get_model_for_tier(tier)


def configure_logging(level: str = "INFO") -> None:
    """Configure standard Python logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
