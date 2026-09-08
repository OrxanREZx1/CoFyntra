"""
Tests: application configuration — updated for Milestone 2.

Verifies:
  - Settings load correctly with/without environment variables
  - is_valid_api_key detects placeholder vs valid keys
  - model_name is configurable
  - logging configuration behaves safely
"""

import os
import pytest

from startary.config import Settings, configure_logging, get_settings
from startary.llm.provider import is_valid_api_key


class TestSettings:
    def test_settings_load_without_api_key(self):
        """Settings must initialise successfully when no API key is provided."""
        settings = Settings(_env_file=None, openai_api_key=None)
        assert settings.openai_api_key is None

    def test_placeholder_api_key_detected(self):
        """Placeholder values must be identified as invalid API keys."""
        assert not is_valid_api_key(None)
        assert not is_valid_api_key("")
        assert not is_valid_api_key("   ")
        assert not is_valid_api_key("PASTE_YOUR_OPENAI_API_KEY_HERE")
        assert not is_valid_api_key("sk-...")
        assert is_valid_api_key("sk-proj-actualrealvalidkey123456789")

    def test_default_log_level(self):
        settings = Settings(_env_file=None)
        assert settings.log_level == "INFO"

    def test_default_model_name(self):
        settings = Settings(_env_file=None)
        assert settings.model_name == "gpt-4o"

    def test_database_url_is_none_by_default(self):
        settings = Settings(_env_file=None)
        assert settings.database_url is None

    def test_get_settings_returns_settings_instance(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_model_name_configurable_via_env(self, monkeypatch):
        """STARTARY_MODEL_NAME env var overrides default model."""
        monkeypatch.setenv("STARTARY_MODEL_NAME", "gpt-4-turbo")
        s = Settings(_env_file=None)
        assert s.model_name == "gpt-4-turbo"

    def test_openai_api_key_reads_standard_env_var(self, monkeypatch):
        """OPENAI_API_KEY (no STARTARY_ prefix) should be read."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        s = Settings(_env_file=None)
        assert s.openai_api_key == "sk-test-key-123"


class TestLogging:
    def test_configure_logging_does_not_raise(self):
        configure_logging("DEBUG")
        configure_logging("INFO")
        configure_logging("WARNING")

    def test_configure_logging_handles_invalid_level_gracefully(self):
        configure_logging("NOT_A_LEVEL")  # should not raise
