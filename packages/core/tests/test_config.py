"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest

from packages.core.config import (
    Environment,
    LogLevel,
    SignBridgeConfig,
    clear_config_cache,
    get_config,
)


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear config cache before and after each test."""
    clear_config_cache()
    yield
    clear_config_cache()


@pytest.fixture
def clean_env():
    """Remove SignBridge env vars and restore after test."""
    env_vars = [
        "SIGNBRIDGE_DATA_DIR",
        "SIGNBRIDGE_SIGNS_DIR",
        "SIGNBRIDGE_CACHE_DIR",
        "SIGNBRIDGE_EXPORTS_DIR",
        "SIGNBRIDGE_ENV",
        "SIGNBRIDGE_LOG_LEVEL",
        "SIGNBRIDGE_DEBUG",
        "API_HOST",
        "API_PORT",
        "API_CORS_ORIGINS",
    ]

    # Store original values
    original = {k: os.environ.get(k) for k in env_vars}

    # Remove env vars
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


class TestEnvironment:
    """Tests for Environment enum."""

    def test_values(self):
        """Test enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.PRODUCTION.value == "production"
        assert Environment.TESTING.value == "testing"

    def test_from_string(self):
        """Test creating from string."""
        assert Environment("development") == Environment.DEVELOPMENT
        assert Environment("production") == Environment.PRODUCTION


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_values(self):
        """Test enum values."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"


class TestSignBridgeConfig:
    """Tests for SignBridgeConfig dataclass."""

    def test_create_config(self):
        """Test creating a config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SignBridgeConfig(
                data_dir=Path(tmpdir) / "data",
                signs_dir=Path(tmpdir) / "signs",
                cache_dir=Path(tmpdir) / "cache",
                exports_dir=Path(tmpdir) / "exports",
                env=Environment.TESTING,
                log_level=LogLevel.INFO,
                debug=False,
                api_host="localhost",
                api_port=8080,
                cors_origins=("http://localhost:3000",),
            )

            assert config.data_dir == Path(tmpdir) / "data"
            assert config.env == Environment.TESTING
            assert config.api_port == 8080

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SignBridgeConfig(
                data_dir=Path(tmpdir),
                signs_dir=Path(tmpdir),
                cache_dir=Path(tmpdir),
                exports_dir=Path(tmpdir),
                env=Environment.TESTING,
                log_level=LogLevel.INFO,
                debug=False,
                api_host="localhost",
                api_port=8080,
                cors_origins=("*",),
            )

            with pytest.raises(Exception):  # FrozenInstanceError
                config.api_port = 9000

    def test_convenience_properties(self):
        """Test convenience directory properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            signs_dir = Path(tmpdir) / "signs"
            config = SignBridgeConfig(
                data_dir=Path(tmpdir),
                signs_dir=signs_dir,
                cache_dir=Path(tmpdir) / "cache",
                exports_dir=Path(tmpdir) / "exports",
                env=Environment.TESTING,
                log_level=LogLevel.INFO,
                debug=False,
                api_host="localhost",
                api_port=8080,
                cors_origins=("*",),
            )

            assert config.verified_signs_dir == signs_dir / "verified"
            assert config.pending_signs_dir == signs_dir / "pending"
            assert config.imported_signs_dir == signs_dir / "imported"
            assert config.rejected_signs_dir == signs_dir / "rejected"

    def test_environment_properties(self):
        """Test environment check properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_args = dict(
                data_dir=Path(tmpdir),
                signs_dir=Path(tmpdir),
                cache_dir=Path(tmpdir),
                exports_dir=Path(tmpdir),
                log_level=LogLevel.INFO,
                debug=False,
                api_host="localhost",
                api_port=8080,
                cors_origins=("*",),
            )

            dev_config = SignBridgeConfig(**base_args, env=Environment.DEVELOPMENT)
            prod_config = SignBridgeConfig(**base_args, env=Environment.PRODUCTION)
            test_config = SignBridgeConfig(**base_args, env=Environment.TESTING)

            assert dev_config.is_development is True
            assert dev_config.is_production is False
            assert dev_config.is_testing is False

            assert prod_config.is_development is False
            assert prod_config.is_production is True

            assert test_config.is_testing is True


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_default(self, clean_env):
        """Test get_config with defaults."""
        config = get_config()

        assert config.env == Environment.DEVELOPMENT
        assert config.log_level == LogLevel.INFO
        assert config.debug is True  # Default in development
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000
        assert config.cors_origins == ("*",)

    def test_get_config_singleton(self, clean_env):
        """Test that get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_clear_config_cache(self, clean_env):
        """Test clearing the config cache."""
        config1 = get_config()
        clear_config_cache()
        config2 = get_config()

        # Should be equal but not the same object
        assert config1 is not config2
        assert config1.env == config2.env

    def test_env_override(self, clean_env):
        """Test environment variable override."""
        os.environ["SIGNBRIDGE_ENV"] = "production"

        config = get_config()

        assert config.env == Environment.PRODUCTION
        assert config.debug is False  # Debug is False in production

    def test_log_level_override(self, clean_env):
        """Test log level override."""
        os.environ["SIGNBRIDGE_LOG_LEVEL"] = "DEBUG"

        config = get_config()

        assert config.log_level == LogLevel.DEBUG

    def test_debug_override(self, clean_env):
        """Test debug mode override."""
        os.environ["SIGNBRIDGE_ENV"] = "production"
        os.environ["SIGNBRIDGE_DEBUG"] = "true"

        config = get_config()

        assert config.debug is True

    def test_api_settings_override(self, clean_env):
        """Test API settings override."""
        os.environ["API_HOST"] = "127.0.0.1"
        os.environ["API_PORT"] = "9000"
        os.environ["API_CORS_ORIGINS"] = "http://localhost:3000,http://localhost:3001"

        config = get_config()

        assert config.api_host == "127.0.0.1"
        assert config.api_port == 9000
        assert config.cors_origins == ("http://localhost:3000", "http://localhost:3001")

    def test_path_override(self, clean_env):
        """Test path override with env vars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["SIGNBRIDGE_DATA_DIR"] = tmpdir

            config = get_config()

            assert config.data_dir == Path(tmpdir)

    def test_invalid_env_fallback(self, clean_env):
        """Test invalid environment falls back to development."""
        os.environ["SIGNBRIDGE_ENV"] = "invalid_env"

        config = get_config()

        assert config.env == Environment.DEVELOPMENT

    def test_invalid_log_level_fallback(self, clean_env):
        """Test invalid log level falls back to INFO."""
        os.environ["SIGNBRIDGE_LOG_LEVEL"] = "INVALID"

        config = get_config()

        assert config.log_level == LogLevel.INFO
