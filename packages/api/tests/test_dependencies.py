"""Tests for API dependencies module."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestGetConfig:
    """Tests for get_config function."""

    def test_default_values(self):
        """Test that get_config returns correct default values."""
        from packages.api.dependencies import get_config

        # Clear environment variables that might affect the test
        env_vars = ["SIGNS_DIR", "VIDEO_CACHE_DIR", "API_HOST", "API_PORT", "API_CORS_ORIGINS"]
        with patch.dict(os.environ, {}, clear=True):
            # Remove any env vars that might be set
            for var in env_vars:
                os.environ.pop(var, None)

            config = get_config()

            assert config["signs_dir"] == Path("data/signs")
            assert config["cache_dir"] == Path("data/cache/videos")
            assert config["api_host"] == "0.0.0.0"
            assert config["api_port"] == 8000
            assert config["cors_origins"] == ["*"]

    def test_reads_signs_dir_from_environment(self, monkeypatch):
        """Test that signs_dir is read from SIGNS_DIR environment variable."""
        from packages.api.dependencies import get_config

        monkeypatch.setenv("SIGNS_DIR", "/custom/signs/path")

        config = get_config()

        assert config["signs_dir"] == Path("/custom/signs/path")

    def test_reads_cache_dir_from_environment(self, monkeypatch):
        """Test that cache_dir is read from VIDEO_CACHE_DIR environment variable."""
        from packages.api.dependencies import get_config

        monkeypatch.setenv("VIDEO_CACHE_DIR", "/custom/cache/path")

        config = get_config()

        assert config["cache_dir"] == Path("/custom/cache/path")

    def test_reads_api_host_from_environment(self, monkeypatch):
        """Test that api_host is read from API_HOST environment variable."""
        from packages.api.dependencies import get_config

        monkeypatch.setenv("API_HOST", "127.0.0.1")

        config = get_config()

        assert config["api_host"] == "127.0.0.1"

    def test_reads_api_port_from_environment(self, monkeypatch):
        """Test that api_port is read from API_PORT environment variable and converted to int."""
        from packages.api.dependencies import get_config

        monkeypatch.setenv("API_PORT", "3000")

        config = get_config()

        assert config["api_port"] == 3000
        assert isinstance(config["api_port"], int)

    def test_cors_origins_parsing_single(self, monkeypatch):
        """Test that single CORS origin is parsed correctly."""
        from packages.api.dependencies import get_config

        monkeypatch.setenv("API_CORS_ORIGINS", "https://example.com")

        config = get_config()

        assert config["cors_origins"] == ["https://example.com"]

    def test_cors_origins_parsing_multiple(self, monkeypatch):
        """Test that comma-separated CORS origins are parsed correctly."""
        from packages.api.dependencies import get_config

        monkeypatch.setenv("API_CORS_ORIGINS", "https://example.com,https://api.example.com,http://localhost:3000")

        config = get_config()

        assert config["cors_origins"] == [
            "https://example.com",
            "https://api.example.com",
            "http://localhost:3000",
        ]


class TestSingletonFactories:
    """Tests for singleton factory functions."""

    def setup_method(self):
        """Clear all lru_cache instances before each test."""
        from packages.api import dependencies

        # Clear all cached singletons
        dependencies.get_sign_store.cache_clear()
        dependencies.get_video_service.cache_clear()
        dependencies.get_sign_service.cache_clear()
        dependencies.get_translation_service.cache_clear()

    def teardown_method(self):
        """Clear all lru_cache instances after each test."""
        from packages.api import dependencies

        # Clear all cached singletons
        dependencies.get_sign_store.cache_clear()
        dependencies.get_video_service.cache_clear()
        dependencies.get_sign_service.cache_clear()
        dependencies.get_translation_service.cache_clear()

    @patch("packages.api.dependencies.SignStore")
    def test_get_sign_store_returns_sign_store(self, mock_sign_store_class):
        """Test that get_sign_store returns a SignStore instance."""
        from packages.api.dependencies import get_sign_store

        mock_instance = MagicMock()
        mock_sign_store_class.return_value = mock_instance

        result = get_sign_store()

        assert result is mock_instance
        mock_sign_store_class.assert_called_once()

    @patch("packages.api.dependencies.SignStore")
    def test_get_sign_store_singleton(self, mock_sign_store_class):
        """Test that get_sign_store returns the same object on subsequent calls."""
        from packages.api.dependencies import get_sign_store

        mock_instance = MagicMock()
        mock_sign_store_class.return_value = mock_instance

        result1 = get_sign_store()
        result2 = get_sign_store()

        assert result1 is result2
        # Should only be called once due to caching
        mock_sign_store_class.assert_called_once()

    @patch("packages.api.dependencies.VideoService")
    def test_get_video_service_returns_video_service(self, mock_video_service_class):
        """Test that get_video_service returns a VideoService instance."""
        from packages.api.dependencies import get_video_service

        mock_instance = MagicMock()
        mock_video_service_class.return_value = mock_instance

        result = get_video_service()

        assert result is mock_instance
        mock_video_service_class.assert_called_once()

    @patch("packages.api.dependencies.VideoService")
    def test_get_video_service_singleton(self, mock_video_service_class):
        """Test that get_video_service returns the same object on subsequent calls."""
        from packages.api.dependencies import get_video_service

        mock_instance = MagicMock()
        mock_video_service_class.return_value = mock_instance

        result1 = get_video_service()
        result2 = get_video_service()

        assert result1 is result2
        # Should only be called once due to caching
        mock_video_service_class.assert_called_once()

    @patch("packages.api.dependencies.SignService")
    @patch("packages.api.dependencies.get_sign_store")
    def test_get_sign_service_returns_sign_service(
        self, mock_get_sign_store, mock_sign_service_class
    ):
        """Test that get_sign_service returns a SignService instance."""
        from packages.api.dependencies import get_sign_service

        mock_store = MagicMock()
        mock_get_sign_store.return_value = mock_store
        mock_instance = MagicMock()
        mock_sign_service_class.return_value = mock_instance

        result = get_sign_service()

        assert result is mock_instance
        mock_sign_service_class.assert_called_once_with(sign_store=mock_store)

    @patch("packages.api.dependencies.SignService")
    @patch("packages.api.dependencies.get_sign_store")
    def test_get_sign_service_singleton(
        self, mock_get_sign_store, mock_sign_service_class
    ):
        """Test that get_sign_service returns the same object on subsequent calls."""
        from packages.api.dependencies import get_sign_service

        mock_store = MagicMock()
        mock_get_sign_store.return_value = mock_store
        mock_instance = MagicMock()
        mock_sign_service_class.return_value = mock_instance

        result1 = get_sign_service()
        result2 = get_sign_service()

        assert result1 is result2
        # Should only be called once due to caching
        mock_sign_service_class.assert_called_once()

    @patch("packages.api.dependencies.TranslationService")
    @patch("packages.api.dependencies.get_video_service")
    @patch("packages.api.dependencies.get_sign_store")
    def test_get_translation_service_returns_translation_service(
        self, mock_get_sign_store, mock_get_video_service, mock_translation_service_class
    ):
        """Test that get_translation_service returns a TranslationService instance."""
        from packages.api.dependencies import get_translation_service

        mock_store = MagicMock()
        mock_video_service = MagicMock()
        mock_get_sign_store.return_value = mock_store
        mock_get_video_service.return_value = mock_video_service
        mock_instance = MagicMock()
        mock_translation_service_class.return_value = mock_instance

        result = get_translation_service()

        assert result is mock_instance
        mock_translation_service_class.assert_called_once_with(
            sign_store=mock_store,
            video_service=mock_video_service,
        )

    @patch("packages.api.dependencies.TranslationService")
    @patch("packages.api.dependencies.get_video_service")
    @patch("packages.api.dependencies.get_sign_store")
    def test_get_translation_service_singleton(
        self, mock_get_sign_store, mock_get_video_service, mock_translation_service_class
    ):
        """Test that get_translation_service returns the same object on subsequent calls."""
        from packages.api.dependencies import get_translation_service

        mock_store = MagicMock()
        mock_video_service = MagicMock()
        mock_get_sign_store.return_value = mock_store
        mock_get_video_service.return_value = mock_video_service
        mock_instance = MagicMock()
        mock_translation_service_class.return_value = mock_instance

        result1 = get_translation_service()
        result2 = get_translation_service()

        assert result1 is result2
        # Should only be called once due to caching
        mock_translation_service_class.assert_called_once()
