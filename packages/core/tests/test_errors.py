"""Tests for custom exception classes."""

import pytest

from packages.core.errors import (
    ConfigurationError,
    DatabaseError,
    GlossNotFoundError,
    MissingConfigError,
    SignAlreadyExistsError,
    SignBridgeError,
    SignError,
    SignNotFoundError,
    SignValidationError,
    SignVerificationError,
    StorageError,
    TranslationError,
    UnsupportedLanguageError,
    VideoError,
    VideoExportError,
    VideoLoadError,
    VideoNotFoundError,
)


class TestSignBridgeError:
    """Tests for base SignBridgeError."""

    def test_basic_error(self):
        """Test creating a basic error."""
        error = SignBridgeError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.code == "SignBridgeError"
        assert error.details == {}

    def test_error_with_code(self):
        """Test error with custom code."""
        error = SignBridgeError("Failed", code="custom_error")

        assert error.code == "custom_error"

    def test_error_with_details(self):
        """Test error with details."""
        error = SignBridgeError("Failed", details={"key": "value"})

        assert error.details == {"key": "value"}

    def test_to_dict(self):
        """Test converting error to dictionary."""
        error = SignBridgeError(
            "Failed",
            code="test_error",
            details={"foo": "bar"},
        )

        result = error.to_dict()

        assert result["error"] == "test_error"
        assert result["message"] == "Failed"
        assert result["details"] == {"foo": "bar"}

    def test_to_dict_without_details(self):
        """Test to_dict without details."""
        error = SignBridgeError("Failed")

        result = error.to_dict()

        assert result["error"] == "SignBridgeError"
        assert result["message"] == "Failed"
        assert "details" not in result

    def test_inheritance(self):
        """Test exception inheritance."""
        error = SignBridgeError("test")

        assert isinstance(error, Exception)


class TestConfigurationErrors:
    """Tests for configuration errors."""

    def test_configuration_error(self):
        """Test basic configuration error."""
        error = ConfigurationError("Bad config")

        assert isinstance(error, SignBridgeError)
        assert error.message == "Bad config"

    def test_missing_config_error(self):
        """Test missing config error."""
        error = MissingConfigError("database_url", "DATABASE_URL")

        assert isinstance(error, ConfigurationError)
        assert "database_url" in error.message
        assert "DATABASE_URL" in error.message
        assert error.code == "missing_config"
        assert error.details["config_key"] == "database_url"
        assert error.details["env_var"] == "DATABASE_URL"

    def test_missing_config_without_env_var(self):
        """Test missing config error without env var."""
        error = MissingConfigError("api_key")

        assert "api_key" in error.message
        assert error.details["env_var"] is None


class TestSignErrors:
    """Tests for sign-related errors."""

    def test_sign_error(self):
        """Test basic sign error."""
        error = SignError("Sign problem")

        assert isinstance(error, SignBridgeError)

    def test_sign_not_found_error(self):
        """Test sign not found error."""
        error = SignNotFoundError("HELLO")

        assert isinstance(error, SignError)
        assert "HELLO" in error.message
        assert error.gloss == "HELLO"
        assert error.suggestions == []
        assert error.code == "sign_not_found"

    def test_sign_not_found_with_suggestions(self):
        """Test sign not found error with suggestions."""
        error = SignNotFoundError("HELO", suggestions=["HELLO", "HELP"])

        assert error.suggestions == ["HELLO", "HELP"]
        assert error.details["suggestions"] == ["HELLO", "HELP"]

    def test_sign_already_exists_error(self):
        """Test sign already exists error."""
        error = SignAlreadyExistsError("HELLO", "pending")

        assert isinstance(error, SignError)
        assert "HELLO" in error.message
        assert "pending" in error.message
        assert error.gloss == "HELLO"
        assert error.status == "pending"
        assert error.code == "sign_already_exists"

    def test_sign_validation_error(self):
        """Test sign validation error."""
        errors = ["Missing video", "Invalid gloss format"]
        error = SignValidationError("HELLO", errors)

        assert isinstance(error, SignError)
        assert "HELLO" in error.message
        assert error.gloss == "HELLO"
        assert error.validation_errors == errors
        assert error.code == "sign_validation_error"

    def test_sign_verification_error(self):
        """Test sign verification error."""
        error = SignVerificationError("HELLO", "Sign is still pending")

        assert isinstance(error, SignError)
        assert "HELLO" in error.message
        assert "pending" in error.message
        assert error.code == "sign_verification_error"


class TestTranslationErrors:
    """Tests for translation-related errors."""

    def test_translation_error(self):
        """Test basic translation error."""
        error = TranslationError("Translation failed")

        assert isinstance(error, SignBridgeError)

    def test_unsupported_language_error(self):
        """Test unsupported language error."""
        error = UnsupportedLanguageError("fr", ["en", "es"])

        assert isinstance(error, TranslationError)
        assert "fr" in error.message
        assert error.code == "unsupported_language"
        assert error.details["language"] == "fr"
        assert error.details["supported"] == ["en", "es"]

    def test_gloss_not_found_error(self):
        """Test gloss not found error."""
        error = GlossNotFoundError("UNKNOWN")

        assert isinstance(error, TranslationError)
        assert "UNKNOWN" in error.message
        assert error.code == "gloss_not_found"

    def test_gloss_not_found_with_alternatives(self):
        """Test gloss not found error with alternatives."""
        error = GlossNotFoundError("UNKNWN", alternatives=["UNKNOWN", "KNOW"])

        assert error.details["alternatives"] == ["UNKNOWN", "KNOW"]


class TestVideoErrors:
    """Tests for video-related errors."""

    def test_video_error(self):
        """Test basic video error."""
        error = VideoError("Video problem")

        assert isinstance(error, SignBridgeError)

    def test_video_not_found_error(self):
        """Test video not found error."""
        error = VideoNotFoundError("/path/to/video.mp4")

        assert isinstance(error, VideoError)
        assert "/path/to/video.mp4" in error.message
        assert error.code == "video_not_found"
        assert error.details["path"] == "/path/to/video.mp4"

    def test_video_not_found_with_gloss(self):
        """Test video not found error with gloss."""
        error = VideoNotFoundError("/path/to/video.mp4", gloss="HELLO")

        assert "HELLO" in error.message
        assert error.details["gloss"] == "HELLO"

    def test_video_load_error(self):
        """Test video load error."""
        error = VideoLoadError("/path/to/video.mp4", "Corrupt file")

        assert isinstance(error, VideoError)
        assert "Corrupt file" in error.message
        assert error.code == "video_load_error"
        assert error.details["reason"] == "Corrupt file"

    def test_video_export_error(self):
        """Test video export error."""
        error = VideoExportError("/output/video.mp4", "Disk full")

        assert isinstance(error, VideoError)
        assert "Disk full" in error.message
        assert error.code == "video_export_error"
        assert error.details["output_path"] == "/output/video.mp4"


class TestDatabaseErrors:
    """Tests for database-related errors."""

    def test_database_error(self):
        """Test basic database error."""
        error = DatabaseError("Database problem")

        assert isinstance(error, SignBridgeError)

    def test_storage_error(self):
        """Test storage error."""
        error = StorageError("write", "/path/to/file", "Permission denied")

        assert isinstance(error, DatabaseError)
        assert "write" in error.message
        assert "Permission denied" in error.message
        assert error.code == "storage_error"
        assert error.details["operation"] == "write"
        assert error.details["path"] == "/path/to/file"
        assert error.details["reason"] == "Permission denied"


class TestErrorCodes:
    """Tests for error code uniqueness."""

    def test_error_codes_are_unique(self):
        """Test that all error codes are unique."""
        errors = [
            SignBridgeError("test"),
            ConfigurationError("test"),
            MissingConfigError("key"),
            SignError("test"),
            SignNotFoundError("HELLO"),
            SignAlreadyExistsError("HELLO", "pending"),
            SignValidationError("HELLO", []),
            SignVerificationError("HELLO", "reason"),
            TranslationError("test"),
            UnsupportedLanguageError("fr", []),
            GlossNotFoundError("UNKNOWN"),
            VideoError("test"),
            VideoNotFoundError("/path"),
            VideoLoadError("/path", "reason"),
            VideoExportError("/path", "reason"),
            DatabaseError("test"),
            StorageError("read", "/path", "reason"),
        ]

        codes = [e.code for e in errors]
        # Base classes use class name as code, so allow some duplicates
        # But specific errors should have unique codes
        specific_codes = [
            e.code for e in errors
            if e.code not in ("SignBridgeError", "ConfigurationError", "SignError",
                              "TranslationError", "VideoError", "DatabaseError")
        ]

        assert len(specific_codes) == len(set(specific_codes)), "Specific error codes should be unique"
