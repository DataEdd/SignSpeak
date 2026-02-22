"""Custom exception classes for SignBridge.

Exception Hierarchy:
    SignBridgeError (base)
    ├── ConfigurationError
    │   └── MissingConfigError
    ├── SignError
    │   ├── SignNotFoundError
    │   ├── SignAlreadyExistsError
    │   ├── SignValidationError
    │   └── SignVerificationError
    ├── TranslationError
    │   ├── UnsupportedLanguageError
    │   └── GlossNotFoundError
    ├── VideoError
    │   ├── VideoNotFoundError
    │   ├── VideoLoadError
    │   └── VideoExportError
    └── DatabaseError
        └── StorageError
"""

from typing import Any, Optional


class SignBridgeError(Exception):
    """Base exception for all SignBridge errors.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        details: Additional error context
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        result: dict[str, Any] = {
            "error": self.code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


# ============ Configuration Errors ============


class ConfigurationError(SignBridgeError):
    """Error in application configuration."""

    pass


class MissingConfigError(ConfigurationError):
    """Required configuration value is missing."""

    def __init__(self, config_key: str, env_var: Optional[str] = None):
        message = f"Missing required configuration: {config_key}"
        if env_var:
            message += f" (set via {env_var})"
        super().__init__(
            message=message,
            code="missing_config",
            details={"config_key": config_key, "env_var": env_var},
        )


# ============ Sign Errors ============


class SignError(SignBridgeError):
    """Base class for sign-related errors."""

    pass


class SignNotFoundError(SignError):
    """Sign does not exist in the database."""

    def __init__(self, gloss: str, suggestions: Optional[list[str]] = None):
        super().__init__(
            message=f"Sign '{gloss}' not found in database",
            code="sign_not_found",
            details={"gloss": gloss, "suggestions": suggestions or []},
        )
        self.gloss = gloss
        self.suggestions = suggestions or []


class SignAlreadyExistsError(SignError):
    """Sign already exists in the database."""

    def __init__(self, gloss: str, status: str):
        super().__init__(
            message=f"Sign '{gloss}' already exists with status '{status}'",
            code="sign_already_exists",
            details={"gloss": gloss, "status": status},
        )
        self.gloss = gloss
        self.status = status


class SignValidationError(SignError):
    """Sign data failed validation."""

    def __init__(self, gloss: str, errors: list[str]):
        super().__init__(
            message=f"Sign '{gloss}' validation failed: {'; '.join(errors)}",
            code="sign_validation_error",
            details={"gloss": gloss, "validation_errors": errors},
        )
        self.gloss = gloss
        self.validation_errors = errors


class SignVerificationError(SignError):
    """Error during sign verification process."""

    def __init__(self, gloss: str, reason: str):
        super().__init__(
            message=f"Cannot verify sign '{gloss}': {reason}",
            code="sign_verification_error",
            details={"gloss": gloss, "reason": reason},
        )


# ============ Translation Errors ============


class TranslationError(SignBridgeError):
    """Base class for translation-related errors."""

    pass


class UnsupportedLanguageError(TranslationError):
    """Language is not supported for translation."""

    def __init__(self, language: str, supported: list[str]):
        super().__init__(
            message=f"Language '{language}' is not supported",
            code="unsupported_language",
            details={"language": language, "supported": supported},
        )


class GlossNotFoundError(TranslationError):
    """Gloss not available for translation output."""

    def __init__(self, gloss: str, alternatives: Optional[list[str]] = None):
        super().__init__(
            message=f"No sign available for gloss '{gloss}'",
            code="gloss_not_found",
            details={"gloss": gloss, "alternatives": alternatives or []},
        )


# ============ Video Errors ============


class VideoError(SignBridgeError):
    """Base class for video-related errors."""

    pass


class VideoNotFoundError(VideoError):
    """Video file does not exist."""

    def __init__(self, path: str, gloss: Optional[str] = None):
        message = f"Video not found: {path}"
        if gloss:
            message = f"Video not found for sign '{gloss}': {path}"
        super().__init__(
            message=message,
            code="video_not_found",
            details={"path": path, "gloss": gloss},
        )


class VideoLoadError(VideoError):
    """Failed to load video file."""

    def __init__(self, path: str, reason: str):
        super().__init__(
            message=f"Failed to load video '{path}': {reason}",
            code="video_load_error",
            details={"path": path, "reason": reason},
        )


class VideoExportError(VideoError):
    """Failed to export video."""

    def __init__(self, output_path: str, reason: str):
        super().__init__(
            message=f"Failed to export video to '{output_path}': {reason}",
            code="video_export_error",
            details={"output_path": output_path, "reason": reason},
        )


# ============ Database Errors ============


class DatabaseError(SignBridgeError):
    """Base class for database-related errors."""

    pass


class StorageError(DatabaseError):
    """Error reading/writing to storage."""

    def __init__(self, operation: str, path: str, reason: str):
        super().__init__(
            message=f"Storage {operation} failed for '{path}': {reason}",
            code="storage_error",
            details={"operation": operation, "path": path, "reason": reason},
        )
