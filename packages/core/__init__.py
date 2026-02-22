"""Core package - shared utilities, types, and configuration.

This package provides common functionality used across all SignBridge packages:
- Configuration management
- Shared type definitions
- Protocol interfaces
- Custom exceptions
- Utility functions

Example usage:
    from packages.core import get_config, SignStatus, SignNotFoundError

    config = get_config()
    print(f"Signs directory: {config.signs_dir}")

    if sign is None:
        raise SignNotFoundError("HELLO")
"""

# Configuration
from .config import (
    Environment,
    LogLevel,
    SignBridgeConfig,
    clear_config_cache,
    get_config,
)

# Types
from .types import (
    LinguisticInfo,
    QualityScore,
    SignStatus,
    TimingInfo,
    VerificationInfo,
    VideoMetadata,
)

# Protocols
from .protocols import (
    Serializable,
    SignLookup,
    SignStore,
    VideoLoader,
)

# Errors
from .errors import (
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

# Utilities
from .utils import (
    compute_file_hash,
    ensure_dir,
    format_duration,
    frames_to_ms,
    gloss_to_path_safe,
    is_valid_gloss,
    ms_to_frames,
    normalize_gloss,
    now_iso,
    parse_iso,
    safe_json_load,
    safe_json_save,
    validate_quality_score,
)

__all__ = [
    # Config
    "Environment",
    "LogLevel",
    "SignBridgeConfig",
    "get_config",
    "clear_config_cache",
    # Types
    "SignStatus",
    "QualityScore",
    "VideoMetadata",
    "TimingInfo",
    "LinguisticInfo",
    "VerificationInfo",
    # Protocols
    "SignLookup",
    "SignStore",
    "VideoLoader",
    "Serializable",
    # Errors
    "SignBridgeError",
    "ConfigurationError",
    "MissingConfigError",
    "SignError",
    "SignNotFoundError",
    "SignAlreadyExistsError",
    "SignValidationError",
    "SignVerificationError",
    "TranslationError",
    "UnsupportedLanguageError",
    "GlossNotFoundError",
    "VideoError",
    "VideoNotFoundError",
    "VideoLoadError",
    "VideoExportError",
    "DatabaseError",
    "StorageError",
    # Utils
    "normalize_gloss",
    "gloss_to_path_safe",
    "now_iso",
    "parse_iso",
    "ms_to_frames",
    "frames_to_ms",
    "format_duration",
    "compute_file_hash",
    "ensure_dir",
    "safe_json_load",
    "safe_json_save",
    "is_valid_gloss",
    "validate_quality_score",
]
