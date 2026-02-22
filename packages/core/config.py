"""Environment and path configuration for SignBridge.

Provides centralized configuration with sensible defaults.
All configuration is loaded from environment variables.

Usage:
    from packages.core import get_config

    config = get_config()
    print(f"Signs directory: {config.signs_dir}")
    print(f"Environment: {config.env}")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path


class Environment(Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(Enum):
    """Logging level."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SignBridgeConfig:
    """Application configuration.

    Immutable configuration object created from environment variables.
    All paths are absolute and validated.

    Attributes:
        data_dir: Base data directory
        signs_dir: Sign database directory
        cache_dir: Cache directory for generated files
        exports_dir: Directory for exported videos
        env: Current environment (development/production/testing)
        log_level: Logging level
        debug: Debug mode enabled
        api_host: API server host
        api_port: API server port
        cors_origins: Allowed CORS origins
    """

    # Core paths
    data_dir: Path
    signs_dir: Path
    cache_dir: Path
    exports_dir: Path

    # Environment
    env: Environment
    log_level: LogLevel
    debug: bool

    # API settings
    api_host: str
    api_port: int
    cors_origins: tuple[str, ...]

    def __post_init__(self) -> None:
        """Ensure directories exist in non-testing environments."""
        if self.env != Environment.TESTING:
            for path in [self.data_dir, self.signs_dir, self.cache_dir, self.exports_dir]:
                path.mkdir(parents=True, exist_ok=True)

    @property
    def verified_signs_dir(self) -> Path:
        """Get the verified signs directory."""
        return self.signs_dir / "verified"

    @property
    def pending_signs_dir(self) -> Path:
        """Get the pending signs directory."""
        return self.signs_dir / "pending"

    @property
    def imported_signs_dir(self) -> Path:
        """Get the imported signs directory."""
        return self.signs_dir / "imported"

    @property
    def rejected_signs_dir(self) -> Path:
        """Get the rejected signs directory."""
        return self.signs_dir / "rejected"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.env == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.env == Environment.TESTING


def _find_project_root() -> Path:
    """Find project root by looking for packages/ directory.

    Walks up from the current file's location to find the project root.
    Falls back to current working directory if not found.

    Returns:
        Path to project root
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "packages").exists():
            return parent
    return Path.cwd()


def _get_env_path(var: str, default: Path) -> Path:
    """Get path from environment variable or use default.

    Args:
        var: Environment variable name
        default: Default path if not set

    Returns:
        Absolute path from env var or default
    """
    value = os.environ.get(var)
    if value:
        path = Path(value)
        # Make relative paths absolute from project root
        if not path.is_absolute():
            path = _find_project_root() / path
        return path
    return default


@lru_cache(maxsize=1)
def get_config() -> SignBridgeConfig:
    """Get the application configuration (singleton).

    Configuration is loaded from environment variables:
    - SIGNBRIDGE_DATA_DIR: Base data directory
    - SIGNBRIDGE_SIGNS_DIR: Signs database directory
    - SIGNBRIDGE_CACHE_DIR: Cache directory for generated files
    - SIGNBRIDGE_EXPORTS_DIR: Directory for exported videos
    - SIGNBRIDGE_ENV: Environment (development/production/testing)
    - SIGNBRIDGE_LOG_LEVEL: Logging level (DEBUG/INFO/WARNING/ERROR)
    - SIGNBRIDGE_DEBUG: Enable debug mode (1/true/yes)
    - API_HOST: API server host (default: 0.0.0.0)
    - API_PORT: API server port (default: 8000)
    - API_CORS_ORIGINS: Comma-separated CORS origins (default: *)

    Returns:
        Immutable SignBridgeConfig instance
    """
    project_root = _find_project_root()

    # Determine environment
    env_str = os.environ.get("SIGNBRIDGE_ENV", "development").lower()
    try:
        env = Environment(env_str)
    except ValueError:
        env = Environment.DEVELOPMENT

    # Determine log level
    log_str = os.environ.get("SIGNBRIDGE_LOG_LEVEL", "INFO").upper()
    try:
        log_level = LogLevel(log_str)
    except ValueError:
        log_level = LogLevel.INFO

    # Debug mode
    debug_str = os.environ.get("SIGNBRIDGE_DEBUG", "").lower()
    debug = debug_str in ("1", "true", "yes") or env == Environment.DEVELOPMENT

    # Paths
    data_dir = _get_env_path("SIGNBRIDGE_DATA_DIR", project_root / "data")
    signs_dir = _get_env_path("SIGNBRIDGE_SIGNS_DIR", data_dir / "signs")
    cache_dir = _get_env_path("SIGNBRIDGE_CACHE_DIR", data_dir / "cache")
    exports_dir = _get_env_path("SIGNBRIDGE_EXPORTS_DIR", data_dir / "exports")

    # API settings
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = int(os.environ.get("API_PORT", "8000"))
    cors_str = os.environ.get("API_CORS_ORIGINS", "*")
    cors_origins = tuple(s.strip() for s in cors_str.split(",") if s.strip())

    return SignBridgeConfig(
        data_dir=data_dir,
        signs_dir=signs_dir,
        cache_dir=cache_dir,
        exports_dir=exports_dir,
        env=env,
        log_level=log_level,
        debug=debug,
        api_host=api_host,
        api_port=api_port,
        cors_origins=cors_origins,
    )


def clear_config_cache() -> None:
    """Clear the configuration cache.

    Useful for testing when environment variables change
    between test cases.
    """
    get_config.cache_clear()
