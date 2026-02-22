"""Common utility functions for SignBridge.

Provides helper functions used across multiple packages.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


# ============ String Utilities ============


def normalize_gloss(gloss: str) -> str:
    """Normalize a gloss to canonical form.

    - Converts to uppercase
    - Removes leading/trailing whitespace
    - Replaces internal whitespace with hyphens

    Args:
        gloss: The gloss string to normalize

    Returns:
        Normalized gloss string

    Examples:
        >>> normalize_gloss("hello")
        'HELLO'
        >>> normalize_gloss("thank you")
        'THANK-YOU'
    """
    gloss = gloss.strip().upper()
    gloss = re.sub(r"\s+", "-", gloss)
    return gloss


def gloss_to_path_safe(gloss: str) -> str:
    """Convert gloss to filesystem-safe string.

    Args:
        gloss: The gloss to convert

    Returns:
        Filesystem-safe string
    """
    gloss = normalize_gloss(gloss)
    gloss = re.sub(r"[^\w\-]", "_", gloss)
    return gloss


# ============ Time Utilities ============


def now_iso() -> str:
    """Get current time in ISO format (UTC).

    Returns:
        ISO formatted datetime string with Z suffix
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(iso_str: str) -> datetime:
    """Parse ISO format datetime string.

    Args:
        iso_str: ISO formatted datetime string

    Returns:
        datetime object (timezone-aware if Z suffix present)
    """
    if iso_str.endswith("Z"):
        iso_str = iso_str[:-1] + "+00:00"
    return datetime.fromisoformat(iso_str)


def ms_to_frames(duration_ms: float, fps: float) -> int:
    """Convert milliseconds to frame count.

    Args:
        duration_ms: Duration in milliseconds
        fps: Frames per second

    Returns:
        Number of frames (minimum 1)
    """
    return max(1, int((duration_ms / 1000) * fps))


def frames_to_ms(num_frames: int, fps: float) -> float:
    """Convert frame count to milliseconds.

    Args:
        num_frames: Number of frames
        fps: Frames per second

    Returns:
        Duration in milliseconds
    """
    return (num_frames / fps) * 1000


def format_duration(ms: int) -> str:
    """Format milliseconds as human-readable duration.

    Args:
        ms: Duration in milliseconds

    Returns:
        Formatted string (e.g., "1.5s", "2m 30s")
    """
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.0f}s"


# ============ File Utilities ============


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file.

    Args:
        path: Path to file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hex digest of file hash
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating if necessary.

    Args:
        path: Directory path

    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_json_load(path: Path, default: T | None = None) -> Any:
    """Safely load JSON file, returning default on error.

    Args:
        path: Path to JSON file
        default: Value to return if file doesn't exist or is invalid

    Returns:
        Parsed JSON or default value
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def safe_json_save(path: Path, data: Any, indent: int = 2) -> bool:
    """Safely save data as JSON.

    Creates parent directories if needed.

    Args:
        path: Path to save to
        data: Data to serialize
        indent: JSON indentation

    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except (OSError, TypeError):
        return False


# ============ Validation Utilities ============


def is_valid_gloss(gloss: str) -> bool:
    """Check if gloss is valid format.

    Valid glosses:
    - Non-empty
    - Only letters, numbers, and hyphens after normalization
    - No leading/trailing hyphens

    Args:
        gloss: Gloss to validate

    Returns:
        True if valid
    """
    if not gloss:
        return False
    normalized = normalize_gloss(gloss)
    if not normalized:
        return False
    # Single character or multi-character with valid format
    if not re.match(r"^[A-Z0-9]([A-Z0-9\-]*[A-Z0-9])?$", normalized):
        return False
    return True


def validate_quality_score(score: int) -> bool:
    """Validate quality score is in valid range (1-5).

    Args:
        score: Quality score to validate

    Returns:
        True if valid
    """
    return 1 <= score <= 5
