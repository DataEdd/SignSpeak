"""Configuration and path utilities for CLI."""

import os
from pathlib import Path
from functools import lru_cache

from packages.database import SignStore, SignSearch, SignVerifier


def get_signs_dir() -> Path:
    """Get the signs database directory.

    Checks in order:
    1. SIGNBRIDGE_SIGNS_DIR environment variable
    2. data/signs relative to project root
    """
    env_dir = os.environ.get("SIGNBRIDGE_SIGNS_DIR")
    if env_dir:
        return Path(env_dir)

    # Walk up to find project root (contains packages/)
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "packages").exists():
            signs_dir = parent / "data" / "signs"
            signs_dir.mkdir(parents=True, exist_ok=True)
            return signs_dir

    # Fallback to current directory
    return Path.cwd() / "data" / "signs"


@lru_cache
def get_store() -> SignStore:
    """Get a SignStore instance."""
    return SignStore(get_signs_dir())


@lru_cache
def get_search() -> SignSearch:
    """Get a SignSearch instance."""
    return SignSearch(get_store())


@lru_cache
def get_verifier() -> SignVerifier:
    """Get a SignVerifier instance."""
    return SignVerifier(get_store())
