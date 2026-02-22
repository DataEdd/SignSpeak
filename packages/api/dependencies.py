"""FastAPI dependency injection for services."""

import os
from functools import lru_cache
from pathlib import Path

from packages.database import SignStore

from .services import TranslationService, VideoService, SignService


# Configuration from environment
def get_config():
    """Get configuration from environment variables."""
    return {
        "signs_dir": Path(os.getenv("SIGNS_DIR", "data/signs")),
        "cache_dir": Path(os.getenv("VIDEO_CACHE_DIR", "data/cache/videos")),
        "api_host": os.getenv("API_HOST", "0.0.0.0"),
        "api_port": int(os.getenv("API_PORT", "8000")),
        "cors_origins": os.getenv("API_CORS_ORIGINS", "*").split(","),
    }


@lru_cache()
def get_sign_store() -> SignStore:
    """Get or create the sign store singleton."""
    config = get_config()
    return SignStore(base_path=config["signs_dir"])


@lru_cache()
def get_video_service() -> VideoService:
    """Get or create the video service singleton."""
    config = get_config()
    return VideoService(
        signs_dir=config["signs_dir"],
        cache_dir=config["cache_dir"],
    )


@lru_cache()
def get_sign_service() -> SignService:
    """Get or create the sign service singleton."""
    store = get_sign_store()
    return SignService(sign_store=store)


@lru_cache()
def get_translation_service() -> TranslationService:
    """Get or create the translation service singleton."""
    store = get_sign_store()
    video_service = get_video_service()
    return TranslationService(
        sign_store=store,
        video_service=video_service,
    )
