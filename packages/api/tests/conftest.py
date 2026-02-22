"""
Shared fixtures for API package tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_sign_service():
    """Create mock SignService."""
    service = MagicMock()

    # Default return values
    service.list_signs.return_value = {
        "signs": [],
        "total": 0,
    }
    service.get_sign.return_value = None
    service.search_signs.return_value = []
    service.get_stats.return_value = {
        "total_signs": 0,
        "verified_signs": 0,
        "pending_signs": 0,
        "imported_signs": 0,
        "rejected_signs": 0,
        "categories": {},
    }

    return service


@pytest.fixture
def mock_translation_service():
    """Create mock TranslationService."""
    service = MagicMock()

    # Default return values
    service.translate_text.return_value = {
        "glosses": ["HELLO", "WORLD"],
        "video_url": "/api/videos/abc123.mp4",
        "confidence": 0.95,
        "quality": "HIGH",
        "missing_signs": [],
        "fingerspelled": [],
    }
    service.get_gloss_preview.return_value = {
        "glosses": ["HELLO", "WORLD"],
        "available_signs": ["HELLO", "WORLD"],
        "missing_signs": [],
        "fingerspelled": [],
        "confidence": 0.95,
        "quality": "HIGH",
    }

    return service


@pytest.fixture
def mock_video_service():
    """Create mock VideoService."""
    service = MagicMock()

    # Default return values
    service.get_video_path.return_value = None
    service.delete_video.return_value = False
    service.get_cache_stats.return_value = {
        "file_count": 0,
        "total_size_bytes": 0,
        "total_size_mb": 0.0,
        "cache_dir": "/tmp/cache",
    }
    service.cleanup_cache.return_value = 0

    return service


@pytest.fixture
def sample_sign_dict():
    """Create sample sign response dictionary."""
    return {
        "gloss": "HELLO",
        "english": ["hello", "hi"],
        "category": "greeting",
        "source": "recorded",
        "status": "verified",
        "quality_score": 5,
        "verified_by": "tester",
        "verified_date": "2024-01-15",
        "video_url": "/api/signs/HELLO/video",
    }


@pytest.fixture
def sample_pending_sign_dict():
    """Create sample pending sign response dictionary."""
    return {
        "gloss": "WORLD",
        "english": ["world", "earth"],
        "category": "noun",
        "source": "imported",
        "status": "pending",
        "quality_score": None,
        "verified_by": None,
        "verified_date": None,
        "video_url": "/api/signs/WORLD/video",
    }


@pytest.fixture
def mock_sign_store():
    """Create mock SignStore."""
    store = MagicMock()
    store.list_signs.return_value = []
    store.get_sign.return_value = None
    store.get_verified_sign.return_value = None
    return store


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_signs_dir():
    """Create temporary signs directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "verified").mkdir()
        (path / "pending").mkdir()
        yield path
