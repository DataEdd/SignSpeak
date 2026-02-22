"""Shared fixtures for CLI tests."""

import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock

from packages.database import Sign, SignStatus, VideoInfo


@pytest.fixture
def cli_runner():
    """Create a CliRunner instance for testing CLI commands."""
    return CliRunner()


@pytest.fixture
def mock_store():
    """Create a mock SignStore with default behaviors."""
    store = MagicMock()
    store.count_signs.return_value = {
        "verified": 10,
        "pending": 5,
        "imported": 3,
        "rejected": 2,
    }
    store.get_sign.return_value = None
    store.get_verified_sign.return_value = None
    store.add_sign.return_value = None
    store.delete_sign.return_value = True
    store.list_verified.return_value = []
    store.list_signs.return_value = []
    return store


@pytest.fixture
def mock_sign():
    """Create a mock Sign with verified status."""
    return Sign(
        gloss="HELLO",
        english=["hello", "hi", "hey"],
        category="greeting",
        source="recorded",
        status=SignStatus.VERIFIED,
        quality_score=5,
        verified_by="tester",
        video=VideoInfo(file="video.mp4", fps=30, duration_ms=1200),
    )


@pytest.fixture
def mock_pending_sign():
    """Create a mock Sign with pending status."""
    return Sign(
        gloss="WORLD",
        english=["world", "earth"],
        category="noun",
        source="recorded",
        status=SignStatus.PENDING,
        quality_score=None,
        verified_by=None,
        video=VideoInfo(file="video.mp4", fps=30, duration_ms=1000),
    )


@pytest.fixture
def mock_imported_sign():
    """Create a mock Sign with imported status."""
    return Sign(
        gloss="THANK-YOU",
        english=["thank you", "thanks"],
        category="greeting",
        source="wlasl",
        status=SignStatus.IMPORTED,
        quality_score=None,
        verified_by=None,
        video=VideoInfo(file="video.mp4", fps=30, duration_ms=800),
    )


@pytest.fixture
def mock_verifier():
    """Create a mock Verifier with default behaviors."""
    verifier = MagicMock()
    verifier.verify.return_value = None
    verifier.reject.return_value = None
    verifier.get_verification_queue.return_value = []
    verifier.get_verification_stats.return_value = {
        "pending_review": 5,
        "verified": 10,
        "rejected": 2,
        "approval_rate": 83.3,
    }

    # Mock quality check result
    quality_result = MagicMock()
    quality_result.passed = True
    quality_result.score = 4
    quality_result.issues = []
    quality_result.suggestions = []
    verifier.check_sign_quality.return_value = quality_result

    return verifier


@pytest.fixture
def mock_importer():
    """Create a mock Importer with default behaviors."""
    importer = MagicMock()
    importer.import_sign.return_value = None
    importer.import_all.return_value = (5, [])
    importer.list_available.return_value = ["HELLO", "WORLD", "THANK-YOU"]
    return importer


@pytest.fixture
def mock_translation_result():
    """Create a mock translation result."""
    result = MagicMock()
    result.glosses = ["HELLO", "HOW", "YOU"]

    validation = MagicMock()
    validation.coverage = 0.75
    validation.missing = ["HOW"]
    validation.fingerspelled = []
    result.validation = validation

    return result
