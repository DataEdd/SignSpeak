"""Tests for CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.main import app
from packages.database import Sign, SignStatus, VideoInfo


runner = CliRunner()


@pytest.fixture
def mock_store():
    """Create a mock SignStore."""
    store = MagicMock()
    store.count_signs.return_value = {
        "verified": 10,
        "pending": 5,
        "imported": 3,
        "rejected": 2,
    }
    return store


@pytest.fixture
def mock_sign():
    """Create a mock Sign."""
    return Sign(
        gloss="HELLO",
        english=["hello", "hi"],
        category="greeting",
        source="recorded",
        status=SignStatus.VERIFIED,
        quality_score=5,
        verified_by="tester",
        video=VideoInfo(file="video.mp4"),
    )


class TestListCommand:
    """Tests for the list command."""

    def test_list_with_stats(self, mock_store):
        """Test list --stats flag."""
        with patch("src.commands.list.get_search") as mock_get_search:
            search = MagicMock()
            search.get_statistics.return_value = {
                "total_verified": 10,
                "total_pending": 5,
                "categories": {"greeting": 3, "question": 2},
                "sources": {"recorded": 7, "wlasl": 3},
            }
            mock_get_search.return_value = search

            result = runner.invoke(app, ["list", "--stats"])
            assert result.exit_code == 0
            assert "10" in result.output


class TestShowCommand:
    """Tests for the show command."""

    def test_show_existing_sign(self, mock_store, mock_sign):
        """Test showing an existing sign."""
        with patch("src.main.get_store") as mock_get_store:
            mock_get_store.return_value.get_sign.return_value = mock_sign

            result = runner.invoke(app, ["show", "HELLO"])
            assert result.exit_code == 0
            assert "HELLO" in result.output

    def test_show_nonexistent_sign(self, mock_store):
        """Test showing a sign that doesn't exist."""
        with patch("src.main.get_store") as mock_get_store:
            mock_get_store.return_value.get_sign.return_value = None

            result = runner.invoke(app, ["show", "NONEXISTENT"])
            assert result.exit_code == 1
            assert "not found" in result.output


class TestCountCommand:
    """Tests for the count command."""

    def test_count(self, mock_store):
        """Test count command output."""
        with patch("src.main.get_store") as mock_get_store:
            mock_get_store.return_value = mock_store

            result = runner.invoke(app, ["count"])
            assert result.exit_code == 0
            assert "Verified" in result.output
            assert "10" in result.output


class TestVersionFlag:
    """Tests for version flag."""

    def test_version(self):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "SignBridge CLI" in result.output
