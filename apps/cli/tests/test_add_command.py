"""Tests for the add command."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.main import app
from packages.database import Sign, SignStatus, VideoInfo


runner = CliRunner()


class TestAddCommand:
    """Tests for the add command."""

    def test_add_success(self, mock_store, mock_sign):
        """Test adding a new sign successfully with all required options."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = Path(tmp.name)

        try:
            with patch("src.commands.add.get_store") as mock_get_store:
                mock_get_store.return_value = mock_store
                mock_store.get_sign.return_value = None
                mock_store.add_sign.return_value = mock_sign

                result = runner.invoke(
                    app,
                    ["add", "HELLO", "--video", str(video_path), "--english", "hello,hi"],
                )

                assert result.exit_code == 0
                assert "Added HELLO" in result.output
                mock_store.add_sign.assert_called_once()
        finally:
            video_path.unlink(missing_ok=True)

    def test_add_video_not_found(self, mock_store):
        """Test error when video file doesn't exist."""
        with patch("src.commands.add.get_store") as mock_get_store:
            mock_get_store.return_value = mock_store

            result = runner.invoke(
                app,
                ["add", "HELLO", "--video", "/nonexistent/video.mp4", "--english", "hello"],
            )

            assert result.exit_code == 1
            assert "Video file not found" in result.output
            mock_store.add_sign.assert_not_called()

    def test_add_duplicate_without_force(self, mock_store, mock_sign):
        """Test warning when sign exists and --force is not used."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = Path(tmp.name)

        try:
            with patch("src.commands.add.get_store") as mock_get_store:
                mock_get_store.return_value = mock_store
                mock_store.get_sign.return_value = mock_sign

                # Simulate user declining to overwrite
                result = runner.invoke(
                    app,
                    ["add", "HELLO", "--video", str(video_path), "--english", "hello"],
                    input="n\n",
                )

                assert result.exit_code == 0
                assert "already exists" in result.output
                mock_store.add_sign.assert_not_called()
        finally:
            video_path.unlink(missing_ok=True)

    def test_add_duplicate_with_force(self, mock_store, mock_sign):
        """Test overwrite when --force is used on existing sign."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = Path(tmp.name)

        try:
            new_sign = Sign(
                gloss="HELLO",
                english=["hello"],
                category="",
                source="recorded",
                status=SignStatus.PENDING,
                video=VideoInfo(file="video.mp4"),
            )

            with patch("src.commands.add.get_store") as mock_get_store:
                mock_get_store.return_value = mock_store
                mock_store.get_sign.return_value = mock_sign
                mock_store.add_sign.return_value = new_sign

                result = runner.invoke(
                    app,
                    ["add", "HELLO", "--video", str(video_path), "--english", "hello", "--force"],
                )

                assert result.exit_code == 0
                assert "Added HELLO" in result.output
                mock_store.delete_sign.assert_called_once_with("HELLO")
                mock_store.add_sign.assert_called_once()
        finally:
            video_path.unlink(missing_ok=True)

    def test_add_normalizes_gloss_to_uppercase(self, mock_store, mock_sign):
        """Test that gloss is normalized to uppercase (hello -> HELLO)."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = Path(tmp.name)

        try:
            with patch("src.commands.add.get_store") as mock_get_store:
                mock_get_store.return_value = mock_store
                mock_store.get_sign.return_value = None
                mock_store.add_sign.return_value = mock_sign

                result = runner.invoke(
                    app,
                    ["add", "hello", "--video", str(video_path), "--english", "hello"],
                )

                assert result.exit_code == 0
                # Verify add_sign was called with uppercase gloss
                call_kwargs = mock_store.add_sign.call_args[1]
                assert call_kwargs["gloss"] == "HELLO"
        finally:
            video_path.unlink(missing_ok=True)

    def test_add_parses_english_list(self, mock_store, mock_sign):
        """Test that comma-separated English list is parsed correctly."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = Path(tmp.name)

        try:
            with patch("src.commands.add.get_store") as mock_get_store:
                mock_get_store.return_value = mock_store
                mock_store.get_sign.return_value = None
                mock_store.add_sign.return_value = mock_sign

                result = runner.invoke(
                    app,
                    ["add", "HELLO", "--video", str(video_path), "--english", "hello,hi,hey"],
                )

                assert result.exit_code == 0
                call_kwargs = mock_store.add_sign.call_args[1]
                assert call_kwargs["english"] == ["hello", "hi", "hey"]
        finally:
            video_path.unlink(missing_ok=True)

    def test_add_with_category(self, mock_store, mock_sign):
        """Test that --category option is passed correctly."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = Path(tmp.name)

        try:
            with patch("src.commands.add.get_store") as mock_get_store:
                mock_get_store.return_value = mock_store
                mock_store.get_sign.return_value = None
                mock_store.add_sign.return_value = mock_sign

                result = runner.invoke(
                    app,
                    [
                        "add", "HELLO",
                        "--video", str(video_path),
                        "--english", "hello",
                        "--category", "greeting",
                    ],
                )

                assert result.exit_code == 0
                call_kwargs = mock_store.add_sign.call_args[1]
                assert call_kwargs["category"] == "greeting"
        finally:
            video_path.unlink(missing_ok=True)

    def test_add_with_source(self, mock_store, mock_sign):
        """Test that --source option is passed correctly."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = Path(tmp.name)

        try:
            with patch("src.commands.add.get_store") as mock_get_store:
                mock_get_store.return_value = mock_store
                mock_store.get_sign.return_value = None
                mock_store.add_sign.return_value = mock_sign

                result = runner.invoke(
                    app,
                    [
                        "add", "HELLO",
                        "--video", str(video_path),
                        "--english", "hello",
                        "--source", "wlasl",
                    ],
                )

                assert result.exit_code == 0
                call_kwargs = mock_store.add_sign.call_args[1]
                assert call_kwargs["source"] == "wlasl"
        finally:
            video_path.unlink(missing_ok=True)

    def test_add_error_handling_file_not_found(self, mock_store):
        """Test error handling when add_sign raises FileNotFoundError."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = Path(tmp.name)

        try:
            with patch("src.commands.add.get_store") as mock_get_store:
                mock_get_store.return_value = mock_store
                mock_store.get_sign.return_value = None
                mock_store.add_sign.side_effect = FileNotFoundError("Video file not found")

                result = runner.invoke(
                    app,
                    ["add", "HELLO", "--video", str(video_path), "--english", "hello"],
                )

                assert result.exit_code == 1
                assert "Error" in result.output
        finally:
            video_path.unlink(missing_ok=True)

    def test_add_error_handling_value_error(self, mock_store):
        """Test error handling when add_sign raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = Path(tmp.name)

        try:
            with patch("src.commands.add.get_store") as mock_get_store:
                mock_get_store.return_value = mock_store
                mock_store.get_sign.return_value = None
                mock_store.add_sign.side_effect = ValueError("Invalid sign data")

                result = runner.invoke(
                    app,
                    ["add", "HELLO", "--video", str(video_path), "--english", "hello"],
                )

                assert result.exit_code == 1
                assert "Error" in result.output
        finally:
            video_path.unlink(missing_ok=True)

    def test_add_default_source_is_recorded(self, mock_store, mock_sign):
        """Test that default source is 'recorded' when not specified."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = Path(tmp.name)

        try:
            with patch("src.commands.add.get_store") as mock_get_store:
                mock_get_store.return_value = mock_store
                mock_store.get_sign.return_value = None
                mock_store.add_sign.return_value = mock_sign

                result = runner.invoke(
                    app,
                    ["add", "HELLO", "--video", str(video_path), "--english", "hello"],
                )

                assert result.exit_code == 0
                call_kwargs = mock_store.add_sign.call_args[1]
                assert call_kwargs["source"] == "recorded"
        finally:
            video_path.unlink(missing_ok=True)

    def test_add_strips_whitespace_from_english(self, mock_store, mock_sign):
        """Test that whitespace is stripped from English translations."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = Path(tmp.name)

        try:
            with patch("src.commands.add.get_store") as mock_get_store:
                mock_get_store.return_value = mock_store
                mock_store.get_sign.return_value = None
                mock_store.add_sign.return_value = mock_sign

                result = runner.invoke(
                    app,
                    ["add", "HELLO", "--video", str(video_path), "--english", " hello , hi , hey "],
                )

                assert result.exit_code == 0
                call_kwargs = mock_store.add_sign.call_args[1]
                assert call_kwargs["english"] == ["hello", "hi", "hey"]
        finally:
            video_path.unlink(missing_ok=True)

    def test_add_filters_empty_english_entries(self, mock_store, mock_sign):
        """Test that empty entries are filtered from English translations."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = Path(tmp.name)

        try:
            with patch("src.commands.add.get_store") as mock_get_store:
                mock_get_store.return_value = mock_store
                mock_store.get_sign.return_value = None
                mock_store.add_sign.return_value = mock_sign

                result = runner.invoke(
                    app,
                    ["add", "HELLO", "--video", str(video_path), "--english", "hello,,hi,"],
                )

                assert result.exit_code == 0
                call_kwargs = mock_store.add_sign.call_args[1]
                assert call_kwargs["english"] == ["hello", "hi"]
        finally:
            video_path.unlink(missing_ok=True)
