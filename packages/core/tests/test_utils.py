"""Tests for utility functions."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from packages.core.utils import (
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


class TestNormalizeGloss:
    """Tests for normalize_gloss function."""

    def test_uppercase(self):
        """Test conversion to uppercase."""
        assert normalize_gloss("hello") == "HELLO"
        assert normalize_gloss("Hello") == "HELLO"
        assert normalize_gloss("HELLO") == "HELLO"

    def test_strip_whitespace(self):
        """Test stripping leading/trailing whitespace."""
        assert normalize_gloss("  hello  ") == "HELLO"
        assert normalize_gloss("\thello\n") == "HELLO"

    def test_internal_spaces_to_hyphens(self):
        """Test converting internal spaces to hyphens."""
        assert normalize_gloss("thank you") == "THANK-YOU"
        assert normalize_gloss("how are you") == "HOW-ARE-YOU"

    def test_multiple_spaces(self):
        """Test handling multiple spaces."""
        assert normalize_gloss("thank   you") == "THANK-YOU"

    def test_empty_string(self):
        """Test empty string handling."""
        assert normalize_gloss("") == ""
        assert normalize_gloss("   ") == ""


class TestGlossToPathSafe:
    """Tests for gloss_to_path_safe function."""

    def test_basic(self):
        """Test basic conversion."""
        assert gloss_to_path_safe("hello") == "HELLO"
        assert gloss_to_path_safe("HELLO") == "HELLO"

    def test_spaces(self):
        """Test handling spaces."""
        assert gloss_to_path_safe("thank you") == "THANK-YOU"

    def test_special_characters(self):
        """Test handling special characters."""
        assert gloss_to_path_safe("don't") == "DON_T"
        assert gloss_to_path_safe("test/path") == "TEST_PATH"
        assert gloss_to_path_safe("a*b?c") == "A_B_C"


class TestTimeUtilities:
    """Tests for time utility functions."""

    def test_now_iso(self):
        """Test now_iso returns valid ISO format."""
        result = now_iso()

        assert result.endswith("Z")
        # Should be parseable
        parsed = parse_iso(result)
        assert isinstance(parsed, datetime)

    def test_parse_iso_with_z(self):
        """Test parsing ISO string with Z suffix."""
        result = parse_iso("2024-01-15T10:30:00Z")

        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 0

    def test_parse_iso_with_offset(self):
        """Test parsing ISO string with offset."""
        result = parse_iso("2024-01-15T10:30:00+00:00")

        assert result.year == 2024
        assert result.hour == 10

    def test_ms_to_frames(self):
        """Test milliseconds to frames conversion."""
        # 1000ms at 30fps = 30 frames
        assert ms_to_frames(1000, 30) == 30

        # 500ms at 30fps = 15 frames
        assert ms_to_frames(500, 30) == 15

        # Very short duration still returns at least 1
        assert ms_to_frames(1, 30) == 1

        # Zero returns 1 (minimum)
        assert ms_to_frames(0, 30) == 1

    def test_frames_to_ms(self):
        """Test frames to milliseconds conversion."""
        # 30 frames at 30fps = 1000ms
        assert frames_to_ms(30, 30) == 1000.0

        # 15 frames at 30fps = 500ms
        assert frames_to_ms(15, 30) == 500.0

        # 60 frames at 60fps = 1000ms
        assert frames_to_ms(60, 60) == 1000.0

    def test_format_duration(self):
        """Test duration formatting."""
        assert format_duration(500) == "0.5s"
        assert format_duration(1500) == "1.5s"
        assert format_duration(30000) == "30.0s"
        assert format_duration(60000) == "1m 0s"
        assert format_duration(90000) == "1m 30s"
        assert format_duration(150000) == "2m 30s"


class TestFileUtilities:
    """Tests for file utility functions."""

    def test_ensure_dir(self):
        """Test ensure_dir creates directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a" / "b" / "c"

            result = ensure_dir(path)

            assert path.exists()
            assert path.is_dir()
            assert result == path

    def test_ensure_dir_existing(self):
        """Test ensure_dir with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            result = ensure_dir(path)

            assert path.exists()
            assert result == path

    def test_compute_file_hash(self):
        """Test computing file hash."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            path = Path(f.name)

        try:
            hash1 = compute_file_hash(path)
            hash2 = compute_file_hash(path)

            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 produces 64 hex chars
        finally:
            path.unlink()

    def test_compute_file_hash_different_content(self):
        """Test that different content produces different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "file1"
            path2 = Path(tmpdir) / "file2"

            path1.write_bytes(b"content 1")
            path2.write_bytes(b"content 2")

            hash1 = compute_file_hash(path1)
            hash2 = compute_file_hash(path2)

            assert hash1 != hash2

    def test_safe_json_load(self):
        """Test safe JSON loading."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            path = Path(f.name)

        try:
            result = safe_json_load(path)

            assert result == {"key": "value"}
        finally:
            path.unlink()

    def test_safe_json_load_missing_file(self):
        """Test safe_json_load with missing file."""
        path = Path("/nonexistent/file.json")

        result = safe_json_load(path, default={"default": True})

        assert result == {"default": True}

    def test_safe_json_load_invalid_json(self):
        """Test safe_json_load with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            path = Path(f.name)

        try:
            result = safe_json_load(path, default=[])

            assert result == []
        finally:
            path.unlink()

    def test_safe_json_save(self):
        """Test safe JSON saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.json"

            result = safe_json_save(path, {"key": "value"})

            assert result is True
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data == {"key": "value"}

    def test_safe_json_save_creates_dirs(self):
        """Test safe_json_save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a" / "b" / "output.json"

            result = safe_json_save(path, {"nested": True})

            assert result is True
            assert path.exists()


class TestValidationUtilities:
    """Tests for validation utility functions."""

    def test_is_valid_gloss_valid(self):
        """Test valid glosses."""
        assert is_valid_gloss("HELLO") is True
        assert is_valid_gloss("hello") is True
        assert is_valid_gloss("A") is True
        assert is_valid_gloss("THANK-YOU") is True
        assert is_valid_gloss("SIGN-1") is True
        assert is_valid_gloss("ABC123") is True

    def test_is_valid_gloss_invalid(self):
        """Test invalid glosses."""
        assert is_valid_gloss("") is False
        assert is_valid_gloss("   ") is False
        assert is_valid_gloss("-HELLO") is False
        assert is_valid_gloss("HELLO-") is False
        assert is_valid_gloss("-") is False

    def test_validate_quality_score(self):
        """Test quality score validation."""
        assert validate_quality_score(1) is True
        assert validate_quality_score(3) is True
        assert validate_quality_score(5) is True
        assert validate_quality_score(0) is False
        assert validate_quality_score(6) is False
        assert validate_quality_score(-1) is False
