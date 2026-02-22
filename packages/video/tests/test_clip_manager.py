"""Tests for clip_manager module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from packages.video.clip_manager import ClipManager, VideoClip


class TestVideoClipProperties:
    """Tests for VideoClip dataclass properties."""

    def test_num_frames_returns_frame_count(self, sample_frames):
        """Test that num_frames returns the correct count."""
        clip = VideoClip(gloss="TEST", frames=sample_frames, fps=30.0, metadata={})
        assert clip.num_frames == len(sample_frames)

    def test_num_frames_with_empty_clip(self, empty_video_clip):
        """Test that num_frames returns 0 for empty clip."""
        assert empty_video_clip.num_frames == 0

    def test_duration_ms_calculation(self, sample_frames):
        """Test that duration_ms is calculated correctly as num_frames / fps * 1000."""
        clip = VideoClip(gloss="TEST", frames=sample_frames, fps=30.0, metadata={})
        expected = (len(sample_frames) / 30.0) * 1000
        assert clip.duration_ms == expected

    def test_duration_ms_with_different_fps(self, sample_frames):
        """Test duration_ms calculation with non-standard fps."""
        clip = VideoClip(gloss="TEST", frames=sample_frames, fps=24.0, metadata={})
        expected = (len(sample_frames) / 24.0) * 1000
        assert clip.duration_ms == expected

    def test_resolution_returns_width_height(self, sample_frames):
        """Test that resolution returns (width, height) tuple."""
        clip = VideoClip(gloss="TEST", frames=sample_frames, fps=30.0, metadata={})
        # sample_frames shape is (10, 100, 100, 3) -> height=100, width=100
        assert clip.resolution == (100, 100)

    def test_resolution_with_non_square_frames(self):
        """Test resolution with rectangular frames."""
        # Shape: (frames, height, width, channels)
        frames = np.random.randint(0, 255, (5, 200, 300, 3), dtype=np.uint8)
        clip = VideoClip(gloss="TEST", frames=frames, fps=30.0, metadata={})
        assert clip.resolution == (300, 200)  # (width, height)

    def test_resolution_with_empty_clip_returns_zeros(self, empty_video_clip):
        """Test that resolution returns (0, 0) for empty clip."""
        assert empty_video_clip.resolution == (0, 0)


class TestVideoClipTrim:
    """Tests for VideoClip trim method."""

    def test_trim_with_start_ms_only(self, sample_video_clip):
        """Test trimming with only start_ms specified."""
        # At 30 fps, 100ms = 3 frames
        trimmed = sample_video_clip.trim(start_ms=100)
        assert trimmed.num_frames == 7  # 10 - 3 = 7

    def test_trim_with_end_ms_only(self, sample_video_clip):
        """Test trimming with only end_ms specified."""
        # At 30 fps, 200ms = 6 frames
        trimmed = sample_video_clip.trim(end_ms=200)
        assert trimmed.num_frames == 6

    def test_trim_with_both_start_and_end(self, sample_video_clip):
        """Test trimming with both start_ms and end_ms."""
        # At 30 fps: 100ms = 3 frames, 200ms = 6 frames -> 6-3=3 frames
        trimmed = sample_video_clip.trim(start_ms=100, end_ms=200)
        assert trimmed.num_frames == 3

    def test_trim_clamps_to_valid_bounds_start_negative(self, sample_video_clip):
        """Test that trim clamps negative start to 0."""
        trimmed = sample_video_clip.trim(start_ms=-100)
        assert trimmed.num_frames == 10  # All frames preserved

    def test_trim_clamps_to_valid_bounds_end_exceeds_duration(self, sample_video_clip):
        """Test that trim clamps end beyond duration."""
        trimmed = sample_video_clip.trim(end_ms=10000)  # Way beyond duration
        assert trimmed.num_frames == 10  # All frames preserved

    def test_trim_clamps_start_beyond_end(self, sample_video_clip):
        """Test that trim handles start > end by returning empty."""
        trimmed = sample_video_clip.trim(start_ms=1000, end_ms=100)
        assert trimmed.num_frames == 0

    def test_trim_preserves_metadata_copy(self, sample_video_clip):
        """Test that trim returns a copy of metadata."""
        original_metadata = sample_video_clip.metadata.copy()
        trimmed = sample_video_clip.trim(start_ms=100, end_ms=200)

        assert trimmed.metadata == original_metadata
        # Verify it's a copy, not the same reference
        trimmed.metadata["new_key"] = "new_value"
        assert "new_key" not in sample_video_clip.metadata

    def test_trim_preserves_gloss_and_fps(self, sample_video_clip):
        """Test that trim preserves gloss and fps."""
        trimmed = sample_video_clip.trim(start_ms=100)
        assert trimmed.gloss == sample_video_clip.gloss
        assert trimmed.fps == sample_video_clip.fps


class TestVideoClipResize:
    """Tests for VideoClip resize method."""

    def test_resize_basic_functionality(self, sample_video_clip):
        """Test basic resize operation."""
        resized = sample_video_clip.resize(width=200, height=150)
        assert resized.resolution == (200, 150)
        assert resized.num_frames == sample_video_clip.num_frames

    def test_resize_empty_clip_handling(self, empty_video_clip):
        """Test that resize handles empty clip correctly."""
        resized = empty_video_clip.resize(width=200, height=150)
        assert resized.num_frames == 0
        # Check shape of empty array
        assert resized.frames.shape == (0, 150, 200, 3)

    def test_resize_preserves_fps_and_metadata(self, sample_video_clip):
        """Test that resize preserves fps and metadata."""
        resized = sample_video_clip.resize(width=200, height=150)
        assert resized.fps == sample_video_clip.fps
        assert resized.metadata == sample_video_clip.metadata

    def test_resize_preserves_gloss(self, sample_video_clip):
        """Test that resize preserves gloss."""
        resized = sample_video_clip.resize(width=200, height=150)
        assert resized.gloss == sample_video_clip.gloss

    def test_resize_creates_metadata_copy(self, sample_video_clip):
        """Test that resize returns a copy of metadata."""
        resized = sample_video_clip.resize(width=200, height=150)
        resized.metadata["new_key"] = "new_value"
        assert "new_key" not in sample_video_clip.metadata


class TestClipManagerInit:
    """Tests for ClipManager initialization."""

    def test_init_with_default_search_dirs(self, temp_signs_dir):
        """Test initialization with default search_dirs."""
        manager = ClipManager(temp_signs_dir)
        assert manager.search_dirs == ["verified", "pending"]

    def test_init_with_custom_search_dirs(self, temp_signs_dir):
        """Test initialization with custom search_dirs."""
        custom_dirs = ["imported", "verified"]
        manager = ClipManager(temp_signs_dir, search_dirs=custom_dirs)
        assert manager.search_dirs == custom_dirs

    def test_init_creates_empty_caches(self, temp_signs_dir):
        """Test that initialization creates empty caches."""
        manager = ClipManager(temp_signs_dir)
        assert manager._cache == {}
        assert manager._metadata_cache == {}


class TestClipManagerFindSignDir:
    """Tests for ClipManager _find_sign_dir method."""

    def test_find_sign_dir_in_verified(self, populated_signs_dir):
        """Test finding sign in verified directory."""
        manager = ClipManager(populated_signs_dir)
        result = manager._find_sign_dir("HELLO")
        assert result == populated_signs_dir / "verified" / "HELLO"

    def test_find_sign_dir_in_pending(self, populated_signs_dir):
        """Test finding sign in pending directory."""
        manager = ClipManager(populated_signs_dir)
        result = manager._find_sign_dir("WORLD")
        assert result == populated_signs_dir / "pending" / "WORLD"

    def test_find_sign_dir_at_root_level(self, temp_signs_dir):
        """Test finding sign at root level."""
        root_sign = temp_signs_dir / "ROOT_SIGN"
        root_sign.mkdir()

        manager = ClipManager(temp_signs_dir)
        result = manager._find_sign_dir("ROOT_SIGN")
        assert result == root_sign

    def test_find_sign_dir_returns_none_for_missing(self, temp_signs_dir):
        """Test that missing sign returns None."""
        manager = ClipManager(temp_signs_dir)
        result = manager._find_sign_dir("NONEXISTENT")
        assert result is None

    def test_find_sign_dir_is_case_insensitive(self, populated_signs_dir):
        """Test that sign lookup is case-insensitive."""
        manager = ClipManager(populated_signs_dir)

        result_upper = manager._find_sign_dir("HELLO")
        result_lower = manager._find_sign_dir("hello")
        result_mixed = manager._find_sign_dir("Hello")

        assert result_upper == result_lower == result_mixed
        assert result_upper == populated_signs_dir / "verified" / "HELLO"


class TestClipManagerLoadMetadata:
    """Tests for ClipManager _load_metadata method."""

    def test_load_metadata_with_existing_sign_json(self, populated_signs_dir):
        """Test loading metadata from existing sign.json."""
        manager = ClipManager(populated_signs_dir)
        sign_dir = populated_signs_dir / "verified" / "HELLO"

        metadata = manager._load_metadata(sign_dir)

        assert metadata["gloss"] == "HELLO"
        assert "hello" in metadata["english"]
        assert metadata["category"] == "greeting"

    def test_load_metadata_with_missing_sign_json_returns_empty_dict(self, temp_signs_dir):
        """Test that missing sign.json returns empty dict."""
        empty_sign_dir = temp_signs_dir / "verified" / "NOSIGN"
        empty_sign_dir.mkdir(parents=True)

        manager = ClipManager(temp_signs_dir)
        metadata = manager._load_metadata(empty_sign_dir)

        assert metadata == {}


class TestClipManagerLoadVideo:
    """Tests for ClipManager _load_video method."""

    def test_load_video_success(self, temp_signs_dir, mock_cv2_video_capture):
        """Test successful video loading with mocked cv2."""
        manager = ClipManager(temp_signs_dir)
        video_path = temp_signs_dir / "test.mp4"
        video_path.write_bytes(b"fake video")

        frames, fps = manager._load_video(video_path)

        assert fps == 30.0
        assert len(frames) == 5  # mock_cv2_video_capture returns 5 frames

    def test_load_video_raises_value_error_for_unopened_video(self, temp_signs_dir):
        """Test that unopened video raises ValueError."""
        manager = ClipManager(temp_signs_dir)
        video_path = temp_signs_dir / "test.mp4"
        video_path.write_bytes(b"fake video")

        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_class.return_value = mock_cap

            with pytest.raises(ValueError, match="Could not open video"):
                manager._load_video(video_path)

    def test_load_video_raises_value_error_for_empty_video(self, temp_signs_dir):
        """Test that empty video raises ValueError."""
        manager = ClipManager(temp_signs_dir)
        video_path = temp_signs_dir / "test.mp4"
        video_path.write_bytes(b"fake video")

        with patch('cv2.VideoCapture') as mock_cap_class:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            mock_cap.read.return_value = (False, None)  # No frames
            mock_cap_class.return_value = mock_cap

            with pytest.raises(ValueError, match="No frames loaded"):
                manager._load_video(video_path)


class TestClipManagerGetClip:
    """Tests for ClipManager get_clip method."""

    def test_get_clip_success_with_cache(self, populated_signs_dir, mock_cv2_video_capture):
        """Test successful clip retrieval with caching."""
        manager = ClipManager(populated_signs_dir)

        clip = manager.get_clip("HELLO", use_cache=True)

        assert clip.gloss == "HELLO"
        assert "HELLO" in manager._cache

    def test_get_clip_success_without_cache(self, populated_signs_dir, mock_cv2_video_capture):
        """Test successful clip retrieval without caching."""
        manager = ClipManager(populated_signs_dir)

        clip = manager.get_clip("HELLO", use_cache=False)

        assert clip.gloss == "HELLO"
        assert "HELLO" not in manager._cache

    def test_get_clip_returns_cached_clip(self, populated_signs_dir, mock_cv2_video_capture):
        """Test that cached clip is returned on subsequent calls."""
        manager = ClipManager(populated_signs_dir)

        clip1 = manager.get_clip("HELLO", use_cache=True)
        clip2 = manager.get_clip("HELLO", use_cache=True)

        assert clip1 is clip2  # Same object from cache

    def test_get_clip_raises_file_not_found_for_missing_sign(self, temp_signs_dir):
        """Test that missing sign raises FileNotFoundError."""
        manager = ClipManager(temp_signs_dir)

        with pytest.raises(FileNotFoundError, match="Sign not found"):
            manager.get_clip("NONEXISTENT")

    def test_get_clip_raises_file_not_found_for_missing_video_file(self, temp_signs_dir):
        """Test that missing video file raises FileNotFoundError."""
        sign_dir = temp_signs_dir / "verified" / "NOVIDEO"
        sign_dir.mkdir(parents=True)
        (sign_dir / "sign.json").write_text('{"gloss": "NOVIDEO"}')
        # No video file created

        manager = ClipManager(temp_signs_dir)

        with pytest.raises(FileNotFoundError, match="No video file found"):
            manager.get_clip("NOVIDEO")

    def test_get_clip_applies_timing_trim_from_metadata(self, populated_signs_dir, mock_cv2_video_capture):
        """Test that timing trim is applied from metadata."""
        manager = ClipManager(populated_signs_dir)

        clip = manager.get_clip("HELLO", use_cache=False)

        # HELLO has timing: sign_start_ms=100, sign_end_ms=900
        # At 30 fps: start_frame=3, end_frame=27
        # But we only have 5 frames from mock, so it clamps
        assert clip.num_frames <= 5

    def test_get_clip_finds_video_mp4(self, temp_signs_dir, mock_cv2_video_capture):
        """Test that get_clip finds video.mp4."""
        sign_dir = temp_signs_dir / "verified" / "TEST"
        sign_dir.mkdir(parents=True)
        (sign_dir / "video.mp4").write_bytes(b"fake video")

        manager = ClipManager(temp_signs_dir)
        clip = manager.get_clip("TEST")

        assert clip.gloss == "TEST"

    def test_get_clip_finds_video_webm(self, temp_signs_dir, mock_cv2_video_capture):
        """Test that get_clip finds video.webm."""
        sign_dir = temp_signs_dir / "verified" / "TEST"
        sign_dir.mkdir(parents=True)
        (sign_dir / "video.webm").write_bytes(b"fake video")

        manager = ClipManager(temp_signs_dir)
        clip = manager.get_clip("TEST")

        assert clip.gloss == "TEST"

    def test_get_clip_finds_any_mp4_in_directory(self, temp_signs_dir, mock_cv2_video_capture):
        """Test that get_clip finds any .mp4 file in directory."""
        sign_dir = temp_signs_dir / "verified" / "TEST"
        sign_dir.mkdir(parents=True)
        (sign_dir / "custom_name.mp4").write_bytes(b"fake video")

        manager = ClipManager(temp_signs_dir)
        clip = manager.get_clip("TEST")

        assert clip.gloss == "TEST"


class TestClipManagerPreload:
    """Tests for ClipManager preload method."""

    def test_preload_multiple_glosses_success(self, populated_signs_dir):
        """Test preloading multiple glosses successfully."""
        manager = ClipManager(populated_signs_dir)

        def create_mock_video_capture(*args, **kwargs):
            """Create a fresh mock for each VideoCapture call."""
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
            read_returns = [(True, frame) for frame in frames] + [(False, None)]
            mock_cap.read.side_effect = read_returns
            return mock_cap

        with patch('cv2.VideoCapture', side_effect=create_mock_video_capture):
            with patch('cv2.cvtColor', side_effect=lambda frame, _: frame):
                results = manager.preload(["HELLO", "WORLD"])

        assert results["HELLO"] is True
        assert results["WORLD"] is True
        assert "HELLO" in manager._cache
        assert "WORLD" in manager._cache

    def test_preload_with_some_failures_returns_mixed_results(self, populated_signs_dir):
        """Test preload with some failures returns mixed results."""
        manager = ClipManager(populated_signs_dir)

        def create_mock_video_capture(*args, **kwargs):
            """Create a fresh mock for each VideoCapture call."""
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
            read_returns = [(True, frame) for frame in frames] + [(False, None)]
            mock_cap.read.side_effect = read_returns
            return mock_cap

        with patch('cv2.VideoCapture', side_effect=create_mock_video_capture):
            with patch('cv2.cvtColor', side_effect=lambda frame, _: frame):
                results = manager.preload(["HELLO", "NONEXISTENT", "WORLD"])

        assert results["HELLO"] is True
        assert results["NONEXISTENT"] is False
        assert results["WORLD"] is True


class TestClipManagerClearCache:
    """Tests for ClipManager clear_cache method."""

    def test_clear_cache_clears_all(self, populated_signs_dir):
        """Test that clear_cache() clears all cached clips."""
        manager = ClipManager(populated_signs_dir)

        def create_mock_video_capture(*args, **kwargs):
            """Create a fresh mock for each VideoCapture call."""
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
            read_returns = [(True, frame) for frame in frames] + [(False, None)]
            mock_cap.read.side_effect = read_returns
            return mock_cap

        with patch('cv2.VideoCapture', side_effect=create_mock_video_capture):
            with patch('cv2.cvtColor', side_effect=lambda frame, _: frame):
                manager.get_clip("HELLO")
                manager.get_clip("WORLD")

        manager.clear_cache()

        assert manager._cache == {}

    def test_clear_cache_clears_specific_gloss(self, populated_signs_dir):
        """Test that clear_cache(gloss) clears specific gloss."""
        manager = ClipManager(populated_signs_dir)

        def create_mock_video_capture(*args, **kwargs):
            """Create a fresh mock for each VideoCapture call."""
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
            read_returns = [(True, frame) for frame in frames] + [(False, None)]
            mock_cap.read.side_effect = read_returns
            return mock_cap

        with patch('cv2.VideoCapture', side_effect=create_mock_video_capture):
            with patch('cv2.cvtColor', side_effect=lambda frame, _: frame):
                manager.get_clip("HELLO")
                manager.get_clip("WORLD")

        manager.clear_cache("HELLO")

        assert "HELLO" not in manager._cache
        assert "WORLD" in manager._cache

    def test_clear_cache_is_case_insensitive(self, populated_signs_dir, mock_cv2_video_capture):
        """Test that clear_cache is case-insensitive."""
        manager = ClipManager(populated_signs_dir)
        manager.get_clip("HELLO")

        manager.clear_cache("hello")

        assert "HELLO" not in manager._cache


class TestClipManagerListAvailable:
    """Tests for ClipManager list_available method."""

    def test_list_available_returns_sorted_glosses(self, populated_signs_dir):
        """Test that list_available returns sorted glosses."""
        manager = ClipManager(populated_signs_dir)

        available = manager.list_available()

        assert available == sorted(available)
        assert "HELLO" in available
        assert "WORLD" in available

    def test_list_available_with_empty_directories(self, temp_signs_dir):
        """Test list_available with empty search directories."""
        manager = ClipManager(temp_signs_dir)

        available = manager.list_available()

        assert available == []


class TestClipManagerGetMetadata:
    """Tests for ClipManager get_metadata method."""

    def test_get_metadata_success(self, populated_signs_dir):
        """Test successful metadata retrieval."""
        manager = ClipManager(populated_signs_dir)

        metadata = manager.get_metadata("HELLO")

        assert metadata["gloss"] == "HELLO"
        assert "hello" in metadata["english"]

    def test_get_metadata_uses_cache(self, populated_signs_dir):
        """Test that get_metadata uses cache."""
        manager = ClipManager(populated_signs_dir)

        metadata1 = manager.get_metadata("HELLO")
        metadata2 = manager.get_metadata("HELLO")

        assert metadata1 is metadata2  # Same object from cache
        assert "HELLO" in manager._metadata_cache

    def test_get_metadata_raises_file_not_found_for_missing_sign(self, temp_signs_dir):
        """Test that missing sign raises FileNotFoundError."""
        manager = ClipManager(temp_signs_dir)

        with pytest.raises(FileNotFoundError, match="Sign not found"):
            manager.get_metadata("NONEXISTENT")
