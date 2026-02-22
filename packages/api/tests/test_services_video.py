"""Tests for video service."""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from packages.api.services.video_service import VideoService


class TestVideoService:
    """Tests for VideoService class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.temp_signs_dir = tempfile.mkdtemp()
        self.temp_cache_dir = tempfile.mkdtemp()

        with patch("packages.api.services.video_service.ClipManager") as MockClipManager, \
             patch("packages.api.services.video_service.VideoExporter") as MockExporter:
            self.mock_clip_manager_class = MockClipManager
            self.mock_exporter_class = MockExporter
            self.mock_clip_manager = MagicMock()
            self.mock_exporter = MagicMock()
            MockClipManager.return_value = self.mock_clip_manager
            MockExporter.return_value = self.mock_exporter

            self.service = VideoService(
                signs_dir=Path(self.temp_signs_dir),
                cache_dir=Path(self.temp_cache_dir),
            )

    def test_init_creates_directories(self):
        """Test that init creates cache directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "new_cache"

            with patch("packages.api.services.video_service.ClipManager"), \
                 patch("packages.api.services.video_service.VideoExporter"):
                service = VideoService(
                    signs_dir=Path(self.temp_signs_dir),
                    cache_dir=cache_path,
                )

            assert cache_path.exists()

    def test_init_creates_clip_manager(self):
        """Test that init creates ClipManager with correct arguments."""
        with patch("packages.api.services.video_service.ClipManager") as MockClipManager, \
             patch("packages.api.services.video_service.VideoExporter"):
            service = VideoService(
                signs_dir=Path(self.temp_signs_dir),
                cache_dir=Path(self.temp_cache_dir),
            )

            MockClipManager.assert_called_once()
            call_kwargs = MockClipManager.call_args[1]
            assert call_kwargs["signs_dir"] == Path(self.temp_signs_dir)
            assert call_kwargs["search_dirs"] == ["verified", "pending"]

    def test_speed_fps_map_values(self):
        """Test that SPEED_FPS_MAP contains correct speed mappings."""
        assert VideoService.SPEED_FPS_MAP["slow"] == 20
        assert VideoService.SPEED_FPS_MAP["normal"] == 30
        assert VideoService.SPEED_FPS_MAP["fast"] == 40

    def test_format_map_values(self):
        """Test that FORMAT_MAP contains correct format mappings."""
        from packages.video import ExportFormat

        assert VideoService.FORMAT_MAP["mp4"] == ExportFormat.MP4
        assert VideoService.FORMAT_MAP["webm"] == ExportFormat.WEBM
        assert VideoService.FORMAT_MAP["gif"] == ExportFormat.GIF

    @patch("packages.api.services.video_service.Compositor")
    @patch("packages.api.services.video_service.ExportSettings")
    def test_create_video_uses_correct_fps(self, MockSettings, MockCompositor):
        """Test that create_video uses correct fps based on speed setting."""
        mock_compositor = MagicMock()
        mock_video_clip = MagicMock()
        mock_video_clip.frames = []
        mock_compositor.compose.return_value = mock_video_clip
        MockCompositor.return_value = mock_compositor

        self.service.create_video(
            glosses=["HELLO"],
            video_id="test123",
            speed="slow",
        )

        call_kwargs = MockCompositor.call_args[1]
        assert call_kwargs["fps"] == 20

    @patch("packages.api.services.video_service.Compositor")
    @patch("packages.api.services.video_service.ExportSettings")
    def test_create_video_uses_correct_format(self, MockSettings, MockCompositor):
        """Test that create_video uses correct export format."""
        from packages.video import ExportFormat

        mock_compositor = MagicMock()
        mock_video_clip = MagicMock()
        mock_video_clip.frames = []
        mock_compositor.compose.return_value = mock_video_clip
        MockCompositor.return_value = mock_compositor

        self.service.create_video(
            glosses=["HELLO"],
            video_id="test123",
            format="webm",
        )

        call_kwargs = MockSettings.call_args[1]
        assert call_kwargs["format"] == ExportFormat.WEBM

    @patch("packages.api.services.video_service.Compositor")
    @patch("packages.api.services.video_service.ExportSettings")
    def test_create_video_builds_compositor(self, MockSettings, MockCompositor):
        """Test that create_video builds compositor with correct settings."""
        mock_compositor = MagicMock()
        mock_video_clip = MagicMock()
        mock_video_clip.frames = []
        mock_compositor.compose.return_value = mock_video_clip
        MockCompositor.return_value = mock_compositor

        self.service.create_video(
            glosses=["HELLO"],
            video_id="test123",
            speed="normal",
        )

        MockCompositor.assert_called_once()
        call_kwargs = MockCompositor.call_args[1]
        assert call_kwargs["clip_manager"] == self.mock_clip_manager
        assert call_kwargs["fps"] == 30
        assert call_kwargs["resolution"] == (720, 540)

    @patch("packages.api.services.video_service.Compositor")
    @patch("packages.api.services.video_service.ExportSettings")
    def test_create_video_adds_clips(self, MockSettings, MockCompositor):
        """Test that create_video adds all glosses as clips."""
        mock_compositor = MagicMock()
        mock_video_clip = MagicMock()
        mock_video_clip.frames = []
        mock_compositor.compose.return_value = mock_video_clip
        MockCompositor.return_value = mock_compositor

        self.service.create_video(
            glosses=["HELLO", "WORLD", "TEST"],
            video_id="test123",
        )

        assert mock_compositor.add_clip.call_count == 3
        mock_compositor.add_clip.assert_any_call("HELLO")
        mock_compositor.add_clip.assert_any_call("WORLD")
        mock_compositor.add_clip.assert_any_call("TEST")

    @patch("packages.api.services.video_service.Compositor")
    @patch("packages.api.services.video_service.ExportSettings")
    @patch("packages.api.services.video_service.TransitionType")
    def test_create_video_adds_transitions(self, MockTransitionType, MockSettings, MockCompositor):
        """Test that create_video adds transitions between clips."""
        mock_compositor = MagicMock()
        mock_video_clip = MagicMock()
        mock_video_clip.frames = []
        mock_compositor.compose.return_value = mock_video_clip
        MockCompositor.return_value = mock_compositor

        self.service.create_video(
            glosses=["HELLO", "WORLD", "TEST"],
            video_id="test123",
            transition_ms=150,
        )

        # Should have 2 transitions for 3 clips (between each pair)
        assert mock_compositor.add_transition.call_count == 2

    @patch("packages.api.services.video_service.Compositor")
    @patch("packages.api.services.video_service.ExportSettings")
    def test_create_video_returns_path(self, MockSettings, MockCompositor):
        """Test that create_video returns the output path."""
        mock_compositor = MagicMock()
        mock_video_clip = MagicMock()
        mock_video_clip.frames = []
        mock_compositor.compose.return_value = mock_video_clip
        MockCompositor.return_value = mock_compositor

        result = self.service.create_video(
            glosses=["HELLO"],
            video_id="test123",
            format="mp4",
        )

        expected_path = Path(self.temp_cache_dir) / "test123.mp4"
        assert result == expected_path

    def test_get_video_path_mp4_exists(self):
        """Test that get_video_path finds existing mp4 file."""
        video_path = Path(self.temp_cache_dir) / "video123.mp4"
        video_path.touch()

        result = self.service.get_video_path("video123")

        assert result == video_path

    def test_get_video_path_webm_exists(self):
        """Test that get_video_path finds existing webm file."""
        video_path = Path(self.temp_cache_dir) / "video123.webm"
        video_path.touch()

        result = self.service.get_video_path("video123")

        assert result == video_path

    def test_get_video_path_gif_exists(self):
        """Test that get_video_path finds existing gif file."""
        video_path = Path(self.temp_cache_dir) / "video123.gif"
        video_path.touch()

        result = self.service.get_video_path("video123")

        assert result == video_path

    def test_get_video_path_not_found(self):
        """Test that get_video_path returns None when video doesn't exist."""
        result = self.service.get_video_path("nonexistent")

        assert result is None

    def test_delete_video_exists(self):
        """Test that delete_video removes existing video and returns True."""
        video_path = Path(self.temp_cache_dir) / "video123.mp4"
        video_path.touch()

        result = self.service.delete_video("video123")

        assert result is True
        assert not video_path.exists()

    def test_delete_video_not_found(self):
        """Test that delete_video returns False when video doesn't exist."""
        result = self.service.delete_video("nonexistent")

        assert result is False

    def test_cleanup_cache_removes_old_files(self):
        """Test that cleanup_cache removes files older than max_age_hours."""
        # Create an old file
        old_file = Path(self.temp_cache_dir) / "old_video.mp4"
        old_file.touch()
        # Set mtime to 2 days ago
        old_time = time.time() - (48 * 3600)
        import os
        os.utime(old_file, (old_time, old_time))

        # Create a recent file
        new_file = Path(self.temp_cache_dir) / "new_video.mp4"
        new_file.touch()

        self.service.cleanup_cache(max_age_hours=24)

        assert not old_file.exists()
        assert new_file.exists()

    def test_cleanup_cache_returns_count(self):
        """Test that cleanup_cache returns the number of deleted files."""
        # Create old files
        for i in range(3):
            old_file = Path(self.temp_cache_dir) / f"old_video_{i}.mp4"
            old_file.touch()
            old_time = time.time() - (48 * 3600)
            import os
            os.utime(old_file, (old_time, old_time))

        result = self.service.cleanup_cache(max_age_hours=24)

        assert result == 3

    def test_get_cache_stats_counts_files(self):
        """Test that get_cache_stats counts files correctly."""
        # Create test files
        for i in range(5):
            test_file = Path(self.temp_cache_dir) / f"video_{i}.mp4"
            test_file.write_bytes(b"test content")

        result = self.service.get_cache_stats()

        assert result["file_count"] == 5

    def test_get_cache_stats_calculates_size(self):
        """Test that get_cache_stats calculates total size correctly."""
        # Create test files with known sizes
        content = b"x" * 1000  # 1000 bytes each
        for i in range(3):
            test_file = Path(self.temp_cache_dir) / f"video_{i}.mp4"
            test_file.write_bytes(content)

        result = self.service.get_cache_stats()

        assert result["total_size_bytes"] == 3000
        assert result["total_size_mb"] == 0.0  # 3000 bytes is essentially 0 MB when rounded
        assert result["cache_dir"] == str(Path(self.temp_cache_dir))
