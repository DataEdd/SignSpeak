"""Tests for videos routes."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from packages.api.main import app
from packages.api.dependencies import get_video_service


class TestGetVideoEndpoint:
    """Tests for GET /api/videos/{video_id} endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_video_service, temp_cache_dir):
        """Set up test fixtures."""
        self.mock_service = mock_video_service
        self.temp_dir = temp_cache_dir

        app.dependency_overrides[get_video_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_get_video_mp4_success(self):
        """Test successful retrieval of an MP4 video."""
        # Create a temp file to return
        video_path = self.temp_dir / "test123.mp4"
        video_path.write_bytes(b"fake mp4 content")
        self.mock_service.get_video_path.return_value = video_path

        response = self.client.get("/api/videos/test123")

        assert response.status_code == 200
        assert response.content == b"fake mp4 content"

    def test_get_video_webm_success(self):
        """Test successful retrieval of a WebM video."""
        video_path = self.temp_dir / "test456.webm"
        video_path.write_bytes(b"fake webm content")
        self.mock_service.get_video_path.return_value = video_path

        response = self.client.get("/api/videos/test456")

        assert response.status_code == 200
        assert response.content == b"fake webm content"

    def test_get_video_gif_success(self):
        """Test successful retrieval of a GIF video."""
        video_path = self.temp_dir / "test789.gif"
        video_path.write_bytes(b"fake gif content")
        self.mock_service.get_video_path.return_value = video_path

        response = self.client.get("/api/videos/test789")

        assert response.status_code == 200
        assert response.content == b"fake gif content"

    def test_get_video_strips_extension(self):
        """Test that video_id extension is stripped before lookup."""
        video_path = self.temp_dir / "abc123.mp4"
        video_path.write_bytes(b"video data")
        self.mock_service.get_video_path.return_value = video_path

        response = self.client.get("/api/videos/abc123.mp4")

        assert response.status_code == 200
        self.mock_service.get_video_path.assert_called_once_with("abc123")

    def test_get_video_not_found_returns_404(self):
        """Test that missing video returns 404."""
        self.mock_service.get_video_path.return_value = None

        response = self.client.get("/api/videos/nonexistent")

        assert response.status_code == 404
        assert "video_not_found" in response.json()["detail"]["error"]

    def test_get_video_correct_media_type_mp4(self):
        """Test that MP4 videos are served with correct media type."""
        video_path = self.temp_dir / "video.mp4"
        video_path.write_bytes(b"mp4 data")
        self.mock_service.get_video_path.return_value = video_path

        response = self.client.get("/api/videos/video")

        assert response.status_code == 200
        assert "video/mp4" in response.headers["content-type"]

    def test_get_video_correct_media_type_webm(self):
        """Test that WebM videos are served with correct media type."""
        video_path = self.temp_dir / "video.webm"
        video_path.write_bytes(b"webm data")
        self.mock_service.get_video_path.return_value = video_path

        response = self.client.get("/api/videos/video")

        assert response.status_code == 200
        assert "video/webm" in response.headers["content-type"]

    def test_get_video_correct_media_type_gif(self):
        """Test that GIF videos are served with correct media type."""
        video_path = self.temp_dir / "video.gif"
        video_path.write_bytes(b"gif data")
        self.mock_service.get_video_path.return_value = video_path

        response = self.client.get("/api/videos/video")

        assert response.status_code == 200
        assert "image/gif" in response.headers["content-type"]


class TestDeleteVideoEndpoint:
    """Tests for DELETE /api/videos/{video_id} endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_video_service, temp_cache_dir):
        """Set up test fixtures."""
        self.mock_service = mock_video_service
        self.temp_dir = temp_cache_dir

        app.dependency_overrides[get_video_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_delete_video_success(self):
        """Test successful video deletion returns 204."""
        self.mock_service.delete_video.return_value = True

        response = self.client.delete("/api/videos/test123")

        assert response.status_code == 204
        self.mock_service.delete_video.assert_called_once_with("test123")

    def test_delete_video_not_found_returns_404(self):
        """Test that deleting non-existent video returns 404."""
        self.mock_service.delete_video.return_value = False

        response = self.client.delete("/api/videos/nonexistent")

        assert response.status_code == 404
        assert "video_not_found" in response.json()["detail"]["error"]

    def test_delete_video_strips_extension(self):
        """Test that video_id extension is stripped before deletion."""
        self.mock_service.delete_video.return_value = True

        response = self.client.delete("/api/videos/abc123.webm")

        assert response.status_code == 204
        self.mock_service.delete_video.assert_called_once_with("abc123")


class TestGetCacheStatsEndpoint:
    """Tests for GET /api/videos (cache stats) endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_video_service):
        """Set up test fixtures."""
        self.mock_service = mock_video_service

        app.dependency_overrides[get_video_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_get_cache_stats_success(self):
        """Test successful retrieval of cache statistics."""
        self.mock_service.get_cache_stats.return_value = {
            "file_count": 5,
            "total_size_bytes": 1024000,
            "total_size_mb": 1.0,
            "cache_dir": "/tmp/cache",
        }

        response = self.client.get("/api/videos")

        assert response.status_code == 200
        self.mock_service.get_cache_stats.assert_called_once()

    def test_get_cache_stats_returns_file_count(self):
        """Test that cache stats include file count."""
        self.mock_service.get_cache_stats.return_value = {
            "file_count": 10,
            "total_size_bytes": 2048000,
            "total_size_mb": 2.0,
            "cache_dir": "/tmp/cache",
        }

        response = self.client.get("/api/videos")

        assert response.status_code == 200
        data = response.json()
        assert data["file_count"] == 10

    def test_get_cache_stats_returns_total_size(self):
        """Test that cache stats include total size information."""
        self.mock_service.get_cache_stats.return_value = {
            "file_count": 3,
            "total_size_bytes": 5120000,
            "total_size_mb": 5.0,
            "cache_dir": "/tmp/cache",
        }

        response = self.client.get("/api/videos")

        assert response.status_code == 200
        data = response.json()
        assert data["total_size_bytes"] == 5120000
        assert data["total_size_mb"] == 5.0


class TestCleanupCacheEndpoint:
    """Tests for POST /api/videos/cleanup endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_video_service):
        """Set up test fixtures."""
        self.mock_service = mock_video_service

        app.dependency_overrides[get_video_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_cleanup_cache_default_age(self):
        """Test cache cleanup with default max age (24 hours)."""
        self.mock_service.cleanup_cache.return_value = 5

        response = self.client.post("/api/videos/cleanup")

        assert response.status_code == 200
        self.mock_service.cleanup_cache.assert_called_once_with(24)
        data = response.json()
        assert data["max_age_hours"] == 24

    def test_cleanup_cache_custom_age(self):
        """Test cache cleanup with custom max age."""
        self.mock_service.cleanup_cache.return_value = 3

        response = self.client.post("/api/videos/cleanup?max_age_hours=48")

        assert response.status_code == 200
        self.mock_service.cleanup_cache.assert_called_once_with(48)
        data = response.json()
        assert data["max_age_hours"] == 48

    def test_cleanup_cache_returns_deleted_count(self):
        """Test that cleanup returns the count of deleted files."""
        self.mock_service.cleanup_cache.return_value = 7

        response = self.client.post("/api/videos/cleanup")

        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 7
