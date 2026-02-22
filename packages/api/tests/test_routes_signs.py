"""Tests for signs routes."""

import io
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest
from fastapi.testclient import TestClient

from packages.api.main import app
from packages.api.dependencies import get_sign_service


class TestListSignsEndpoint:
    """Tests for GET /api/signs endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_sign_service, sample_sign_dict):
        """Set up test fixtures."""
        self.mock_service = mock_sign_service
        self.sample_sign = sample_sign_dict

        app.dependency_overrides[get_sign_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_list_signs_default_verified(self):
        """Test that list defaults to verified status."""
        self.mock_service.list_signs.return_value = {"signs": [], "total": 0}

        response = self.client.get("/api/signs")

        assert response.status_code == 200
        self.mock_service.list_signs.assert_called_once()
        call_args = self.mock_service.list_signs.call_args
        assert call_args.kwargs.get("status") == "verified"

    def test_list_signs_with_status_filter(self):
        """Test filtering signs by status parameter."""
        self.mock_service.list_signs.return_value = {"signs": [], "total": 0}

        response = self.client.get("/api/signs?status=pending")

        assert response.status_code == 200
        call_args = self.mock_service.list_signs.call_args
        assert call_args.kwargs.get("status") == "pending"

    def test_list_signs_with_category_filter(self):
        """Test filtering signs by category parameter."""
        self.mock_service.list_signs.return_value = {"signs": [], "total": 0}

        response = self.client.get("/api/signs?category=greeting")

        assert response.status_code == 200
        call_args = self.mock_service.list_signs.call_args
        assert call_args.kwargs.get("category") == "greeting"

    def test_list_signs_pagination_limit(self):
        """Test limit parameter is passed correctly."""
        self.mock_service.list_signs.return_value = {"signs": [], "total": 0}

        response = self.client.get("/api/signs?limit=25")

        assert response.status_code == 200
        call_args = self.mock_service.list_signs.call_args
        assert call_args.kwargs.get("limit") == 25

    def test_list_signs_pagination_offset(self):
        """Test offset parameter is passed correctly."""
        self.mock_service.list_signs.return_value = {"signs": [], "total": 0}

        response = self.client.get("/api/signs?offset=10")

        assert response.status_code == 200
        call_args = self.mock_service.list_signs.call_args
        assert call_args.kwargs.get("offset") == 10

    def test_list_signs_returns_total(self):
        """Test that response includes total count."""
        self.mock_service.list_signs.return_value = {
            "signs": [self.sample_sign],
            "total": 100,
        }

        response = self.client.get("/api/signs")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 100
        assert len(data["signs"]) == 1


class TestSearchSignsEndpoint:
    """Tests for GET /api/signs/search endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_sign_service, sample_sign_dict):
        """Set up test fixtures."""
        self.mock_service = mock_sign_service
        self.sample_sign = sample_sign_dict

        app.dependency_overrides[get_sign_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_search_signs_basic(self):
        """Test basic search functionality."""
        self.mock_service.search_signs.return_value = [self.sample_sign]

        response = self.client.get("/api/signs/search?q=hello")

        assert response.status_code == 200
        self.mock_service.search_signs.assert_called_once()
        call_args = self.mock_service.search_signs.call_args
        assert call_args.kwargs.get("query") == "hello"

    def test_search_signs_with_status(self):
        """Test search with status filter."""
        self.mock_service.search_signs.return_value = []

        response = self.client.get("/api/signs/search?q=hello&status=pending")

        assert response.status_code == 200
        call_args = self.mock_service.search_signs.call_args
        assert call_args.kwargs.get("status") == "pending"

    def test_search_signs_with_limit(self):
        """Test search with limit parameter."""
        self.mock_service.search_signs.return_value = []

        response = self.client.get("/api/signs/search?q=hello&limit=10")

        assert response.status_code == 200
        call_args = self.mock_service.search_signs.call_args
        assert call_args.kwargs.get("limit") == 10

    def test_search_returns_list(self):
        """Test that search returns a list of signs."""
        self.mock_service.search_signs.return_value = [
            self.sample_sign,
            self.sample_sign,
        ]

        response = self.client.get("/api/signs/search?q=test")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2


class TestGetStatsEndpoint:
    """Tests for GET /api/signs/stats endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_sign_service):
        """Set up test fixtures."""
        self.mock_service = mock_sign_service

        app.dependency_overrides[get_sign_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_get_stats_success(self):
        """Test successful stats retrieval."""
        self.mock_service.get_stats.return_value = {
            "total_signs": 100,
            "verified_signs": 50,
            "pending_signs": 30,
            "imported_signs": 15,
            "rejected_signs": 5,
            "categories": {"greeting": 10, "noun": 20},
        }

        response = self.client.get("/api/signs/stats")

        assert response.status_code == 200
        self.mock_service.get_stats.assert_called_once()

    def test_get_stats_returns_counts(self):
        """Test that stats response contains all count fields."""
        self.mock_service.get_stats.return_value = {
            "total_signs": 100,
            "verified_signs": 50,
            "pending_signs": 30,
            "imported_signs": 15,
            "rejected_signs": 5,
            "categories": {"greeting": 10},
        }

        response = self.client.get("/api/signs/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_signs" in data
        assert "verified_signs" in data
        assert "pending_signs" in data
        assert "imported_signs" in data
        assert "rejected_signs" in data
        assert "categories" in data


class TestGetSignEndpoint:
    """Tests for GET /api/signs/{gloss} endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_sign_service, sample_sign_dict):
        """Set up test fixtures."""
        self.mock_service = mock_sign_service
        self.sample_sign = sample_sign_dict

        app.dependency_overrides[get_sign_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_get_sign_success(self):
        """Test successful sign retrieval."""
        self.mock_service.get_sign.return_value = self.sample_sign

        response = self.client.get("/api/signs/HELLO")

        assert response.status_code == 200
        data = response.json()
        assert data["gloss"] == "HELLO"

    def test_get_sign_not_found_returns_404(self):
        """Test that missing sign returns 404."""
        self.mock_service.get_sign.return_value = None

        response = self.client.get("/api/signs/NONEXISTENT")

        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "sign_not_found"

    def test_get_sign_case_insensitive(self):
        """Test that gloss lookup is case insensitive (hello -> HELLO)."""
        self.mock_service.get_sign.return_value = self.sample_sign

        response = self.client.get("/api/signs/hello")

        assert response.status_code == 200
        self.mock_service.get_sign.assert_called_once_with("HELLO")


class TestGetSignVideoEndpoint:
    """Tests for GET /api/signs/{gloss}/video endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_sign_service, sample_sign_dict, tmp_path):
        """Set up test fixtures."""
        self.mock_service = mock_sign_service
        self.sample_sign = sample_sign_dict
        self.tmp_path = tmp_path

        app.dependency_overrides[get_sign_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_get_sign_video_success(self):
        """Test successful video retrieval when file exists."""
        # Create temp video file
        sign_dir = self.tmp_path / "HELLO"
        sign_dir.mkdir()
        video_file = sign_dir / "video.mp4"
        video_file.write_bytes(b"fake video content")

        # Mock sign object with path
        mock_sign_obj = MagicMock()
        mock_sign_obj.path = sign_dir

        self.mock_service.get_sign.return_value = self.sample_sign
        self.mock_service.store.get_sign.return_value = mock_sign_obj

        response = self.client.get("/api/signs/HELLO/video")

        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"

    def test_get_sign_video_sign_not_found_returns_404(self):
        """Test that missing sign returns 404 for video endpoint."""
        self.mock_service.get_sign.return_value = None

        response = self.client.get("/api/signs/NONEXISTENT/video")

        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "sign_not_found"

    def test_get_sign_video_file_missing_returns_404(self):
        """Test that missing video file returns 404."""
        # Mock sign exists but no video file
        mock_sign_obj = MagicMock()
        mock_sign_obj.path = self.tmp_path / "HELLO"  # Directory doesn't exist

        self.mock_service.get_sign.return_value = self.sample_sign
        self.mock_service.store.get_sign.return_value = mock_sign_obj

        response = self.client.get("/api/signs/HELLO/video")

        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "video_not_found"


class TestCreateSignEndpoint:
    """Tests for POST /api/signs endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_sign_service, sample_pending_sign_dict):
        """Set up test fixtures."""
        self.mock_service = mock_sign_service
        self.sample_pending_sign = sample_pending_sign_dict

        app.dependency_overrides[get_sign_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_create_sign_success(self):
        """Test successful sign creation returns 201."""
        self.mock_service.get_sign.return_value = None  # Sign doesn't exist
        self.mock_service.create_sign.return_value = self.sample_pending_sign

        video_content = b"fake video data"
        files = {"video": ("test.mp4", io.BytesIO(video_content), "video/mp4")}
        data = {"gloss": "WORLD", "english": "world, earth", "category": "noun"}

        response = self.client.post("/api/signs", files=files, data=data)

        assert response.status_code == 201
        self.mock_service.create_sign.assert_called_once()

    def test_create_sign_already_exists_returns_409(self):
        """Test that creating duplicate sign returns 409."""
        self.mock_service.get_sign.return_value = self.sample_pending_sign

        video_content = b"fake video data"
        files = {"video": ("test.mp4", io.BytesIO(video_content), "video/mp4")}
        data = {"gloss": "WORLD"}

        response = self.client.post("/api/signs", files=files, data=data)

        assert response.status_code == 409
        data = response.json()
        assert data["detail"]["error"] == "sign_exists"

    def test_create_sign_parses_english_list(self):
        """Test that comma-separated English translations are parsed correctly."""
        self.mock_service.get_sign.return_value = None
        self.mock_service.create_sign.return_value = self.sample_pending_sign

        video_content = b"fake video data"
        files = {"video": ("test.mp4", io.BytesIO(video_content), "video/mp4")}
        data = {"gloss": "WORLD", "english": "world, earth, globe"}

        response = self.client.post("/api/signs", files=files, data=data)

        assert response.status_code == 201
        call_args = self.mock_service.create_sign.call_args
        assert call_args.kwargs.get("english") == ["world", "earth", "globe"]


class TestVerifySignEndpoint:
    """Tests for PUT /api/signs/{gloss}/verify endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_sign_service, sample_sign_dict, sample_pending_sign_dict):
        """Set up test fixtures."""
        self.mock_service = mock_sign_service
        self.sample_sign = sample_sign_dict
        self.sample_pending_sign = sample_pending_sign_dict

        app.dependency_overrides[get_sign_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_verify_sign_success(self):
        """Test successful sign verification."""
        self.mock_service.get_sign.return_value = self.sample_pending_sign
        verified_sign = {**self.sample_pending_sign, "status": "verified"}
        self.mock_service.verify_sign.return_value = verified_sign

        response = self.client.put(
            "/api/signs/WORLD/verify",
            json={"quality_score": 5, "verified_by": "tester"},
        )

        assert response.status_code == 200
        self.mock_service.verify_sign.assert_called_once_with(
            gloss="WORLD",
            quality_score=5,
            verified_by="tester",
        )

    def test_verify_sign_not_found_returns_404(self):
        """Test that verifying missing sign returns 404."""
        self.mock_service.get_sign.return_value = None

        response = self.client.put(
            "/api/signs/NONEXISTENT/verify",
            json={"quality_score": 5, "verified_by": "tester"},
        )

        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "sign_not_found"

    def test_verify_sign_invalid_status_returns_400(self):
        """Test that verifying already verified sign returns 400."""
        self.mock_service.get_sign.return_value = self.sample_sign  # Already verified

        response = self.client.put(
            "/api/signs/HELLO/verify",
            json={"quality_score": 5, "verified_by": "tester"},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"] == "invalid_status"

    def test_verify_sign_invalid_score(self):
        """Test that invalid quality score is rejected."""
        self.mock_service.get_sign.return_value = self.sample_pending_sign

        # Score out of range (1-5)
        response = self.client.put(
            "/api/signs/WORLD/verify",
            json={"quality_score": 10, "verified_by": "tester"},
        )

        assert response.status_code == 422  # Validation error


class TestRejectSignEndpoint:
    """Tests for PUT /api/signs/{gloss}/reject endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_sign_service, sample_pending_sign_dict):
        """Set up test fixtures."""
        self.mock_service = mock_sign_service
        self.sample_pending_sign = sample_pending_sign_dict

        app.dependency_overrides[get_sign_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_reject_sign_success(self):
        """Test successful sign rejection."""
        self.mock_service.get_sign.return_value = self.sample_pending_sign
        rejected_sign = {**self.sample_pending_sign, "status": "rejected"}
        self.mock_service.reject_sign.return_value = rejected_sign

        response = self.client.put(
            "/api/signs/WORLD/reject",
            data={"reason": "Poor quality"},
        )

        assert response.status_code == 200
        self.mock_service.reject_sign.assert_called_once()

    def test_reject_sign_not_found_returns_404(self):
        """Test that rejecting missing sign returns 404."""
        self.mock_service.get_sign.return_value = None

        response = self.client.put(
            "/api/signs/NONEXISTENT/reject",
            data={"reason": "Not found"},
        )

        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "sign_not_found"


class TestDeleteSignEndpoint:
    """Tests for DELETE /api/signs/{gloss} endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_sign_service):
        """Set up test fixtures."""
        self.mock_service = mock_sign_service

        app.dependency_overrides[get_sign_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        app.dependency_overrides.clear()

    def test_delete_sign_success(self):
        """Test successful sign deletion returns 204."""
        self.mock_service.delete_sign.return_value = True

        response = self.client.delete("/api/signs/HELLO")

        assert response.status_code == 204
        self.mock_service.delete_sign.assert_called_once_with("HELLO")

    def test_delete_sign_not_found_returns_404(self):
        """Test that deleting missing sign returns 404."""
        self.mock_service.delete_sign.return_value = False

        response = self.client.delete("/api/signs/NONEXISTENT")

        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "sign_not_found"
