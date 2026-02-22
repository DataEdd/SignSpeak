"""Tests for translate routes."""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from packages.api.main import app
from packages.api.dependencies import get_translation_service


class TestTranslateEndpoint:
    """Tests for POST /api/translate endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_translation_service):
        """Set up test fixtures."""
        self.mock_service = mock_translation_service

        # Override dependency
        app.dependency_overrides[get_translation_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        # Clean up
        app.dependency_overrides.clear()

    def test_translate_success_basic(self):
        """Test successful translation with basic request."""
        response = self.client.post(
            "/api/translate",
            json={"text": "Hello world"}
        )
        assert response.status_code == 200
        assert "glosses" in response.json()

    def test_translate_success_with_options(self):
        """Test successful translation with options provided."""
        response = self.client.post(
            "/api/translate",
            json={
                "text": "Hello world",
                "options": {
                    "speed": "slow",
                    "format": "webm",
                    "include_fingerspelling": True
                }
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "glosses" in data

    def test_translate_returns_glosses(self):
        """Test that translation response includes glosses list."""
        self.mock_service.translate_text.return_value = {
            "glosses": ["HELLO", "WORLD"],
            "video_url": "/api/videos/test123.mp4",
            "confidence": 0.95,
            "quality": "HIGH",
            "missing_signs": [],
            "fingerspelled": [],
        }

        response = self.client.post(
            "/api/translate",
            json={"text": "Hello world"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["glosses"] == ["HELLO", "WORLD"]

    def test_translate_returns_video_url(self):
        """Test that translation response includes video URL."""
        self.mock_service.translate_text.return_value = {
            "glosses": ["HELLO"],
            "video_url": "/api/videos/abc123.mp4",
            "confidence": 0.9,
            "quality": "HIGH",
            "missing_signs": [],
            "fingerspelled": [],
        }

        response = self.client.post(
            "/api/translate",
            json={"text": "Hello"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["video_url"] == "/api/videos/abc123.mp4"

    def test_translate_returns_confidence(self):
        """Test that translation response includes confidence score."""
        self.mock_service.translate_text.return_value = {
            "glosses": ["HELLO"],
            "video_url": "/api/videos/test.mp4",
            "confidence": 0.87,
            "quality": "MEDIUM",
            "missing_signs": [],
            "fingerspelled": [],
        }

        response = self.client.post(
            "/api/translate",
            json={"text": "Hello"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["confidence"] == 0.87

    def test_translate_returns_quality(self):
        """Test that translation response includes quality level."""
        self.mock_service.translate_text.return_value = {
            "glosses": ["HELLO"],
            "video_url": "/api/videos/test.mp4",
            "confidence": 0.95,
            "quality": "HIGH",
            "missing_signs": [],
            "fingerspelled": [],
        }

        response = self.client.post(
            "/api/translate",
            json={"text": "Hello"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["quality"] == "HIGH"

    def test_translate_returns_missing_signs(self):
        """Test that translation response includes missing signs list."""
        self.mock_service.translate_text.return_value = {
            "glosses": ["HELLO", "XYZWORD"],
            "video_url": "/api/videos/test.mp4",
            "confidence": 0.6,
            "quality": "LOW",
            "missing_signs": ["XYZWORD"],
            "fingerspelled": [],
        }

        response = self.client.post(
            "/api/translate",
            json={"text": "Hello xyzword"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["missing_signs"] == ["XYZWORD"]

    def test_translate_returns_fingerspelled(self):
        """Test that translation response includes fingerspelled words list."""
        self.mock_service.translate_text.return_value = {
            "glosses": ["HELLO", "J-O-H-N"],
            "video_url": "/api/videos/test.mp4",
            "confidence": 0.85,
            "quality": "MEDIUM",
            "missing_signs": [],
            "fingerspelled": ["JOHN"],
        }

        response = self.client.post(
            "/api/translate",
            json={"text": "Hello John"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["fingerspelled"] == ["JOHN"]

    def test_translate_invalid_text_empty(self):
        """Test that empty text returns 400 error."""
        response = self.client.post(
            "/api/translate",
            json={"text": ""}
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_translate_invalid_text_too_long(self):
        """Test that text over 1000 characters returns 400 error."""
        long_text = "a" * 1001
        response = self.client.post(
            "/api/translate",
            json={"text": long_text}
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_translate_value_error_returns_400(self):
        """Test that ValueError from service returns 400 status."""
        self.mock_service.translate_text.side_effect = ValueError("Invalid input text")

        response = self.client.post(
            "/api/translate",
            json={"text": "Hello world"}
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"] == "invalid_input"
        assert "Invalid input text" in data["detail"]["message"]

    def test_translate_internal_error_returns_500(self):
        """Test that unexpected exceptions return 500 status."""
        self.mock_service.translate_text.side_effect = RuntimeError("Database connection failed")

        response = self.client.post(
            "/api/translate",
            json={"text": "Hello world"}
        )
        assert response.status_code == 500
        data = response.json()
        assert data["detail"]["error"] == "translation_failed"
        assert "Database connection failed" in data["detail"]["message"]

    def test_translate_extracts_speed_option(self):
        """Test that speed option is correctly extracted and passed to service."""
        response = self.client.post(
            "/api/translate",
            json={
                "text": "Hello",
                "options": {"speed": "fast", "format": "mp4"}
            }
        )
        assert response.status_code == 200

        # Verify the service was called with correct speed
        self.mock_service.translate_text.assert_called_once()
        call_kwargs = self.mock_service.translate_text.call_args.kwargs
        assert call_kwargs["speed"] == "fast"

    def test_translate_extracts_format_option(self):
        """Test that format option is correctly extracted and passed to service."""
        response = self.client.post(
            "/api/translate",
            json={
                "text": "Hello",
                "options": {"speed": "normal", "format": "gif"}
            }
        )
        assert response.status_code == 200

        # Verify the service was called with correct format
        self.mock_service.translate_text.assert_called_once()
        call_kwargs = self.mock_service.translate_text.call_args.kwargs
        assert call_kwargs["video_format"] == "gif"


class TestPreviewEndpoint:
    """Tests for POST /api/translate/preview endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_translation_service):
        """Set up test fixtures."""
        self.mock_service = mock_translation_service

        # Override dependency
        app.dependency_overrides[get_translation_service] = lambda: self.mock_service
        self.client = TestClient(app, raise_server_exceptions=False)

        yield

        # Clean up
        app.dependency_overrides.clear()

    def test_preview_success(self):
        """Test successful preview request."""
        response = self.client.post(
            "/api/translate/preview",
            json={"text": "Hello world"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "glosses" in data

    def test_preview_returns_glosses(self):
        """Test that preview response includes glosses list."""
        self.mock_service.get_gloss_preview.return_value = {
            "glosses": ["HELLO", "WORLD"],
            "available_signs": ["HELLO", "WORLD"],
            "missing_signs": [],
            "fingerspelled": [],
            "confidence": 0.95,
            "quality": "HIGH",
        }

        response = self.client.post(
            "/api/translate/preview",
            json={"text": "Hello world"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["glosses"] == ["HELLO", "WORLD"]

    def test_preview_returns_available_signs(self):
        """Test that preview response includes available signs list."""
        self.mock_service.get_gloss_preview.return_value = {
            "glosses": ["HELLO", "WORLD"],
            "available_signs": ["HELLO", "WORLD"],
            "missing_signs": [],
            "fingerspelled": [],
            "confidence": 0.95,
            "quality": "HIGH",
        }

        response = self.client.post(
            "/api/translate/preview",
            json={"text": "Hello world"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["available_signs"] == ["HELLO", "WORLD"]

    def test_preview_returns_missing_signs(self):
        """Test that preview response includes missing signs list."""
        self.mock_service.get_gloss_preview.return_value = {
            "glosses": ["HELLO", "UNKNOWNWORD"],
            "available_signs": ["HELLO"],
            "missing_signs": ["UNKNOWNWORD"],
            "fingerspelled": [],
            "confidence": 0.7,
            "quality": "MEDIUM",
        }

        response = self.client.post(
            "/api/translate/preview",
            json={"text": "Hello unknownword"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["missing_signs"] == ["UNKNOWNWORD"]

    def test_preview_returns_confidence(self):
        """Test that preview response includes confidence score."""
        self.mock_service.get_gloss_preview.return_value = {
            "glosses": ["HELLO"],
            "available_signs": ["HELLO"],
            "missing_signs": [],
            "fingerspelled": [],
            "confidence": 0.92,
            "quality": "HIGH",
        }

        response = self.client.post(
            "/api/translate/preview",
            json={"text": "Hello"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["confidence"] == 0.92

    def test_preview_invalid_input_returns_400(self):
        """Test that ValueError from service returns 400 status."""
        self.mock_service.get_gloss_preview.side_effect = ValueError("Invalid text provided")

        response = self.client.post(
            "/api/translate/preview",
            json={"text": "Hello world"}
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["error"] == "invalid_input"
        assert "Invalid text provided" in data["detail"]["message"]
