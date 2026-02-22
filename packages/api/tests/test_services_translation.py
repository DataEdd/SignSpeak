"""Tests for translation service."""

from unittest.mock import MagicMock, patch
import pytest

from packages.api.services.translation_service import TranslationService


class TestTranslationService:
    """Tests for TranslationService class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.mock_store = MagicMock()
        self.mock_video_service = MagicMock()
        self.service = TranslationService(
            sign_store=self.mock_store,
            video_service=self.mock_video_service,
        )

    def test_init_stores_dependencies(self):
        """Test that init stores dependencies correctly."""
        assert self.service.sign_store == self.mock_store
        assert self.service.video_service == self.mock_video_service

    @patch("packages.api.services.translation_service.translate")
    def test_translate_text_calls_translate(self, mock_translate):
        """Test that translate_text calls the translate function with correct arguments."""
        mock_result = MagicMock()
        mock_result.glosses = ["HELLO", "WORLD"]
        mock_result.confidence = 0.95
        mock_result.quality.value = "HIGH"
        mock_result.validation = None
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        self.mock_video_service.create_video.return_value = "/tmp/video.mp4"

        self.service.translate_text("Hello world")

        mock_translate.assert_called_once_with("Hello world", store=self.mock_store)

    @patch("packages.api.services.translation_service.translate")
    def test_translate_text_calls_video_service(self, mock_translate):
        """Test that translate_text calls video_service.create_video with correct arguments."""
        mock_result = MagicMock()
        mock_result.glosses = ["HELLO", "WORLD"]
        mock_result.confidence = 0.95
        mock_result.quality.value = "HIGH"
        mock_result.validation = None
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        self.mock_video_service.create_video.return_value = "/tmp/video.mp4"

        self.service.translate_text("Hello world", speed="normal", video_format="mp4")

        self.mock_video_service.create_video.assert_called_once()
        call_kwargs = self.mock_video_service.create_video.call_args[1]
        assert call_kwargs["glosses"] == ["HELLO", "WORLD"]
        assert call_kwargs["speed"] == "normal"
        assert call_kwargs["format"] == "mp4"

    @patch("packages.api.services.translation_service.translate")
    def test_translate_text_returns_glosses(self, mock_translate):
        """Test that translate_text returns the correct glosses."""
        mock_result = MagicMock()
        mock_result.glosses = ["HELLO", "WORLD"]
        mock_result.confidence = 0.95
        mock_result.quality.value = "HIGH"
        mock_result.validation = None
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        self.mock_video_service.create_video.return_value = "/tmp/video.mp4"

        result = self.service.translate_text("Hello world")

        assert result["glosses"] == ["HELLO", "WORLD"]

    @patch("packages.api.services.translation_service.translate")
    @patch("packages.api.services.translation_service.uuid")
    def test_translate_text_returns_video_url(self, mock_uuid, mock_translate):
        """Test that translate_text returns the correct video URL."""
        mock_uuid.uuid4.return_value = MagicMock()
        mock_uuid.uuid4.return_value.__str__ = MagicMock(return_value="abc12345")

        mock_result = MagicMock()
        mock_result.glosses = ["HELLO"]
        mock_result.confidence = 0.95
        mock_result.quality.value = "HIGH"
        mock_result.validation = None
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        self.mock_video_service.create_video.return_value = "/tmp/video.mp4"

        result = self.service.translate_text("Hello", video_format="mp4")

        assert result["video_url"] == "/api/videos/abc12345.mp4"

    @patch("packages.api.services.translation_service.translate")
    def test_translate_text_returns_confidence(self, mock_translate):
        """Test that translate_text returns the correct confidence score."""
        mock_result = MagicMock()
        mock_result.glosses = ["HELLO"]
        mock_result.confidence = 0.87
        mock_result.quality.value = "MEDIUM"
        mock_result.validation = None
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        self.mock_video_service.create_video.return_value = "/tmp/video.mp4"

        result = self.service.translate_text("Hello")

        assert result["confidence"] == 0.87

    @patch("packages.api.services.translation_service.translate")
    def test_translate_text_returns_quality(self, mock_translate):
        """Test that translate_text returns the correct quality value."""
        mock_result = MagicMock()
        mock_result.glosses = ["HELLO"]
        mock_result.confidence = 0.95
        mock_result.quality.value = "HIGH"
        mock_result.validation = None
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        self.mock_video_service.create_video.return_value = "/tmp/video.mp4"

        result = self.service.translate_text("Hello")

        assert result["quality"] == "HIGH"

    @patch("packages.api.services.translation_service.translate")
    def test_translate_text_returns_missing_signs(self, mock_translate):
        """Test that translate_text returns missing signs from validation."""
        mock_validation = MagicMock()
        mock_validation.missing_glosses = ["UNKNOWN", "WORD"]

        mock_result = MagicMock()
        mock_result.glosses = ["HELLO", "UNKNOWN", "WORD"]
        mock_result.confidence = 0.70
        mock_result.quality.value = "LOW"
        mock_result.validation = mock_validation
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        self.mock_video_service.create_video.return_value = "/tmp/video.mp4"

        result = self.service.translate_text("Hello unknown word")

        assert result["missing_signs"] == ["UNKNOWN", "WORD"]

    @patch("packages.api.services.translation_service.translate")
    def test_translate_text_returns_fingerspelled(self, mock_translate):
        """Test that translate_text returns fingerspelled words."""
        mock_result = MagicMock()
        mock_result.glosses = ["HELLO", "fs-JOHN"]
        mock_result.confidence = 0.90
        mock_result.quality.value = "HIGH"
        mock_result.validation = None
        mock_result.fingerspelled = ["JOHN"]
        mock_translate.return_value = mock_result

        self.mock_video_service.create_video.return_value = "/tmp/video.mp4"

        result = self.service.translate_text("Hello John")

        assert result["fingerspelled"] == ["JOHN"]

    @patch("packages.api.services.translation_service.translate")
    def test_translate_text_with_speed_option(self, mock_translate):
        """Test that translate_text passes speed option to video service."""
        mock_result = MagicMock()
        mock_result.glosses = ["HELLO"]
        mock_result.confidence = 0.95
        mock_result.quality.value = "HIGH"
        mock_result.validation = None
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        self.mock_video_service.create_video.return_value = "/tmp/video.mp4"

        self.service.translate_text("Hello", speed="slow")

        call_kwargs = self.mock_video_service.create_video.call_args[1]
        assert call_kwargs["speed"] == "slow"

    @patch("packages.api.services.translation_service.translate")
    def test_translate_text_with_format_option(self, mock_translate):
        """Test that translate_text passes format option to video service."""
        mock_result = MagicMock()
        mock_result.glosses = ["HELLO"]
        mock_result.confidence = 0.95
        mock_result.quality.value = "HIGH"
        mock_result.validation = None
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        self.mock_video_service.create_video.return_value = "/tmp/video.webm"

        self.service.translate_text("Hello", video_format="webm")

        call_kwargs = self.mock_video_service.create_video.call_args[1]
        assert call_kwargs["format"] == "webm"

    @patch("packages.api.services.translation_service.translate")
    def test_get_gloss_preview_returns_glosses(self, mock_translate):
        """Test that get_gloss_preview returns the correct glosses."""
        mock_result = MagicMock()
        mock_result.glosses = ["HELLO", "WORLD"]
        mock_result.confidence = 0.95
        mock_result.quality.value = "HIGH"
        mock_result.validation = None
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        result = self.service.get_gloss_preview("Hello world")

        assert result["glosses"] == ["HELLO", "WORLD"]

    @patch("packages.api.services.translation_service.translate")
    def test_get_gloss_preview_returns_available_signs(self, mock_translate):
        """Test that get_gloss_preview returns available signs."""
        mock_validation = MagicMock()
        mock_validation.missing_glosses = ["UNKNOWN"]

        mock_result = MagicMock()
        mock_result.glosses = ["HELLO", "UNKNOWN"]
        mock_result.confidence = 0.80
        mock_result.quality.value = "MEDIUM"
        mock_result.validation = mock_validation
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        result = self.service.get_gloss_preview("Hello unknown")

        assert result["available_signs"] == ["HELLO"]

    @patch("packages.api.services.translation_service.translate")
    def test_get_gloss_preview_returns_missing_signs(self, mock_translate):
        """Test that get_gloss_preview returns missing signs."""
        mock_validation = MagicMock()
        mock_validation.missing_glosses = ["UNKNOWN", "WORD"]

        mock_result = MagicMock()
        mock_result.glosses = ["HELLO", "UNKNOWN", "WORD"]
        mock_result.confidence = 0.70
        mock_result.quality.value = "LOW"
        mock_result.validation = mock_validation
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        result = self.service.get_gloss_preview("Hello unknown word")

        assert result["missing_signs"] == ["UNKNOWN", "WORD"]

    @patch("packages.api.services.translation_service.translate")
    def test_get_gloss_preview_no_video_generation(self, mock_translate):
        """Test that get_gloss_preview does not generate video."""
        mock_result = MagicMock()
        mock_result.glosses = ["HELLO"]
        mock_result.confidence = 0.95
        mock_result.quality.value = "HIGH"
        mock_result.validation = None
        mock_result.fingerspelled = []
        mock_translate.return_value = mock_result

        self.service.get_gloss_preview("Hello")

        self.mock_video_service.create_video.assert_not_called()
