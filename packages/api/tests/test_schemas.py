"""Tests for API schemas module."""

import pytest
from pydantic import ValidationError

from packages.api.schemas import (
    VideoFormat,
    TranslationSpeed,
    TranslateRequest,
    TranslationOptions,
    TranslateResponse,
    SignStatus,
    SignResponse,
    SignListResponse,
    SignCreateRequest,
    SignVerifyRequest,
    SignSearchParams,
    ErrorResponse,
    HealthResponse,
    StatsResponse,
)


class TestVideoFormat:
    """Tests for VideoFormat enum."""

    def test_mp4_value(self):
        """Test MP4 enum value."""
        assert VideoFormat.MP4.value == "mp4"

    def test_webm_value(self):
        """Test WEBM enum value."""
        assert VideoFormat.WEBM.value == "webm"

    def test_gif_value(self):
        """Test GIF enum value."""
        assert VideoFormat.GIF.value == "gif"

    def test_invalid_value_raises(self):
        """Test that invalid value raises error."""
        with pytest.raises(ValueError):
            VideoFormat("invalid")


class TestTranslationSpeed:
    """Tests for TranslationSpeed enum."""

    def test_slow_value(self):
        """Test SLOW enum value."""
        assert TranslationSpeed.SLOW.value == "slow"

    def test_normal_value(self):
        """Test NORMAL enum value."""
        assert TranslationSpeed.NORMAL.value == "normal"

    def test_fast_value(self):
        """Test FAST enum value."""
        assert TranslationSpeed.FAST.value == "fast"

    def test_invalid_value_raises(self):
        """Test that invalid value raises error."""
        with pytest.raises(ValueError):
            TranslationSpeed("invalid")


class TestTranslateRequest:
    """Tests for TranslateRequest model."""

    def test_valid_request(self):
        """Test creating a valid translation request."""
        request = TranslateRequest(text="Hello world")
        assert request.text == "Hello world"
        assert request.options is None

    def test_text_required(self):
        """Test that text field is required."""
        with pytest.raises(ValidationError) as exc_info:
            TranslateRequest()
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("text",) for e in errors)

    def test_text_min_length(self):
        """Test that empty string fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            TranslateRequest(text="")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("text",) and e["type"] == "string_too_short" for e in errors)

    def test_text_max_length(self):
        """Test that text over 1000 characters fails validation."""
        long_text = "a" * 1001
        with pytest.raises(ValidationError) as exc_info:
            TranslateRequest(text=long_text)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("text",) and e["type"] == "string_too_long" for e in errors)

    def test_options_optional(self):
        """Test that options field is optional and None is valid."""
        request = TranslateRequest(text="Hello", options=None)
        assert request.options is None

    def test_options_with_values(self):
        """Test request with custom options."""
        options = TranslationOptions(speed=TranslationSpeed.SLOW, format=VideoFormat.WEBM)
        request = TranslateRequest(text="Hello", options=options)
        assert request.options.speed == TranslationSpeed.SLOW
        assert request.options.format == VideoFormat.WEBM


class TestTranslationOptions:
    """Tests for TranslationOptions model."""

    def test_defaults(self):
        """Test default values: speed=normal, format=mp4, include_fingerspelling=True."""
        options = TranslationOptions()
        assert options.speed == TranslationSpeed.NORMAL
        assert options.format == VideoFormat.MP4
        assert options.include_fingerspelling is True

    def test_custom_values(self):
        """Test creating options with custom values."""
        options = TranslationOptions(
            speed=TranslationSpeed.FAST,
            format=VideoFormat.GIF,
            include_fingerspelling=False
        )
        assert options.speed == TranslationSpeed.FAST
        assert options.format == VideoFormat.GIF
        assert options.include_fingerspelling is False


class TestTranslateResponse:
    """Tests for TranslateResponse model."""

    def test_all_fields_required(self):
        """Test that glosses, video_url, confidence, and quality are required."""
        with pytest.raises(ValidationError) as exc_info:
            TranslateResponse()
        errors = exc_info.value.errors()
        required_fields = {"glosses", "video_url", "confidence", "quality"}
        error_locs = {e["loc"][0] for e in errors}
        assert required_fields.issubset(error_locs)

    def test_confidence_min_range(self):
        """Test that confidence must be >= 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            TranslateResponse(
                glosses=["HELLO"],
                video_url="/video/test.mp4",
                confidence=-0.1,
                quality="high"
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("confidence",) for e in errors)

    def test_confidence_max_range(self):
        """Test that confidence must be <= 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            TranslateResponse(
                glosses=["HELLO"],
                video_url="/video/test.mp4",
                confidence=1.1,
                quality="high"
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("confidence",) for e in errors)

    def test_missing_signs_default_empty_list(self):
        """Test that missing_signs defaults to empty list."""
        response = TranslateResponse(
            glosses=["HELLO"],
            video_url="/video/test.mp4",
            confidence=0.9,
            quality="high"
        )
        assert response.missing_signs == []

    def test_fingerspelled_default_empty_list(self):
        """Test that fingerspelled defaults to empty list."""
        response = TranslateResponse(
            glosses=["HELLO"],
            video_url="/video/test.mp4",
            confidence=0.9,
            quality="high"
        )
        assert response.fingerspelled == []


class TestSignStatus:
    """Tests for SignStatus enum."""

    def test_pending_value(self):
        """Test PENDING enum value."""
        assert SignStatus.PENDING.value == "pending"

    def test_verified_value(self):
        """Test VERIFIED enum value."""
        assert SignStatus.VERIFIED.value == "verified"

    def test_imported_value(self):
        """Test IMPORTED enum value."""
        assert SignStatus.IMPORTED.value == "imported"

    def test_rejected_value(self):
        """Test REJECTED enum value."""
        assert SignStatus.REJECTED.value == "rejected"


class TestSignResponse:
    """Tests for SignResponse model."""

    def test_required_fields(self):
        """Test that required fields must be provided."""
        with pytest.raises(ValidationError) as exc_info:
            SignResponse()
        errors = exc_info.value.errors()
        required_fields = {"gloss", "english", "category", "source", "status"}
        error_locs = {e["loc"][0] for e in errors}
        assert required_fields.issubset(error_locs)

    def test_optional_fields_can_be_none(self):
        """Test that optional fields can be None."""
        response = SignResponse(
            gloss="HELLO",
            english=["hello", "hi"],
            category="greetings",
            source="recorded",
            status=SignStatus.VERIFIED,
            quality_score=None,
            verified_by=None,
            verified_date=None,
            video_url=None
        )
        assert response.quality_score is None
        assert response.verified_by is None
        assert response.verified_date is None
        assert response.video_url is None


class TestSignListResponse:
    """Tests for SignListResponse model."""

    def test_structure(self):
        """Test SignListResponse structure with signs list and total count."""
        sign = SignResponse(
            gloss="HELLO",
            english=["hello"],
            category="greetings",
            source="recorded",
            status=SignStatus.VERIFIED
        )
        response = SignListResponse(signs=[sign], total=1)
        assert len(response.signs) == 1
        assert response.total == 1
        assert response.signs[0].gloss == "HELLO"


class TestSignCreateRequest:
    """Tests for SignCreateRequest model."""

    def test_valid_request(self):
        """Test creating a valid sign create request."""
        request = SignCreateRequest(gloss="HELLO")
        assert request.gloss == "HELLO"

    def test_gloss_required(self):
        """Test that gloss field is required."""
        with pytest.raises(ValidationError) as exc_info:
            SignCreateRequest()
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("gloss",) for e in errors)

    def test_gloss_max_length(self):
        """Test that gloss over 50 characters fails validation."""
        long_gloss = "A" * 51
        with pytest.raises(ValidationError) as exc_info:
            SignCreateRequest(gloss=long_gloss)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("gloss",) and e["type"] == "string_too_long" for e in errors)

    def test_english_default_empty(self):
        """Test that english defaults to empty list."""
        request = SignCreateRequest(gloss="HELLO")
        assert request.english == []

    def test_category_default_empty(self):
        """Test that category defaults to empty string."""
        request = SignCreateRequest(gloss="HELLO")
        assert request.category == ""

    def test_source_default_recorded(self):
        """Test that source defaults to 'recorded'."""
        request = SignCreateRequest(gloss="HELLO")
        assert request.source == "recorded"


class TestSignVerifyRequest:
    """Tests for SignVerifyRequest model."""

    def test_valid_request(self):
        """Test creating a valid sign verify request."""
        request = SignVerifyRequest(quality_score=4, verified_by="John Doe")
        assert request.quality_score == 4
        assert request.verified_by == "John Doe"

    def test_quality_score_required(self):
        """Test that quality_score field is required."""
        with pytest.raises(ValidationError) as exc_info:
            SignVerifyRequest(verified_by="John Doe")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("quality_score",) for e in errors)

    def test_quality_score_min(self):
        """Test that quality_score must be >= 1."""
        with pytest.raises(ValidationError) as exc_info:
            SignVerifyRequest(quality_score=0, verified_by="John Doe")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("quality_score",) for e in errors)

    def test_quality_score_max(self):
        """Test that quality_score must be <= 5."""
        with pytest.raises(ValidationError) as exc_info:
            SignVerifyRequest(quality_score=6, verified_by="John Doe")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("quality_score",) for e in errors)

    def test_verified_by_required(self):
        """Test that verified_by field is required."""
        with pytest.raises(ValidationError) as exc_info:
            SignVerifyRequest(quality_score=4)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("verified_by",) for e in errors)

    def test_verified_by_min_length(self):
        """Test that verified_by must have at least 1 character."""
        with pytest.raises(ValidationError) as exc_info:
            SignVerifyRequest(quality_score=4, verified_by="")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("verified_by",) for e in errors)


class TestSignSearchParams:
    """Tests for SignSearchParams model."""

    def test_defaults(self):
        """Test default values for search params."""
        params = SignSearchParams()
        assert params.q is None
        assert params.status is None
        assert params.category is None
        assert params.limit == 50
        assert params.offset == 0

    def test_limit_min(self):
        """Test that limit must be >= 1."""
        with pytest.raises(ValidationError) as exc_info:
            SignSearchParams(limit=0)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("limit",) for e in errors)

    def test_limit_max(self):
        """Test that limit must be <= 200."""
        with pytest.raises(ValidationError) as exc_info:
            SignSearchParams(limit=201)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("limit",) for e in errors)

    def test_offset_min(self):
        """Test that offset must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            SignSearchParams(offset=-1)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("offset",) for e in errors)


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_structure(self):
        """Test ErrorResponse structure with error and message."""
        response = ErrorResponse(error="not_found", message="Sign not found")
        assert response.error == "not_found"
        assert response.message == "Sign not found"

    def test_details_optional(self):
        """Test that details field is optional and defaults to None."""
        response = ErrorResponse(error="not_found", message="Sign not found")
        assert response.details is None

        response_with_details = ErrorResponse(
            error="not_found",
            message="Sign not found",
            details={"gloss": "UNKNOWN"}
        )
        assert response_with_details.details == {"gloss": "UNKNOWN"}


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_defaults(self):
        """Test default values for health response."""
        response = HealthResponse()
        assert response.status == "healthy"
        assert response.version == "2.0.0"
        assert response.services == {}


class TestStatsResponse:
    """Tests for StatsResponse model."""

    def test_structure(self):
        """Test StatsResponse structure with all required fields."""
        response = StatsResponse(
            total_signs=100,
            verified_signs=80,
            pending_signs=10,
            imported_signs=5,
            rejected_signs=5,
            categories={"greetings": 20, "numbers": 15}
        )
        assert response.total_signs == 100
        assert response.verified_signs == 80
        assert response.pending_signs == 10
        assert response.imported_signs == 5
        assert response.rejected_signs == 5
        assert response.categories == {"greetings": 20, "numbers": 15}
