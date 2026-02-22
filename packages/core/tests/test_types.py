"""Tests for shared type definitions."""

import pytest

from packages.core.types import (
    LinguisticInfo,
    QualityScore,
    SignStatus,
    TimingInfo,
    VerificationInfo,
    VideoMetadata,
)


class TestSignStatus:
    """Tests for SignStatus enum."""

    def test_values(self):
        """Test enum values."""
        assert SignStatus.PENDING.value == "pending"
        assert SignStatus.VERIFIED.value == "verified"
        assert SignStatus.IMPORTED.value == "imported"
        assert SignStatus.REJECTED.value == "rejected"

    def test_from_string(self):
        """Test creating from string value."""
        assert SignStatus("pending") == SignStatus.PENDING
        assert SignStatus("verified") == SignStatus.VERIFIED

    def test_invalid_value(self):
        """Test invalid value raises error."""
        with pytest.raises(ValueError):
            SignStatus("invalid")


class TestQualityScore:
    """Tests for QualityScore enum."""

    def test_values(self):
        """Test enum values."""
        assert QualityScore.UNUSABLE == 1
        assert QualityScore.POOR == 2
        assert QualityScore.ACCEPTABLE == 3
        assert QualityScore.GOOD == 4
        assert QualityScore.PERFECT == 5

    def test_int_comparison(self):
        """Test integer comparison (IntEnum behavior)."""
        assert QualityScore.GOOD > QualityScore.ACCEPTABLE
        assert QualityScore.PERFECT >= 5
        assert QualityScore.UNUSABLE < 2

    def test_int_operations(self):
        """Test integer operations."""
        assert QualityScore.GOOD + 1 == 5
        assert int(QualityScore.PERFECT) == 5


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""

    def test_basic_creation(self):
        """Test creating VideoMetadata."""
        video = VideoMetadata(file="video.mp4", fps=30, duration_ms=1000, resolution="720x540")

        assert video.file == "video.mp4"
        assert video.fps == 30
        assert video.duration_ms == 1000
        assert video.resolution == "720x540"

    def test_defaults(self):
        """Test default values."""
        video = VideoMetadata(file="video.mp4")

        assert video.fps == 30
        assert video.duration_ms == 0
        assert video.resolution == ""

    def test_resolution_tuple(self):
        """Test resolution_tuple property."""
        video = VideoMetadata(file="video.mp4", resolution="720x540")

        assert video.resolution_tuple == (720, 540)

    def test_resolution_tuple_empty(self):
        """Test resolution_tuple with empty resolution."""
        video = VideoMetadata(file="video.mp4")

        assert video.resolution_tuple == (0, 0)

    def test_width_height(self):
        """Test width and height properties."""
        video = VideoMetadata(file="video.mp4", resolution="1920x1080")

        assert video.width == 1920
        assert video.height == 1080

    def test_to_dict(self):
        """Test converting to dictionary."""
        video = VideoMetadata(file="video.mp4", fps=30, duration_ms=1000, resolution="720x540")

        result = video.to_dict()

        assert result == {
            "file": "video.mp4",
            "fps": 30,
            "duration_ms": 1000,
            "resolution": "720x540",
        }

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "file": "video.mp4",
            "fps": 60,
            "duration_ms": 2000,
            "resolution": "1080x720",
        }

        video = VideoMetadata.from_dict(data)

        assert video.file == "video.mp4"
        assert video.fps == 60
        assert video.duration_ms == 2000
        assert video.resolution == "1080x720"

    def test_from_dict_with_tuple_resolution(self):
        """Test from_dict handles tuple resolution."""
        data = {"file": "video.mp4", "resolution": [1920, 1080]}

        video = VideoMetadata.from_dict(data)

        assert video.resolution == "1920x1080"

    def test_round_trip(self):
        """Test to_dict/from_dict round trip."""
        original = VideoMetadata(file="test.mp4", fps=60, duration_ms=5000, resolution="1080x720")

        result = VideoMetadata.from_dict(original.to_dict())

        assert result.file == original.file
        assert result.fps == original.fps
        assert result.duration_ms == original.duration_ms
        assert result.resolution == original.resolution


class TestTimingInfo:
    """Tests for TimingInfo dataclass."""

    def test_basic_creation(self):
        """Test creating TimingInfo."""
        timing = TimingInfo(sign_start_ms=100, sign_end_ms=1000)

        assert timing.sign_start_ms == 100
        assert timing.sign_end_ms == 1000

    def test_defaults(self):
        """Test default values."""
        timing = TimingInfo()

        assert timing.sign_start_ms == 0
        assert timing.sign_end_ms == 0

    def test_duration_ms(self):
        """Test duration_ms property."""
        timing = TimingInfo(sign_start_ms=100, sign_end_ms=1000)

        assert timing.duration_ms == 900

    def test_to_dict(self):
        """Test converting to dictionary."""
        timing = TimingInfo(sign_start_ms=100, sign_end_ms=1000)

        result = timing.to_dict()

        assert result == {"sign_start_ms": 100, "sign_end_ms": 1000}

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"sign_start_ms": 200, "sign_end_ms": 800}

        timing = TimingInfo.from_dict(data)

        assert timing.sign_start_ms == 200
        assert timing.sign_end_ms == 800

    def test_round_trip(self):
        """Test to_dict/from_dict round trip."""
        original = TimingInfo(sign_start_ms=150, sign_end_ms=950)

        result = TimingInfo.from_dict(original.to_dict())

        assert result.sign_start_ms == original.sign_start_ms
        assert result.sign_end_ms == original.sign_end_ms


class TestLinguisticInfo:
    """Tests for LinguisticInfo dataclass."""

    def test_basic_creation(self):
        """Test creating LinguisticInfo."""
        linguistics = LinguisticInfo(
            handshape="B",
            location="forehead",
            movement="outward",
            two_handed=False,
        )

        assert linguistics.handshape == "B"
        assert linguistics.location == "forehead"
        assert linguistics.movement == "outward"
        assert linguistics.two_handed is False

    def test_defaults(self):
        """Test default values."""
        linguistics = LinguisticInfo()

        assert linguistics.handshape == ""
        assert linguistics.location == ""
        assert linguistics.movement == ""
        assert linguistics.two_handed is False

    def test_to_dict(self):
        """Test converting to dictionary."""
        linguistics = LinguisticInfo(
            handshape="5",
            location="chest",
            movement="circular",
            two_handed=True,
        )

        result = linguistics.to_dict()

        assert result == {
            "handshape": "5",
            "location": "chest",
            "movement": "circular",
            "two_handed": True,
        }

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "handshape": "A",
            "location": "chin",
            "movement": "tap",
            "two_handed": False,
        }

        linguistics = LinguisticInfo.from_dict(data)

        assert linguistics.handshape == "A"
        assert linguistics.location == "chin"
        assert linguistics.movement == "tap"
        assert linguistics.two_handed is False

    def test_round_trip(self):
        """Test to_dict/from_dict round trip."""
        original = LinguisticInfo(
            handshape="C",
            location="neutral space",
            movement="upward",
            two_handed=True,
        )

        result = LinguisticInfo.from_dict(original.to_dict())

        assert result.handshape == original.handshape
        assert result.location == original.location
        assert result.movement == original.movement
        assert result.two_handed == original.two_handed


class TestVerificationInfo:
    """Tests for VerificationInfo dataclass."""

    def test_basic_creation(self):
        """Test creating VerificationInfo."""
        verification = VerificationInfo(
            verified_by="expert",
            verified_date="2024-01-15",
            quality_score=5,
            notes="Clear and well-framed",
        )

        assert verification.verified_by == "expert"
        assert verification.verified_date == "2024-01-15"
        assert verification.quality_score == 5
        assert verification.notes == "Clear and well-framed"

    def test_defaults(self):
        """Test default notes."""
        verification = VerificationInfo(
            verified_by="tester",
            verified_date="2024-01-15",
            quality_score=4,
        )

        assert verification.notes == ""

    def test_to_dict(self):
        """Test converting to dictionary."""
        verification = VerificationInfo(
            verified_by="expert",
            verified_date="2024-01-15",
            quality_score=5,
            notes="Good",
        )

        result = verification.to_dict()

        assert result == {
            "verified_by": "expert",
            "verified_date": "2024-01-15",
            "quality_score": 5,
            "notes": "Good",
        }

    def test_to_dict_without_notes(self):
        """Test to_dict without notes."""
        verification = VerificationInfo(
            verified_by="expert",
            verified_date="2024-01-15",
            quality_score=5,
        )

        result = verification.to_dict()

        assert "notes" not in result

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "verified_by": "tester",
            "verified_date": "2024-02-01",
            "quality_score": 4,
            "notes": "Minor issues",
        }

        verification = VerificationInfo.from_dict(data)

        assert verification.verified_by == "tester"
        assert verification.verified_date == "2024-02-01"
        assert verification.quality_score == 4
        assert verification.notes == "Minor issues"

    def test_round_trip(self):
        """Test to_dict/from_dict round trip."""
        original = VerificationInfo(
            verified_by="admin",
            verified_date="2024-03-01",
            quality_score=3,
            notes="Some blur",
        )

        result = VerificationInfo.from_dict(original.to_dict())

        assert result.verified_by == original.verified_by
        assert result.verified_date == original.verified_date
        assert result.quality_score == original.quality_score
        assert result.notes == original.notes
