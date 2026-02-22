"""Shared type definitions for SignBridge.

Contains canonical type definitions used across packages.
Domain-specific types should remain in their respective packages.

Core Types (shared across packages):
- SignStatus: Verification workflow states
- QualityScore: Quality rating for signs
- VideoMetadata: Video file information
- TimingInfo: Sign timing within video
- LinguisticInfo: ASL linguistic features
- VerificationInfo: Verification metadata

Domain-Specific Types (remain in packages):
- database.Sign: Full sign record with path
- translation.TokenType: Grammar parsing tokens
- translation.GlossSequence: Translation output
- video.VideoClip: Loaded video frames
- video.TransitionType: Video transition effects
"""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Optional


# ============ Enums ============


class SignStatus(Enum):
    """Sign verification workflow status.

    Signs progress through: PENDING -> VERIFIED (or REJECTED)
    IMPORTED signs are auto-imported and need review.
    """

    PENDING = "pending"
    VERIFIED = "verified"
    IMPORTED = "imported"
    REJECTED = "rejected"


class QualityScore(IntEnum):
    """Quality score for sign videos (1-5 scale).

    Used during verification to rate sign quality.
    IntEnum allows direct comparison (score >= QualityScore.GOOD).
    """

    UNUSABLE = 1  # Reject - cannot be used
    POOR = 2  # Major issues, re-record recommended
    ACCEPTABLE = 3  # Some issues, usable with review
    GOOD = 4  # Minor issues, usable
    PERFECT = 5  # Clear, correct, well-framed


# ============ Dataclasses ============


@dataclass
class VideoMetadata:
    """Metadata about a video file.

    Stores information about video properties without loading the video.
    """

    file: str
    fps: int = 30
    duration_ms: int = 0
    resolution: str = ""

    @property
    def resolution_tuple(self) -> tuple[int, int]:
        """Get resolution as (width, height) tuple."""
        if not self.resolution or "x" not in self.resolution:
            return (0, 0)
        parts = self.resolution.split("x")
        return (int(parts[0]), int(parts[1]))

    @property
    def width(self) -> int:
        """Get video width in pixels."""
        return self.resolution_tuple[0]

    @property
    def height(self) -> int:
        """Get video height in pixels."""
        return self.resolution_tuple[1]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file": self.file,
            "fps": self.fps,
            "duration_ms": self.duration_ms,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoMetadata":
        """Create from dictionary."""
        resolution = data.get("resolution", "")
        # Handle both string and tuple/list formats
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            resolution = f"{resolution[0]}x{resolution[1]}"

        return cls(
            file=data.get("file", ""),
            fps=data.get("fps", 30),
            duration_ms=data.get("duration_ms", 0),
            resolution=resolution,
        )


@dataclass
class TimingInfo:
    """Timing information for a sign within a video.

    Specifies when the sign starts and ends within the video clip.
    """

    sign_start_ms: int = 0
    sign_end_ms: int = 0

    @property
    def duration_ms(self) -> int:
        """Get duration of the sign in milliseconds."""
        return self.sign_end_ms - self.sign_start_ms

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sign_start_ms": self.sign_start_ms,
            "sign_end_ms": self.sign_end_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimingInfo":
        """Create from dictionary."""
        return cls(
            sign_start_ms=data.get("sign_start_ms", 0),
            sign_end_ms=data.get("sign_end_ms", 0),
        )


@dataclass
class LinguisticInfo:
    """ASL linguistic features of a sign.

    Describes the phonological parameters of the sign:
    - handshape: The shape of the hand(s)
    - location: Where the sign is made relative to the body
    - movement: The motion of the sign
    - two_handed: Whether both hands are used
    """

    handshape: str = ""
    location: str = ""
    movement: str = ""
    two_handed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "handshape": self.handshape,
            "location": self.location,
            "movement": self.movement,
            "two_handed": self.two_handed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LinguisticInfo":
        """Create from dictionary."""
        return cls(
            handshape=data.get("handshape", ""),
            location=data.get("location", ""),
            movement=data.get("movement", ""),
            two_handed=data.get("two_handed", False),
        )


@dataclass
class VerificationInfo:
    """Information about sign verification.

    Tracks who verified a sign, when, and with what quality score.
    """

    verified_by: str
    verified_date: str  # ISO format date string
    quality_score: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "verified_by": self.verified_by,
            "verified_date": self.verified_date,
            "quality_score": self.quality_score,
        }
        if self.notes:
            result["notes"] = self.notes
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationInfo":
        """Create from dictionary."""
        return cls(
            verified_by=data.get("verified_by", ""),
            verified_date=data.get("verified_date", ""),
            quality_score=data.get("quality_score", 3),
            notes=data.get("notes", ""),
        )
