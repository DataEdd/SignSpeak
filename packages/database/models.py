"""Data models for sign database."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class SignStatus(Enum):
    """Sign verification status."""
    PENDING = "pending"
    VERIFIED = "verified"
    IMPORTED = "imported"
    REJECTED = "rejected"


class QualityScore(Enum):
    """Quality scores for sign videos."""
    UNUSABLE = 1   # Reject
    POOR = 2       # Major issues, re-record recommended
    ACCEPTABLE = 3 # Some issues, review needed
    GOOD = 4       # Minor issues, usable
    PERFECT = 5    # Clear, correct, well-framed


@dataclass
class VideoInfo:
    """Video file metadata."""
    file: str
    fps: int = 30
    duration_ms: int = 0
    resolution: str = ""


@dataclass
class TimingInfo:
    """Sign timing within video."""
    sign_start_ms: int = 0
    sign_end_ms: int = 0


@dataclass
class LinguisticInfo:
    """ASL linguistic features."""
    handshape: str = ""
    location: str = ""
    movement: str = ""
    two_handed: bool = False


@dataclass
class Sign:
    """A sign in the database."""
    gloss: str
    english: list[str] = field(default_factory=list)
    category: str = ""
    source: str = "recorded"
    status: SignStatus = SignStatus.PENDING
    quality_score: Optional[int] = None
    verified_by: Optional[str] = None
    verified_date: Optional[str] = None
    video: Optional[VideoInfo] = None
    timing: Optional[TimingInfo] = None
    linguistics: Optional[LinguisticInfo] = None
    path: Optional[Path] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "gloss": self.gloss,
            "english": self.english,
            "category": self.category,
            "source": self.source,
        }
        if self.quality_score is not None:
            data["quality_score"] = self.quality_score
        if self.verified_by:
            data["verified_by"] = self.verified_by
        if self.verified_date:
            data["verified_date"] = self.verified_date
        if self.video:
            data["video"] = {
                "file": self.video.file,
                "fps": self.video.fps,
                "duration_ms": self.video.duration_ms,
                "resolution": self.video.resolution,
            }
        if self.timing:
            data["timing"] = {
                "sign_start_ms": self.timing.sign_start_ms,
                "sign_end_ms": self.timing.sign_end_ms,
            }
        if self.linguistics:
            data["linguistics"] = {
                "handshape": self.linguistics.handshape,
                "location": self.linguistics.location,
                "movement": self.linguistics.movement,
                "two_handed": self.linguistics.two_handed,
            }
        return data

    @classmethod
    def from_dict(cls, data: dict, status: SignStatus, path: Path) -> "Sign":
        """Create Sign from dictionary."""
        video = None
        if "video" in data:
            v = data["video"]
            video = VideoInfo(
                file=v.get("file", ""),
                fps=v.get("fps", 30),
                duration_ms=v.get("duration_ms", 0),
                resolution=v.get("resolution", ""),
            )

        timing = None
        if "timing" in data:
            t = data["timing"]
            timing = TimingInfo(
                sign_start_ms=t.get("sign_start_ms", 0),
                sign_end_ms=t.get("sign_end_ms", 0),
            )

        linguistics = None
        if "linguistics" in data:
            lng = data["linguistics"]
            linguistics = LinguisticInfo(
                handshape=lng.get("handshape", ""),
                location=lng.get("location", ""),
                movement=lng.get("movement", ""),
                two_handed=lng.get("two_handed", False),
            )

        return cls(
            gloss=data.get("gloss", ""),
            english=data.get("english", []),
            category=data.get("category", ""),
            source=data.get("source", "recorded"),
            status=status,
            quality_score=data.get("quality_score"),
            verified_by=data.get("verified_by"),
            verified_date=data.get("verified_date"),
            video=video,
            timing=timing,
            linguistics=linguistics,
            path=path,
        )
