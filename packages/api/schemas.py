"""Pydantic schemas for API request/response models."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class VideoFormat(str, Enum):
    """Supported video output formats."""
    MP4 = "mp4"
    WEBM = "webm"
    GIF = "gif"


class TranslationSpeed(str, Enum):
    """Video playback speed options."""
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"


# ============ Translation Schemas ============

class TranslateRequest(BaseModel):
    """Request body for translation endpoint."""
    text: str = Field(..., min_length=1, max_length=1000, description="English text to translate")
    options: Optional["TranslationOptions"] = None


class TranslationOptions(BaseModel):
    """Optional parameters for translation."""
    speed: TranslationSpeed = TranslationSpeed.NORMAL
    format: VideoFormat = VideoFormat.MP4
    include_fingerspelling: bool = True


class TranslateResponse(BaseModel):
    """Response from translation endpoint."""
    glosses: list[str] = Field(..., description="ASL glosses representing the translation")
    video_url: str = Field(..., description="URL to stream the generated video")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Translation confidence score")
    quality: str = Field(..., description="Translation quality level")
    missing_signs: list[str] = Field(default_factory=list, description="Signs not in database")
    fingerspelled: list[str] = Field(default_factory=list, description="Words that were fingerspelled")


# ============ Sign Schemas ============

class SignStatus(str, Enum):
    """Sign verification status."""
    PENDING = "pending"
    VERIFIED = "verified"
    IMPORTED = "imported"
    REJECTED = "rejected"


class SignResponse(BaseModel):
    """Response for a single sign."""
    gloss: str
    english: list[str]
    category: str
    source: str
    status: SignStatus
    quality_score: Optional[int] = None
    verified_by: Optional[str] = None
    verified_date: Optional[str] = None
    video_url: Optional[str] = None


class SignListResponse(BaseModel):
    """Response for listing signs."""
    signs: list[SignResponse]
    total: int


class SignCreateRequest(BaseModel):
    """Request to create a new sign (used with multipart form)."""
    gloss: str = Field(..., min_length=1, max_length=50)
    english: list[str] = Field(default_factory=list)
    category: str = ""
    source: str = "recorded"


class SignVerifyRequest(BaseModel):
    """Request to verify a sign."""
    quality_score: int = Field(..., ge=1, le=5, description="Quality score 1-5")
    verified_by: str = Field(..., min_length=1, description="Name of verifier")


class SignSearchParams(BaseModel):
    """Query parameters for sign search."""
    q: Optional[str] = Field(None, description="Search query")
    status: Optional[SignStatus] = None
    category: Optional[str] = None
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)


# ============ Error Schemas ============

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = None


# ============ Health Schemas ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "2.0.0"
    services: dict[str, str] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    """Database statistics response."""
    total_signs: int
    verified_signs: int
    pending_signs: int
    imported_signs: int
    rejected_signs: int
    categories: dict[str, int]


# Forward reference resolution
TranslateRequest.model_rebuild()
