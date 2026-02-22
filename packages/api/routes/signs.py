"""Sign CRUD endpoint routes."""

from pathlib import Path
from typing import Optional
import tempfile
import shutil

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse

from ..schemas import (
    SignResponse,
    SignListResponse,
    SignVerifyRequest,
    SignStatus,
    ErrorResponse,
    StatsResponse,
)
from ..dependencies import get_sign_service
from ..services import SignService


router = APIRouter(prefix="/api/signs", tags=["signs"])


@router.get(
    "",
    response_model=SignListResponse,
    summary="List signs",
)
async def list_signs(
    status: Optional[SignStatus] = Query(None, description="Filter by status"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results to skip"),
    sign_service: SignService = Depends(get_sign_service),
):
    """
    List all signs with optional filtering.

    By default, returns all verified signs. Use the status parameter
    to filter by verification status.
    """
    # Default to verified signs only
    filter_status = status.value if status else "verified"

    result = sign_service.list_signs(
        status=filter_status,
        category=category,
        limit=limit,
        offset=offset,
    )

    return SignListResponse(
        signs=[SignResponse(**s) for s in result["signs"]],
        total=result["total"],
    )


@router.get(
    "/search",
    response_model=list[SignResponse],
    summary="Search signs",
)
async def search_signs(
    q: str = Query(..., min_length=1, description="Search query"),
    status: Optional[SignStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    sign_service: SignService = Depends(get_sign_service),
):
    """
    Search signs by gloss, English translations, or category.
    """
    filter_status = status.value if status else None

    results = sign_service.search_signs(
        query=q,
        status=filter_status,
        limit=limit,
    )

    return [SignResponse(**s) for s in results]


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get database statistics",
)
async def get_stats(
    sign_service: SignService = Depends(get_sign_service),
):
    """Get statistics about the sign database."""
    return StatsResponse(**sign_service.get_stats())


@router.get(
    "/{gloss}",
    response_model=SignResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Sign not found"},
    },
    summary="Get sign details",
)
async def get_sign(
    gloss: str,
    sign_service: SignService = Depends(get_sign_service),
):
    """Get details for a specific sign by gloss."""
    sign = sign_service.get_sign(gloss.upper())

    if not sign:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "sign_not_found",
                "message": f"Sign '{gloss.upper()}' not found in database",
                "details": {"gloss": gloss.upper()},
            },
        )

    return SignResponse(**sign)


@router.get(
    "/{gloss}/video",
    responses={
        404: {"model": ErrorResponse, "description": "Sign or video not found"},
    },
    summary="Get sign video",
)
async def get_sign_video(
    gloss: str,
    sign_service: SignService = Depends(get_sign_service),
):
    """Stream the video for a specific sign."""
    sign = sign_service.get_sign(gloss.upper())

    if not sign:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "sign_not_found",
                "message": f"Sign '{gloss.upper()}' not found",
            },
        )

    # Get the video path from the sign store
    sign_obj = sign_service.store.get_sign(gloss.upper())
    if not sign_obj or not sign_obj.path:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "video_not_found",
                "message": f"No video found for sign '{gloss.upper()}'",
            },
        )

    video_path = sign_obj.path / "video.mp4"
    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "video_not_found",
                "message": f"Video file missing for sign '{gloss.upper()}'",
            },
        )

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"{gloss.upper()}.mp4",
    )


@router.post(
    "",
    response_model=SignResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        409: {"model": ErrorResponse, "description": "Sign already exists"},
    },
    summary="Add new sign",
)
async def create_sign(
    gloss: str = Form(..., description="ASL gloss (e.g., HELLO)"),
    english: str = Form("", description="Comma-separated English translations"),
    category: str = Form("", description="Sign category"),
    source: str = Form("recorded", description="Source of the sign"),
    video: UploadFile = File(..., description="Video file"),
    sign_service: SignService = Depends(get_sign_service),
):
    """
    Add a new sign to the database.

    The sign will be created with 'pending' status and must be
    verified before it can be used in production.
    """
    # Check if sign already exists
    existing = sign_service.get_sign(gloss.upper())
    if existing:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "sign_exists",
                "message": f"Sign '{gloss.upper()}' already exists",
                "details": {"gloss": gloss.upper(), "status": existing["status"]},
            },
        )

    # Parse English translations
    english_list = [e.strip() for e in english.split(",") if e.strip()] if english else []

    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = sign_service.create_sign(
            gloss=gloss,
            video_path=tmp_path,
            english=english_list,
            category=category,
            source=source,
        )
        return SignResponse(**result)
    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


@router.put(
    "/{gloss}/verify",
    response_model=SignResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Sign not found"},
        400: {"model": ErrorResponse, "description": "Sign cannot be verified"},
    },
    summary="Verify a pending sign",
)
async def verify_sign(
    gloss: str,
    request: SignVerifyRequest,
    sign_service: SignService = Depends(get_sign_service),
):
    """
    Verify a pending sign and move it to verified status.

    Only signs with 'pending' or 'imported' status can be verified.
    """
    # Check if sign exists
    sign = sign_service.get_sign(gloss.upper())
    if not sign:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "sign_not_found",
                "message": f"Sign '{gloss.upper()}' not found",
            },
        )

    # Check if sign can be verified
    if sign["status"] not in ["pending", "imported"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_status",
                "message": f"Sign '{gloss.upper()}' has status '{sign['status']}' and cannot be verified",
                "details": {"current_status": sign["status"]},
            },
        )

    result = sign_service.verify_sign(
        gloss=gloss,
        quality_score=request.quality_score,
        verified_by=request.verified_by,
    )

    if not result:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "verification_failed",
                "message": "Failed to verify sign",
            },
        )

    return SignResponse(**result)


@router.put(
    "/{gloss}/reject",
    response_model=SignResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Sign not found"},
    },
    summary="Reject a pending sign",
)
async def reject_sign(
    gloss: str,
    reason: str = Form("", description="Reason for rejection"),
    sign_service: SignService = Depends(get_sign_service),
):
    """Reject a pending sign."""
    sign = sign_service.get_sign(gloss.upper())
    if not sign:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "sign_not_found",
                "message": f"Sign '{gloss.upper()}' not found",
            },
        )

    result = sign_service.reject_sign(gloss, reason)
    if not result:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "rejection_failed",
                "message": "Failed to reject sign",
            },
        )

    return SignResponse(**result)


@router.delete(
    "/{gloss}",
    status_code=204,
    responses={
        404: {"model": ErrorResponse, "description": "Sign not found"},
    },
    summary="Delete a sign",
)
async def delete_sign(
    gloss: str,
    sign_service: SignService = Depends(get_sign_service),
):
    """Delete a sign from the database."""
    deleted = sign_service.delete_sign(gloss)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "sign_not_found",
                "message": f"Sign '{gloss.upper()}' not found",
            },
        )

    return None
