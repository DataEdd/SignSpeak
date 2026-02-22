"""Video serving endpoint routes."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..schemas import ErrorResponse
from ..dependencies import get_video_service
from ..services import VideoService


router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.get(
    "/{video_id}",
    responses={
        404: {"model": ErrorResponse, "description": "Video not found"},
    },
    summary="Get/stream generated video",
)
async def get_video(
    video_id: str,
    video_service: VideoService = Depends(get_video_service),
):
    """
    Stream a generated translation video.

    Videos are cached temporarily and may be cleaned up after 24 hours.
    """
    # Strip extension if provided
    if "." in video_id:
        video_id = video_id.rsplit(".", 1)[0]

    video_path = video_service.get_video_path(video_id)

    if not video_path or not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "video_not_found",
                "message": f"Video '{video_id}' not found or has expired",
                "details": {"video_id": video_id},
            },
        )

    # Determine media type from extension
    ext = video_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".gif": "image/gif",
    }
    media_type = media_types.get(ext, "video/mp4")

    return FileResponse(
        path=video_path,
        media_type=media_type,
        filename=video_path.name,
    )


@router.delete(
    "/{video_id}",
    status_code=204,
    responses={
        404: {"model": ErrorResponse, "description": "Video not found"},
    },
    summary="Delete generated video",
)
async def delete_video(
    video_id: str,
    video_service: VideoService = Depends(get_video_service),
):
    """Delete a generated video from the cache."""
    # Strip extension if provided
    if "." in video_id:
        video_id = video_id.rsplit(".", 1)[0]

    deleted = video_service.delete_video(video_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "video_not_found",
                "message": f"Video '{video_id}' not found",
            },
        )

    return None


@router.get(
    "",
    summary="Get video cache stats",
)
async def get_cache_stats(
    video_service: VideoService = Depends(get_video_service),
):
    """Get statistics about the video cache."""
    return video_service.get_cache_stats()


@router.post(
    "/cleanup",
    summary="Clean up old videos",
)
async def cleanup_cache(
    max_age_hours: int = 24,
    video_service: VideoService = Depends(get_video_service),
):
    """
    Remove videos older than the specified age.

    Returns the number of files deleted.
    """
    deleted = video_service.cleanup_cache(max_age_hours)
    return {"deleted_count": deleted, "max_age_hours": max_age_hours}
