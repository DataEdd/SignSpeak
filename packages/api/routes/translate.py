"""Translation endpoint routes."""

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import (
    TranslateRequest,
    TranslateResponse,
    ErrorResponse,
)
from ..dependencies import get_translation_service
from ..services import TranslationService


router = APIRouter(prefix="/api", tags=["translation"])


@router.post(
    "/translate",
    response_model=TranslateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Translation failed"},
    },
)
async def translate_text(
    request: TranslateRequest,
    translation_service: TranslationService = Depends(get_translation_service),
):
    """
    Translate English text to ASL glosses and generate video.

    - **text**: The English text to translate (required)
    - **options.speed**: Playback speed - slow, normal, or fast (default: normal)
    - **options.format**: Video format - mp4, webm, or gif (default: mp4)
    - **options.include_fingerspelling**: Whether to fingerspell unknown words (default: true)

    Returns the ASL glosses and a URL to stream the generated video.
    """
    try:
        # Extract options
        speed = "normal"
        video_format = "mp4"

        if request.options:
            speed = request.options.speed.value
            video_format = request.options.format.value

        result = translation_service.translate_text(
            text=request.text,
            speed=speed,
            video_format=video_format,
        )

        return TranslateResponse(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_input",
                "message": str(e),
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "translation_failed",
                "message": f"Failed to translate text: {str(e)}",
            },
        )


@router.post(
    "/translate/preview",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def preview_translation(
    request: TranslateRequest,
    translation_service: TranslationService = Depends(get_translation_service),
):
    """
    Preview translation without generating video.

    Useful for showing the user what glosses will be used
    and which signs are available before committing to video generation.
    """
    try:
        result = translation_service.get_gloss_preview(request.text)
        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_input",
                "message": str(e),
            },
        )
