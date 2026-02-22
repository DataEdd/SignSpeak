"""Translation service - orchestrates text-to-ASL translation."""

from pathlib import Path
from typing import Optional
import uuid

from packages.translation import translate, GlossSequence
from packages.database import SignStore

from .video_service import VideoService


class TranslationService:
    """Handles translation from English text to ASL glosses and video."""

    def __init__(
        self,
        sign_store: SignStore,
        video_service: VideoService,
    ):
        self.sign_store = sign_store
        self.video_service = video_service

    def translate_text(
        self,
        text: str,
        speed: str = "normal",
        video_format: str = "mp4",
    ) -> dict:
        """
        Translate English text to ASL glosses and generate video.

        Args:
            text: English text to translate
            speed: Playback speed (slow, normal, fast)
            video_format: Output format (mp4, webm, gif)

        Returns:
            Dictionary with glosses, video_url, confidence, etc.
        """
        # Perform translation
        result: GlossSequence = translate(text, store=self.sign_store)

        # Generate video from glosses
        video_id = str(uuid.uuid4())[:8]
        video_path = self.video_service.create_video(
            glosses=result.glosses,
            video_id=video_id,
            speed=speed,
            format=video_format,
        )

        # Determine missing signs from validation
        missing_signs = []
        if result.validation:
            missing_signs = result.validation.missing_glosses

        return {
            "glosses": result.glosses,
            "video_url": f"/api/videos/{video_id}.{video_format}",
            "confidence": result.confidence,
            "quality": result.quality.value if hasattr(result.quality, 'value') else str(result.quality),
            "missing_signs": missing_signs,
            "fingerspelled": result.fingerspelled,
        }

    def get_gloss_preview(self, text: str) -> dict:
        """
        Get translation preview without generating video.

        Useful for showing the user what glosses will be used
        before committing to video generation.
        """
        result: GlossSequence = translate(text, store=self.sign_store)

        missing_signs = []
        available_signs = []

        if result.validation:
            missing_signs = result.validation.missing_glosses
            # Signs that exist in database
            available_signs = [g for g in result.glosses if g not in missing_signs]
        else:
            available_signs = result.glosses

        return {
            "glosses": result.glosses,
            "available_signs": available_signs,
            "missing_signs": missing_signs,
            "fingerspelled": result.fingerspelled,
            "confidence": result.confidence,
            "quality": result.quality.value if hasattr(result.quality, 'value') else str(result.quality),
        }
