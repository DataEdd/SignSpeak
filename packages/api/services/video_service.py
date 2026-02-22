"""Video service - handles video generation and management."""

from pathlib import Path
from typing import Optional
import os

from packages.video import (
    ClipManager,
    Compositor,
    VideoExporter,
    ExportFormat,
    ExportSettings,
    TransitionType,
)


class VideoService:
    """Handles video composition, export, and caching."""

    # Speed multipliers for playback
    SPEED_FPS_MAP = {
        "slow": 20,
        "normal": 30,
        "fast": 40,
    }

    FORMAT_MAP = {
        "mp4": ExportFormat.MP4,
        "webm": ExportFormat.WEBM,
        "gif": ExportFormat.GIF,
    }

    def __init__(
        self,
        signs_dir: Path,
        cache_dir: Path,
        default_fps: int = 30,
        resolution: tuple[int, int] = (720, 540),
    ):
        self.signs_dir = Path(signs_dir)
        self.cache_dir = Path(cache_dir)
        self.default_fps = default_fps
        self.resolution = resolution

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize clip manager and exporter
        self.clip_manager = ClipManager(
            signs_dir=self.signs_dir,
            search_dirs=["verified", "pending"],
        )
        self.exporter = VideoExporter()

    def create_video(
        self,
        glosses: list[str],
        video_id: str,
        speed: str = "normal",
        format: str = "mp4",
        transition_type: TransitionType = TransitionType.CROSSFADE,
        transition_ms: int = 100,
    ) -> Path:
        """
        Create a video from a sequence of glosses.

        Args:
            glosses: List of ASL glosses to combine
            video_id: Unique identifier for the video
            speed: Playback speed (slow, normal, fast)
            format: Output format (mp4, webm, gif)
            transition_type: Type of transition between clips
            transition_ms: Transition duration in milliseconds

        Returns:
            Path to the generated video file
        """
        fps = self.SPEED_FPS_MAP.get(speed, self.default_fps)
        export_format = self.FORMAT_MAP.get(format, ExportFormat.MP4)

        # Build compositor
        compositor = Compositor(
            clip_manager=self.clip_manager,
            fps=fps,
            resolution=self.resolution,
        )

        # Add clips with transitions
        for i, gloss in enumerate(glosses):
            compositor.add_clip(gloss)
            # Add transition between clips (not after last one)
            if i < len(glosses) - 1:
                compositor.add_transition(transition_type, duration_ms=transition_ms)

        # Compose the video
        video_clip = compositor.compose()

        # Export to file
        output_path = self.cache_dir / f"{video_id}.{format}"
        settings = ExportSettings(
            format=export_format,
            quality="high",
        )

        self.exporter.export(
            frames=video_clip.frames,
            output_path=str(output_path),
            fps=fps,
            settings=settings,
        )

        return output_path

    def get_video_path(self, video_id: str) -> Optional[Path]:
        """
        Get the path to a cached video by ID.

        Searches for the video with any supported extension.
        """
        for ext in ["mp4", "webm", "gif"]:
            path = self.cache_dir / f"{video_id}.{ext}"
            if path.exists():
                return path
        return None

    def delete_video(self, video_id: str) -> bool:
        """Delete a cached video by ID."""
        path = self.get_video_path(video_id)
        if path and path.exists():
            path.unlink()
            return True
        return False

    def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """
        Remove videos older than max_age_hours.

        Returns the number of files deleted.
        """
        import time

        deleted = 0
        cutoff = time.time() - (max_age_hours * 3600)

        for path in self.cache_dir.iterdir():
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1

        return deleted

    def get_cache_stats(self) -> dict:
        """Get statistics about the video cache."""
        total_size = 0
        file_count = 0

        for path in self.cache_dir.iterdir():
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1

        return {
            "file_count": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }
