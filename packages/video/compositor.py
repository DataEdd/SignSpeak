"""
Compositor - Stitch video clips together with transitions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .clip_manager import ClipManager, VideoClip
from .exporter import ExportFormat, ExportSettings, VideoExporter
from .transitions import TransitionType, apply_transition, ms_to_frames


@dataclass
class CompositorSegment:
    """A segment in the composition timeline."""
    clip: VideoClip
    transition_out: Optional[TransitionType] = None
    transition_duration_ms: float = 150


@dataclass
class CompositorSettings:
    """Settings for the compositor."""
    fps: float = 30.0
    resolution: Tuple[int, int] = (720, 540)  # (width, height)
    default_transition: TransitionType = TransitionType.CROSSFADE
    default_transition_duration_ms: float = 150
    background_color: Tuple[int, int, int] = (0, 0, 0)  # RGB


class Compositor:
    """
    Compose sign video clips into a seamless output video.

    Usage:
        comp = Compositor(clip_manager, fps=30, resolution=(720, 540))
        comp.add_clip("HELLO")
        comp.add_transition(TransitionType.CROSSFADE, duration_ms=150)
        comp.add_clip("WORLD")
        comp.export("output.mp4")
    """

    def __init__(
        self,
        clip_manager: Optional[ClipManager] = None,
        signs_dir: Optional[Path] = None,
        fps: float = 30.0,
        resolution: Tuple[int, int] = (720, 540),
        settings: Optional[CompositorSettings] = None
    ):
        """
        Initialize the compositor.

        Args:
            clip_manager: ClipManager instance (or provide signs_dir)
            signs_dir: Path to signs directory (creates ClipManager)
            fps: Output frames per second
            resolution: Output resolution (width, height)
            settings: Full compositor settings
        """
        if settings:
            self.settings = settings
        else:
            self.settings = CompositorSettings(fps=fps, resolution=resolution)

        if clip_manager:
            self.clip_manager = clip_manager
        elif signs_dir:
            self.clip_manager = ClipManager(signs_dir)
        else:
            self.clip_manager = None

        self._segments: List[CompositorSegment] = []
        self._pending_transition: Optional[Tuple[TransitionType, float]] = None

    def add_clip(
        self,
        source: Union[str, VideoClip],
        trim_start_ms: Optional[float] = None,
        trim_end_ms: Optional[float] = None
    ) -> "Compositor":
        """
        Add a clip to the composition.

        Args:
            source: Gloss string or VideoClip instance
            trim_start_ms: Optional start trim point
            trim_end_ms: Optional end trim point

        Returns:
            Self for method chaining
        """
        # Get clip
        if isinstance(source, str):
            if self.clip_manager is None:
                raise ValueError("ClipManager required to load clips by gloss")
            clip = self.clip_manager.get_clip(source)
        else:
            clip = source

        # Apply trim if specified
        if trim_start_ms is not None or trim_end_ms is not None:
            clip = clip.trim(start_ms=trim_start_ms, end_ms=trim_end_ms)

        # Resize to target resolution
        target_w, target_h = self.settings.resolution
        if clip.resolution != (target_w, target_h):
            clip = clip.resize(target_w, target_h)

        # Apply pending transition to previous segment
        if self._pending_transition and self._segments:
            trans_type, trans_duration = self._pending_transition
            self._segments[-1].transition_out = trans_type
            self._segments[-1].transition_duration_ms = trans_duration
            self._pending_transition = None

        # Add segment
        self._segments.append(CompositorSegment(clip=clip))

        return self

    def add_transition(
        self,
        transition_type: Union[TransitionType, str] = TransitionType.CROSSFADE,
        duration_ms: float = 150
    ) -> "Compositor":
        """
        Add a transition before the next clip.

        Args:
            transition_type: Type of transition
            duration_ms: Duration in milliseconds

        Returns:
            Self for method chaining
        """
        if isinstance(transition_type, str):
            transition_type = TransitionType(transition_type.lower())

        self._pending_transition = (transition_type, duration_ms)
        return self

    def add_sequence(
        self,
        glosses: List[str],
        transition_type: Union[TransitionType, str] = TransitionType.CROSSFADE,
        transition_duration_ms: float = 150
    ) -> "Compositor":
        """
        Add multiple clips with the same transition between them.

        Args:
            glosses: List of sign glosses
            transition_type: Transition between clips
            transition_duration_ms: Transition duration

        Returns:
            Self for method chaining
        """
        for i, gloss in enumerate(glosses):
            self.add_clip(gloss)
            if i < len(glosses) - 1:
                self.add_transition(transition_type, transition_duration_ms)

        return self

    def compose(self) -> VideoClip:
        """
        Compose all segments into a single VideoClip.

        Returns:
            Composed VideoClip
        """
        if not self._segments:
            # Return empty clip
            w, h = self.settings.resolution
            return VideoClip(
                gloss="COMPOSED",
                frames=np.empty((0, h, w, 3), dtype=np.uint8),
                fps=self.settings.fps
            )

        if len(self._segments) == 1:
            return self._segments[0].clip

        # Compose segments with transitions
        result_frames = self._segments[0].clip.frames.copy()

        for i in range(len(self._segments) - 1):
            current_seg = self._segments[i]
            next_seg = self._segments[i + 1]

            # Determine transition
            trans_type = current_seg.transition_out or self.settings.default_transition
            trans_duration_ms = current_seg.transition_duration_ms or self.settings.default_transition_duration_ms
            trans_frames = ms_to_frames(trans_duration_ms, self.settings.fps)

            # Apply transition
            result_frames = apply_transition(
                result_frames,
                next_seg.clip.frames,
                trans_type,
                trans_frames,
                self.settings.fps
            )

        return VideoClip(
            gloss="COMPOSED",
            frames=result_frames,
            fps=self.settings.fps
        )

    def export(
        self,
        output_path: str,
        format: Optional[ExportFormat] = None,
        quality: int = 23,
        use_hwaccel: bool = False
    ) -> Path:
        """
        Compose and export to a video file.

        Args:
            output_path: Output file path
            format: Export format (inferred from extension if None)
            quality: Quality setting (0-51, lower is better)
            use_hwaccel: Use hardware acceleration

        Returns:
            Path to exported file
        """
        composed = self.compose()

        export_settings = ExportSettings(
            format=format or ExportFormat.MP4,
            resolution=self.settings.resolution,
            fps=self.settings.fps,
            quality=quality,
            use_hwaccel=use_hwaccel
        )

        exporter = VideoExporter(export_settings)
        return exporter.export(composed.frames, output_path, fps=self.settings.fps)

    def preview_frames(self, max_frames: int = 30) -> np.ndarray:
        """
        Get a subset of frames for preview.

        Args:
            max_frames: Maximum frames to return

        Returns:
            Frame array for preview
        """
        composed = self.compose()

        if composed.num_frames <= max_frames:
            return composed.frames

        # Sample frames evenly
        indices = np.linspace(0, composed.num_frames - 1, max_frames, dtype=int)
        return composed.frames[indices]

    def get_duration_ms(self) -> float:
        """Get estimated total duration in milliseconds."""
        if not self._segments:
            return 0.0

        total_frames = sum(seg.clip.num_frames for seg in self._segments)

        # Subtract overlap from transitions
        for i in range(len(self._segments) - 1):
            seg = self._segments[i]
            if seg.transition_out and seg.transition_out != TransitionType.CUT:
                trans_duration = seg.transition_duration_ms or self.settings.default_transition_duration_ms
                trans_frames = ms_to_frames(trans_duration, self.settings.fps)
                total_frames -= trans_frames

        return (total_frames / self.settings.fps) * 1000

    def clear(self) -> "Compositor":
        """Clear all segments."""
        self._segments.clear()
        self._pending_transition = None
        return self

    @property
    def num_clips(self) -> int:
        """Number of clips in the composition."""
        return len(self._segments)

    @property
    def glosses(self) -> List[str]:
        """List of glosses in the composition."""
        return [seg.clip.gloss for seg in self._segments]


def compose_sequence(
    glosses: List[str],
    signs_dir: Path,
    output_path: str,
    transition_type: TransitionType = TransitionType.CROSSFADE,
    transition_duration_ms: float = 150,
    fps: float = 30.0,
    resolution: Tuple[int, int] = (720, 540)
) -> Path:
    """
    Convenience function to compose a sequence of signs.

    Args:
        glosses: List of sign glosses
        signs_dir: Path to signs directory
        output_path: Output file path
        transition_type: Transition between clips
        transition_duration_ms: Transition duration
        fps: Output FPS
        resolution: Output resolution

    Returns:
        Path to exported file
    """
    comp = Compositor(signs_dir=signs_dir, fps=fps, resolution=resolution)
    comp.add_sequence(glosses, transition_type, transition_duration_ms)
    return comp.export(output_path)
