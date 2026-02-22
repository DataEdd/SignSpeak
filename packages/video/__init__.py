# Video package - clip management and compositing

from .clip_manager import ClipManager, VideoClip
from .compositor import Compositor, CompositorSettings, compose_sequence
from .exporter import ExportFormat, ExportSettings, VideoExporter, export_frames
from .transitions import TransitionType, apply_transition, ms_to_frames, frames_to_ms

__all__ = [
    # Clip management
    "ClipManager",
    "VideoClip",
    # Compositing
    "Compositor",
    "CompositorSettings",
    "compose_sequence",
    # Export
    "VideoExporter",
    "ExportSettings",
    "ExportFormat",
    "export_frames",
    # Transitions
    "TransitionType",
    "apply_transition",
    "ms_to_frames",
    "frames_to_ms",
]
