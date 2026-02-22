"""
Clip Manager - Load, cache, and trim video clips from the sign database.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoClip:
    """Represents a loaded video clip with metadata."""

    gloss: str
    frames: np.ndarray  # Shape: (num_frames, height, width, channels)
    fps: float
    metadata: Dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_ms(self) -> float:
        return (self.num_frames / self.fps) * 1000

    @property
    def resolution(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        if self.num_frames == 0:
            return (0, 0)
        return (self.frames.shape[2], self.frames.shape[1])

    def trim(self, start_ms: Optional[float] = None, end_ms: Optional[float] = None) -> "VideoClip":
        """Return a trimmed copy of this clip."""
        start_frame = 0
        end_frame = self.num_frames

        if start_ms is not None:
            start_frame = int((start_ms / 1000) * self.fps)
        if end_ms is not None:
            end_frame = int((end_ms / 1000) * self.fps)

        start_frame = max(0, min(start_frame, self.num_frames))
        end_frame = max(start_frame, min(end_frame, self.num_frames))

        return VideoClip(
            gloss=self.gloss,
            frames=self.frames[start_frame:end_frame].copy(),
            fps=self.fps,
            metadata=self.metadata.copy()
        )

    def resize(self, width: int, height: int) -> "VideoClip":
        """Return a resized copy of this clip."""
        if self.num_frames == 0:
            return VideoClip(
                gloss=self.gloss,
                frames=np.empty((0, height, width, 3), dtype=np.uint8),
                fps=self.fps,
                metadata=self.metadata.copy()
            )

        resized_frames = np.array([
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            for frame in self.frames
        ])

        return VideoClip(
            gloss=self.gloss,
            frames=resized_frames,
            fps=self.fps,
            metadata=self.metadata.copy()
        )


class ClipManager:
    """Manages loading, caching, and retrieval of sign video clips."""

    def __init__(self, signs_dir: Path, search_dirs: Optional[List[str]] = None):
        """
        Initialize the clip manager.

        Args:
            signs_dir: Base directory containing sign subdirectories
            search_dirs: Subdirectories to search in order (default: ["verified", "pending"])
        """
        self.signs_dir = Path(signs_dir)
        self.search_dirs = search_dirs or ["verified", "pending"]
        self._cache: Dict[str, VideoClip] = {}
        self._metadata_cache: Dict[str, Dict] = {}

    def _find_sign_dir(self, gloss: str) -> Optional[Path]:
        """Find the directory containing a sign."""
        gloss_upper = gloss.upper()

        for subdir in self.search_dirs:
            sign_path = self.signs_dir / subdir / gloss_upper
            if sign_path.exists():
                return sign_path

        # Also check root level
        root_path = self.signs_dir / gloss_upper
        if root_path.exists():
            return root_path

        return None

    def _load_metadata(self, sign_dir: Path) -> Dict:
        """Load metadata from sign.json."""
        metadata_path = sign_dir / "sign.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}

    def _load_video(self, video_path: Path) -> Tuple[np.ndarray, float]:
        """Load video frames from file."""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        if not frames:
            raise ValueError(f"No frames loaded from: {video_path}")

        return np.array(frames), fps

    def get_clip(self, gloss: str, use_cache: bool = True) -> VideoClip:
        """
        Get a video clip for a sign.

        Args:
            gloss: The sign gloss (e.g., "HELLO")
            use_cache: Whether to use/update the cache

        Returns:
            VideoClip instance

        Raises:
            FileNotFoundError: If the sign doesn't exist
            ValueError: If the video can't be loaded
        """
        gloss_upper = gloss.upper()

        if use_cache and gloss_upper in self._cache:
            return self._cache[gloss_upper]

        sign_dir = self._find_sign_dir(gloss_upper)
        if sign_dir is None:
            raise FileNotFoundError(f"Sign not found: {gloss}")

        # Find video file
        video_path = None
        for ext in [".mp4", ".webm", ".mov", ".avi"]:
            candidate = sign_dir / f"video{ext}"
            if candidate.exists():
                video_path = candidate
                break

        if video_path is None:
            # Try any video file in the directory
            for ext in [".mp4", ".webm", ".mov", ".avi"]:
                candidates = list(sign_dir.glob(f"*{ext}"))
                if candidates:
                    video_path = candidates[0]
                    break

        if video_path is None:
            raise FileNotFoundError(f"No video file found for sign: {gloss}")

        # Load metadata and video
        metadata = self._load_metadata(sign_dir)
        frames, fps = self._load_video(video_path)

        clip = VideoClip(
            gloss=gloss_upper,
            frames=frames,
            fps=fps,
            metadata=metadata
        )

        # Apply timing trim if specified in metadata
        timing = metadata.get("timing", {})
        if timing.get("sign_start_ms") or timing.get("sign_end_ms"):
            clip = clip.trim(
                start_ms=timing.get("sign_start_ms"),
                end_ms=timing.get("sign_end_ms")
            )

        if use_cache:
            self._cache[gloss_upper] = clip

        return clip

    def preload(self, glosses: List[str]) -> Dict[str, bool]:
        """
        Preload multiple clips into cache.

        Args:
            glosses: List of sign glosses to preload

        Returns:
            Dict mapping gloss to success status
        """
        results = {}
        for gloss in glosses:
            try:
                self.get_clip(gloss, use_cache=True)
                results[gloss.upper()] = True
            except (FileNotFoundError, ValueError) as e:
                results[gloss.upper()] = False
        return results

    def clear_cache(self, gloss: Optional[str] = None):
        """Clear cache for a specific gloss or all glosses."""
        if gloss is None:
            self._cache.clear()
        else:
            self._cache.pop(gloss.upper(), None)

    def list_available(self) -> List[str]:
        """List all available sign glosses."""
        glosses = set()

        for subdir in self.search_dirs:
            search_path = self.signs_dir / subdir
            if search_path.exists():
                for sign_dir in search_path.iterdir():
                    if sign_dir.is_dir():
                        glosses.add(sign_dir.name)

        return sorted(glosses)

    def get_metadata(self, gloss: str) -> Dict:
        """Get metadata for a sign without loading the video."""
        gloss_upper = gloss.upper()

        if gloss_upper in self._metadata_cache:
            return self._metadata_cache[gloss_upper]

        sign_dir = self._find_sign_dir(gloss_upper)
        if sign_dir is None:
            raise FileNotFoundError(f"Sign not found: {gloss}")

        metadata = self._load_metadata(sign_dir)
        self._metadata_cache[gloss_upper] = metadata
        return metadata
