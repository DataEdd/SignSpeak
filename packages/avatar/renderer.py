"""Render pose sequences to video with 3D avatar."""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .pose_extractor import PoseSequence, FramePose


# MediaPipe pose connections for skeleton visualization
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (15, 17), (15, 19), (15, 21), (17, 19),  # Left hand
    (16, 18), (16, 20), (16, 22), (18, 20),  # Right hand
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17),  # Palm
]


@dataclass
class RenderSettings:
    """Settings for avatar rendering."""
    width: int = 720
    height: int = 540
    fps: float = 30.0
    background_color: Tuple[int, int, int] = (30, 30, 30)  # Dark gray
    pose_color: Tuple[int, int, int] = (0, 255, 100)  # Green
    hand_color: Tuple[int, int, int] = (255, 200, 0)  # Yellow
    face_color: Tuple[int, int, int] = (100, 200, 255)  # Light blue
    line_thickness: int = 2
    point_radius: int = 3
    show_face: bool = False  # Face mesh is dense, optional
    render_mode: str = "skeleton"  # "skeleton" or "avatar"


class AvatarRenderer:
    """Render pose sequences as video."""

    def __init__(self, settings: Optional[RenderSettings] = None):
        self.settings = settings or RenderSettings()

    def _normalize_landmarks(
        self,
        landmarks: List,
        width: int,
        height: int,
        center: bool = True,
    ) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates."""
        if landmarks is None:
            return np.array([])

        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        if center:
            # Center horizontally
            points[:, 0] = points[:, 0] * 0.6 + 0.2

        # Convert to pixel coordinates
        points[:, 0] = points[:, 0] * width
        points[:, 1] = points[:, 1] * height

        return points

    def _draw_connections(
        self,
        frame: np.ndarray,
        points: np.ndarray,
        connections: List[Tuple[int, int]],
        color: Tuple[int, int, int],
    ):
        """Draw skeleton connections."""
        if len(points) == 0:
            return

        for start_idx, end_idx in connections:
            if start_idx >= len(points) or end_idx >= len(points):
                continue

            start = points[start_idx][:2].astype(int)
            end = points[end_idx][:2].astype(int)

            # Check if points are valid
            if np.any(start < 0) or np.any(end < 0):
                continue
            if np.any(start > [self.settings.width, self.settings.height]):
                continue
            if np.any(end > [self.settings.width, self.settings.height]):
                continue

            cv2.line(frame, tuple(start), tuple(end), color, self.settings.line_thickness)

    def _draw_points(
        self,
        frame: np.ndarray,
        points: np.ndarray,
        color: Tuple[int, int, int],
    ):
        """Draw landmark points."""
        if len(points) == 0:
            return

        for point in points:
            x, y = point[:2].astype(int)
            if 0 <= x < self.settings.width and 0 <= y < self.settings.height:
                cv2.circle(frame, (x, y), self.settings.point_radius, color, -1)

    def render_frame(self, frame_pose: FramePose) -> np.ndarray:
        """Render a single frame."""
        # Create background
        frame = np.full(
            (self.settings.height, self.settings.width, 3),
            self.settings.background_color,
            dtype=np.uint8,
        )

        w, h = self.settings.width, self.settings.height

        # Draw pose skeleton
        if frame_pose.pose_landmarks:
            pose_points = self._normalize_landmarks(frame_pose.pose_landmarks, w, h)
            self._draw_connections(frame, pose_points, POSE_CONNECTIONS, self.settings.pose_color)
            self._draw_points(frame, pose_points, self.settings.pose_color)

        # Draw hands
        if frame_pose.left_hand_landmarks:
            left_hand = self._normalize_landmarks(frame_pose.left_hand_landmarks, w, h, center=False)
            # Offset to be on left side of body
            if frame_pose.pose_landmarks and len(frame_pose.pose_landmarks) > 15:
                wrist = frame_pose.pose_landmarks[15]
                offset_x = wrist.x * w - left_hand[:, 0].mean()
                offset_y = wrist.y * h - left_hand[:, 1].mean()
                left_hand[:, 0] += offset_x
                left_hand[:, 1] += offset_y
            self._draw_connections(frame, left_hand, HAND_CONNECTIONS, self.settings.hand_color)
            self._draw_points(frame, left_hand, self.settings.hand_color)

        if frame_pose.right_hand_landmarks:
            right_hand = self._normalize_landmarks(frame_pose.right_hand_landmarks, w, h, center=False)
            if frame_pose.pose_landmarks and len(frame_pose.pose_landmarks) > 16:
                wrist = frame_pose.pose_landmarks[16]
                offset_x = wrist.x * w - right_hand[:, 0].mean()
                offset_y = wrist.y * h - right_hand[:, 1].mean()
                right_hand[:, 0] += offset_x
                right_hand[:, 1] += offset_y
            self._draw_connections(frame, right_hand, HAND_CONNECTIONS, self.settings.hand_color)
            self._draw_points(frame, right_hand, self.settings.hand_color)

        # Draw face mesh (optional, very dense)
        if self.settings.show_face and frame_pose.face_landmarks:
            face_points = self._normalize_landmarks(frame_pose.face_landmarks, w, h)
            # Just draw points, not connections (too many)
            for point in face_points[::3]:  # Every 3rd point
                x, y = point[:2].astype(int)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 1, self.settings.face_color, -1)

        return frame

    def render_sequence(self, pose_sequence: PoseSequence) -> np.ndarray:
        """Render full pose sequence to frame array."""
        frames = []
        for frame_pose in pose_sequence.frames:
            rendered = self.render_frame(frame_pose)
            frames.append(rendered)

        return np.array(frames)

    def render_to_video(
        self,
        pose_sequence: PoseSequence,
        output_path: Path,
        fps: Optional[float] = None,
    ) -> Path:
        """Render pose sequence to video file."""
        fps = fps or pose_sequence.fps or self.settings.fps
        frames = self.render_sequence(pose_sequence)

        return self._export_frames(frames, output_path, fps)

    def render_multiple(
        self,
        pose_sequences: List[PoseSequence],
        output_path: Path,
        transition_frames: int = 5,
    ) -> Path:
        """Render multiple pose sequences concatenated."""
        all_frames = []

        for i, pose_seq in enumerate(pose_sequences):
            frames = self.render_sequence(pose_seq)
            all_frames.append(frames)

            # Add transition (fade or just cut)
            if i < len(pose_sequences) - 1 and transition_frames > 0:
                # Simple: just add blank frames
                blank = np.full(
                    (transition_frames, self.settings.height, self.settings.width, 3),
                    self.settings.background_color,
                    dtype=np.uint8,
                )
                all_frames.append(blank)

        combined = np.concatenate(all_frames, axis=0)
        return self._export_frames(combined, output_path, self.settings.fps)

    def _export_frames(
        self,
        frames: np.ndarray,
        output_path: Path,
        fps: float,
    ) -> Path:
        """Export frames to video using ffmpeg."""
        height, width = frames.shape[1:3]
        output_path = Path(output_path)

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            for frame in frames:
                process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass
        finally:
            if process.stdin:
                process.stdin.close()

        stderr = process.stderr.read()
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

        return output_path


def compose_avatar_sequence(
    glosses: List[str],
    poses_dir: Path,
    output_path: Path,
    settings: Optional[RenderSettings] = None,
) -> Path:
    """
    Compose an avatar animation from a sequence of glosses.

    Args:
        glosses: List of sign glosses
        poses_dir: Directory containing pose JSON files
        output_path: Output video path
        settings: Render settings

    Returns:
        Path to rendered video
    """
    renderer = AvatarRenderer(settings)
    sequences = []

    for gloss in glosses:
        pose_path = poses_dir / f"{gloss}.json"
        if not pose_path.exists():
            # Try in sign directory
            pose_path = poses_dir / gloss / "poses.json"

        if pose_path.exists():
            seq = PoseSequence.load(pose_path)
            sequences.append(seq)
        else:
            print(f"Warning: No pose data for {gloss}")

    if not sequences:
        raise ValueError("No pose sequences found")

    return renderer.render_multiple(sequences, output_path)
