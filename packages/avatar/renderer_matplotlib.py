"""3D Avatar Renderer using matplotlib - works without OpenGL issues."""

import io
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .pose_extractor import PoseSequence, FramePose


# MediaPipe body connections
BODY_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 23), (12, 24),  # shoulders to hips
    (23, 24),  # hips
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (23, 25), (25, 27), (27, 29),  # left leg
    (24, 26), (26, 28), (28, 30),  # right leg
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


@dataclass
class MatplotlibRenderSettings:
    """Settings for matplotlib 3D rendering."""
    width: int = 720
    height: int = 540
    dpi: int = 100
    body_color: str = '#4a9eff'  # Light blue
    hand_color: str = '#ffaa33'  # Orange
    joint_size: float = 40
    line_width: float = 3
    hand_joint_size: float = 20
    hand_line_width: float = 2
    background_color: str = '#1a1a2e'
    elev: float = 15  # Camera elevation
    azim: float = 0  # Camera azimuth


class AvatarMatplotlibRenderer:
    """Render pose sequences as 3D avatar videos using matplotlib."""

    def __init__(self, settings: Optional[MatplotlibRenderSettings] = None):
        self.settings = settings or MatplotlibRenderSettings()

    def _landmarks_to_3d(self, landmarks: List, scale: float = 1.0) -> np.ndarray:
        """Convert MediaPipe landmarks to 3D coordinates."""
        if landmarks is None:
            return np.array([])

        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        # Center and scale - make coordinates suitable for 3D plot
        points[:, 0] = (points[:, 0] - 0.5) * scale  # X centered
        points[:, 1] = -(points[:, 1] - 0.5) * scale  # Y flipped
        points[:, 2] = points[:, 2] * scale * 0.3  # Z scaled down

        return points

    def render_frame(self, frame_pose: FramePose) -> np.ndarray:
        """Render a single frame to an image array."""
        fig = plt.figure(
            figsize=(self.settings.width / self.settings.dpi,
                     self.settings.height / self.settings.dpi),
            dpi=self.settings.dpi,
            facecolor=self.settings.background_color
        )
        ax = fig.add_subplot(111, projection='3d', facecolor=self.settings.background_color)

        # Configure axes
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.3, 0.3])
        ax.view_init(elev=self.settings.elev, azim=self.settings.azim)

        # Hide axes
        ax.set_axis_off()
        ax.grid(False)

        # Draw body
        if frame_pose.pose_landmarks:
            body_points = self._landmarks_to_3d(frame_pose.pose_landmarks)

            # Draw joints (skip face points 0-10)
            body_joints = body_points[11:]
            ax.scatter(
                body_joints[:, 0],
                body_joints[:, 2],  # Swap Y and Z for better view
                body_joints[:, 1],
                c=self.settings.body_color,
                s=self.settings.joint_size,
                depthshade=True
            )

            # Draw bones
            lines = []
            for start_idx, end_idx in BODY_CONNECTIONS:
                if start_idx < len(body_points) and end_idx < len(body_points):
                    p1 = body_points[start_idx]
                    p2 = body_points[end_idx]
                    lines.append([
                        [p1[0], p1[2], p1[1]],
                        [p2[0], p2[2], p2[1]]
                    ])

            if lines:
                lc = Line3DCollection(
                    lines,
                    colors=self.settings.body_color,
                    linewidths=self.settings.line_width
                )
                ax.add_collection3d(lc)

            # Draw hands
            for hand_landmarks, wrist_idx in [
                (frame_pose.left_hand_landmarks, 15),
                (frame_pose.right_hand_landmarks, 16)
            ]:
                if hand_landmarks:
                    hand_points = self._landmarks_to_3d(hand_landmarks, scale=0.25)

                    # Offset to wrist position
                    if len(body_points) > wrist_idx:
                        wrist = body_points[wrist_idx]
                        hand_center = hand_points[0]
                        offset = wrist - hand_center
                        hand_points = hand_points + offset

                    # Draw hand joints
                    ax.scatter(
                        hand_points[:, 0],
                        hand_points[:, 2],
                        hand_points[:, 1],
                        c=self.settings.hand_color,
                        s=self.settings.hand_joint_size,
                        depthshade=True
                    )

                    # Draw hand bones
                    hand_lines = []
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        if start_idx < len(hand_points) and end_idx < len(hand_points):
                            p1 = hand_points[start_idx]
                            p2 = hand_points[end_idx]
                            hand_lines.append([
                                [p1[0], p1[2], p1[1]],
                                [p2[0], p2[2], p2[1]]
                            ])

                    if hand_lines:
                        hand_lc = Line3DCollection(
                            hand_lines,
                            colors=self.settings.hand_color,
                            linewidths=self.settings.hand_line_width
                        )
                        ax.add_collection3d(hand_lc)

        # Convert to image array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=self.settings.background_color,
                    bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        from PIL import Image
        img = Image.open(buf)
        img = img.convert('RGB')
        img = img.resize((self.settings.width, self.settings.height), Image.Resampling.LANCZOS)
        frame = np.array(img)

        plt.close(fig)
        buf.close()

        return frame

    def render_sequence(self, pose_sequence: PoseSequence) -> List[np.ndarray]:
        """Render a full pose sequence."""
        frames = []
        total = len(pose_sequence.frames)
        for i, frame_pose in enumerate(pose_sequence.frames):
            if i % 10 == 0:
                print(f"  Rendering frame {i+1}/{total}...", end='\r')
            rendered = self.render_frame(frame_pose)
            frames.append(rendered)
        print(f"  Rendered {total} frames        ")
        return frames

    def render_to_video(
        self,
        pose_sequence: PoseSequence,
        output_path: Path,
        fps: Optional[float] = None
    ) -> Path:
        """Render pose sequence to video."""
        fps = fps or pose_sequence.fps or 30.0
        print(f"Rendering {pose_sequence.gloss}...")
        frames = self.render_sequence(pose_sequence)
        return self._export_frames(np.array(frames), output_path, fps)

    def render_multiple(
        self,
        pose_sequences: List[PoseSequence],
        output_path: Path,
        transition_frames: int = 5
    ) -> Path:
        """Render multiple sequences concatenated."""
        all_frames = []

        for i, seq in enumerate(pose_sequences):
            print(f"Rendering {seq.gloss} ({i+1}/{len(pose_sequences)})...")
            frames = self.render_sequence(seq)
            all_frames.extend(frames)

            if i < len(pose_sequences) - 1 and transition_frames > 0:
                # Add blank transition frames
                bg_color = tuple(int(self.settings.background_color.lstrip('#')[j:j+2], 16)
                                for j in (0, 2, 4))
                bg = np.full(
                    (self.settings.height, self.settings.width, 3),
                    bg_color,
                    dtype=np.uint8
                )
                for _ in range(transition_frames):
                    all_frames.append(bg)

        fps = pose_sequences[0].fps if pose_sequences else 30.0
        return self._export_frames(np.array(all_frames), output_path, fps)

    def _export_frames(self, frames: np.ndarray, output_path: Path, fps: float) -> Path:
        """Export frames to video using ffmpeg."""
        height, width = frames.shape[1:3]
        output_path = Path(output_path)

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-crf", "20",
            "-preset", "fast",
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

        process.stderr.read()
        process.wait()

        return output_path
