"""3D Avatar Renderer using trimesh and pyrender."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# MediaPipe body connections (indices for 33 pose landmarks)
BODY_CONNECTIONS = [
    # Torso
    (11, 12),  # shoulders
    (11, 23), (12, 24),  # shoulders to hips
    (23, 24),  # hips
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg
    (23, 25), (25, 27), (27, 29), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (30, 32),
]

# Hand connections (indices for 21 hand landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),  # palm
]


@dataclass
class Render3DSettings:
    """Settings for 3D rendering."""
    width: int = 720
    height: int = 540
    body_color: Tuple[float, float, float, float] = (0.3, 0.7, 0.9, 1.0)  # Light blue
    hand_color: Tuple[float, float, float, float] = (0.9, 0.7, 0.3, 1.0)  # Orange
    joint_radius: float = 0.015
    bone_radius: float = 0.008
    hand_joint_radius: float = 0.006
    hand_bone_radius: float = 0.004
    background_color: Tuple[float, float, float, float] = (0.1, 0.1, 0.15, 1.0)
    camera_distance: float = 2.5


# Import pose types
from .pose_extractor import PoseSequence, FramePose


class Avatar3DRenderer:
    """Render pose sequences as 3D avatar videos."""

    def __init__(self, settings: Optional[Render3DSettings] = None):
        # Import dependencies
        try:
            import trimesh
            import pyrender
            from pyrender.constants import RenderFlags
        except ImportError:
            raise ImportError("pyrender and trimesh required. Run: pip install pyrender trimesh")

        self._trimesh = trimesh
        self._pyrender = pyrender
        self._RenderFlags = RenderFlags

        self.settings = settings or Render3DSettings()
        self._setup_renderer()

    def _setup_renderer(self):
        """Initialize the offscreen renderer."""
        self.renderer = self._pyrender.OffscreenRenderer(
            viewport_width=self.settings.width,
            viewport_height=self.settings.height,
        )

    def _create_sphere(self, position: np.ndarray, radius: float, color: Tuple):
        """Create a sphere mesh at given position."""
        sphere = self._trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere.apply_translation(position)
        sphere.visual.vertex_colors = np.array([color] * len(sphere.vertices))
        return sphere

    def _create_cylinder(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        color: Tuple
    ):
        """Create a cylinder between two points."""
        direction = end - start
        length = np.linalg.norm(direction)

        if length < 1e-6:
            return None

        # Create cylinder along Z axis
        cylinder = self._trimesh.creation.cylinder(radius=radius, height=length)

        # Calculate rotation to align with direction
        direction_normalized = direction / length
        z_axis = np.array([0, 0, 1])

        # Rotation axis and angle
        rotation_axis = np.cross(z_axis, direction_normalized)
        rotation_axis_norm = np.linalg.norm(rotation_axis)

        if rotation_axis_norm > 1e-6:
            rotation_axis = rotation_axis / rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1, 1))

            # Rodrigues rotation
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

            transform = np.eye(4)
            transform[:3, :3] = R
            cylinder.apply_transform(transform)

        # Translate to midpoint
        midpoint = (start + end) / 2
        cylinder.apply_translation(midpoint)

        cylinder.visual.vertex_colors = np.array([color] * len(cylinder.vertices))
        return cylinder

    def _landmarks_to_3d(
        self,
        landmarks: List,
        scale: float = 1.0,
        offset: np.ndarray = None
    ) -> np.ndarray:
        """Convert MediaPipe landmarks to 3D coordinates."""
        if landmarks is None:
            return np.array([])

        # MediaPipe gives normalized coords (0-1 for x,y) and relative z
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        # Center and scale
        points[:, 0] = (points[:, 0] - 0.5) * scale
        points[:, 1] = -(points[:, 1] - 0.5) * scale  # Flip Y
        points[:, 2] = -points[:, 2] * scale * 0.5  # Scale Z

        if offset is not None:
            points += offset

        return points

    def _build_body_mesh(self, frame_pose: FramePose):
        """Build a 3D mesh for the body pose."""
        if not frame_pose.pose_landmarks:
            return None

        meshes = []
        points = self._landmarks_to_3d(frame_pose.pose_landmarks)

        color = self.settings.body_color

        # Add joint spheres
        for i, point in enumerate(points):
            # Skip face landmarks (0-10)
            if i < 11:
                continue
            sphere = self._create_sphere(point, self.settings.joint_radius, color)
            meshes.append(sphere)

        # Add bones
        for start_idx, end_idx in BODY_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cylinder = self._create_cylinder(
                    points[start_idx],
                    points[end_idx],
                    self.settings.bone_radius,
                    color
                )
                if cylinder:
                    meshes.append(cylinder)

        if not meshes:
            return None

        return self._trimesh.util.concatenate(meshes)

    def _build_hand_mesh(
        self,
        landmarks: List,
        wrist_position: np.ndarray,
        is_left: bool
    ):
        """Build a 3D mesh for a hand."""
        if landmarks is None:
            return None

        meshes = []

        # Convert hand landmarks
        points = self._landmarks_to_3d(landmarks, scale=0.3)

        # Offset to wrist position
        if wrist_position is not None:
            hand_center = points[0]  # Wrist is index 0
            offset = wrist_position - hand_center
            points += offset

        color = self.settings.hand_color

        # Add joint spheres
        for point in points:
            sphere = self._create_sphere(point, self.settings.hand_joint_radius, color)
            meshes.append(sphere)

        # Add bones
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cylinder = self._create_cylinder(
                    points[start_idx],
                    points[end_idx],
                    self.settings.hand_bone_radius,
                    color
                )
                if cylinder:
                    meshes.append(cylinder)

        if not meshes:
            return None

        return self._trimesh.util.concatenate(meshes)

    def _build_frame_mesh(self, frame_pose: FramePose):
        """Build complete mesh for a frame."""
        meshes = []

        # Body
        body_mesh = self._build_body_mesh(frame_pose)
        if body_mesh:
            meshes.append(body_mesh)

        # Get wrist positions from body pose
        body_points = self._landmarks_to_3d(frame_pose.pose_landmarks) if frame_pose.pose_landmarks else None

        # Left hand
        if frame_pose.left_hand_landmarks:
            left_wrist = body_points[15] if body_points is not None and len(body_points) > 15 else None
            left_hand = self._build_hand_mesh(frame_pose.left_hand_landmarks, left_wrist, is_left=True)
            if left_hand:
                meshes.append(left_hand)

        # Right hand
        if frame_pose.right_hand_landmarks:
            right_wrist = body_points[16] if body_points is not None and len(body_points) > 16 else None
            right_hand = self._build_hand_mesh(frame_pose.right_hand_landmarks, right_wrist, is_left=False)
            if right_hand:
                meshes.append(right_hand)

        if not meshes:
            return None

        return self._trimesh.util.concatenate(meshes)

    def render_frame(self, frame_pose: FramePose) -> np.ndarray:
        """Render a single frame to an image."""
        # Build mesh
        mesh = self._build_frame_mesh(frame_pose)

        if mesh is None:
            # Return empty frame
            return np.full(
                (self.settings.height, self.settings.width, 3),
                (int(self.settings.background_color[0] * 255),
                 int(self.settings.background_color[1] * 255),
                 int(self.settings.background_color[2] * 255)),
                dtype=np.uint8
            )

        # Create scene
        scene = self._pyrender.Scene(
            bg_color=self.settings.background_color,
            ambient_light=[0.3, 0.3, 0.3]
        )

        # Add mesh
        mesh_node = self._pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(mesh_node)

        # Add camera
        camera = self._pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.eye(4)
        camera_pose[2, 3] = self.settings.camera_distance
        scene.add(camera, pose=camera_pose)

        # Add lights
        light = self._pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, :3] = self._trimesh.transformations.euler_matrix(np.pi/4, np.pi/4, 0)[:3, :3]
        scene.add(light, pose=light_pose)

        # Add fill light
        fill_light = self._pyrender.DirectionalLight(color=[0.5, 0.5, 0.6], intensity=1.5)
        fill_pose = np.eye(4)
        fill_pose[:3, :3] = self._trimesh.transformations.euler_matrix(-np.pi/4, -np.pi/4, 0)[:3, :3]
        scene.add(fill_light, pose=fill_pose)

        # Render
        color, _ = self.renderer.render(scene, flags=self._RenderFlags.SHADOWS_DIRECTIONAL)

        return color

    def render_sequence(self, pose_sequence: PoseSequence) -> List[np.ndarray]:
        """Render a full pose sequence."""
        frames = []
        for frame_pose in pose_sequence.frames:
            rendered = self.render_frame(frame_pose)
            frames.append(rendered)
        return frames

    def render_to_video(
        self,
        pose_sequence: PoseSequence,
        output_path: Path,
        fps: Optional[float] = None
    ) -> Path:
        """Render pose sequence to video."""
        fps = fps or pose_sequence.fps or 30.0
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
            frames = self.render_sequence(seq)
            all_frames.extend(frames)

            # Add transition
            if i < len(pose_sequences) - 1 and transition_frames > 0:
                bg = np.full(
                    (self.settings.height, self.settings.width, 3),
                    (int(self.settings.background_color[0] * 255),
                     int(self.settings.background_color[1] * 255),
                     int(self.settings.background_color[2] * 255)),
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

        process.stderr.read()
        process.wait()

        return output_path

    def close(self):
        """Clean up renderer resources."""
        if hasattr(self, 'renderer'):
            self.renderer.delete()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
