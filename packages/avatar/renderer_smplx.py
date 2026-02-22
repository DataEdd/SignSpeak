"""SMPL-X Avatar Renderer for high-quality 3D sign language visualization.

This renderer uses the SMPL-X body model to generate realistic human meshes.
It can work with:
1. SignAvatars SMPL-X parameters (once dataset is obtained)
2. Converted MediaPipe poses (approximate)
"""

import io
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

try:
    import torch
    import smplx
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .pose_extractor import PoseSequence, FramePose


@dataclass
class SMPLXRenderSettings:
    """Settings for SMPL-X avatar rendering."""
    width: int = 720
    height: int = 1280  # Phone portrait aspect ratio (9:16)
    dpi: int = 100
    mesh_color: str = '#7eb8da'
    mesh_alpha: float = 0.9
    edge_color: str = '#3d6a8c'
    edge_alpha: float = 0.3
    background_color: str = '#1a1a2e'
    elev: float = 10
    azim: float = -90  # Front view (looking at XZ plane from -Y)
    model_path: str = 'data/models'
    gender: str = 'neutral'


@dataclass
class SMPLXFrame:
    """SMPL-X parameters for a single frame."""
    root_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_pose: np.ndarray = field(default_factory=lambda: np.zeros(63))
    left_hand_pose: np.ndarray = field(default_factory=lambda: np.zeros(45))
    right_hand_pose: np.ndarray = field(default_factory=lambda: np.zeros(45))
    jaw_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    betas: np.ndarray = field(default_factory=lambda: np.zeros(10))
    expression: np.ndarray = field(default_factory=lambda: np.zeros(10))
    transl: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class SMPLXSequence:
    """A sequence of SMPL-X frames for a sign."""
    gloss: str
    frames: List[SMPLXFrame]
    fps: float = 30.0

    @classmethod
    def load_signavatars(cls, pkl_path: Path, gloss: str = None, neutral_shape: bool = True) -> 'SMPLXSequence':
        """Load from SignAvatars .pkl format.

        Args:
            pkl_path: Path to .pkl file
            gloss: Optional gloss name
            neutral_shape: If True, use neutral body shape (betas=0) for consistent avatar appearance.
                          How2Sign data has custom body shapes that can look different.
        """
        import pickle
        import io

        # Custom unpickler to handle CUDA tensors on CPU
        class CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
                return super().find_class(module, name)

        with open(pkl_path, 'rb') as f:
            data = CPUUnpickler(f).load()

        # SignAvatars format: 182-dim vector per frame
        params = data.get('smplx', data.get('unsmooth_smplx'))
        if params is None:
            raise ValueError(f"No SMPL-X parameters found in {pkl_path}")

        # Convert torch tensor to numpy if needed
        if hasattr(params, 'numpy'):
            params = params.numpy()

        frames = []
        for i in range(len(params)):
            p = params[i]
            if hasattr(p, 'numpy'):
                p = p.numpy()

            # Load raw root_pose - correction happens during rendering (like 13Hacks)
            root_pose = np.array(p[:3])

            # Use neutral shape for consistent appearance, or original shape
            if neutral_shape:
                betas = np.zeros(10)
            else:
                betas = np.array(p[159:169]) if len(p) > 159 else np.zeros(10)

            frame = SMPLXFrame(
                root_pose=root_pose,
                body_pose=np.array(p[3:66]),
                left_hand_pose=np.array(p[66:111]),
                right_hand_pose=np.array(p[111:156]),
                jaw_pose=np.array(p[156:159]) if len(p) > 156 else np.zeros(3),
                betas=betas,
                expression=np.array(p[169:179]) if len(p) > 169 else np.zeros(10),
                transl=np.array(p[179:182]) if len(p) > 179 else np.zeros(3),
            )
            frames.append(frame)

        return cls(gloss=gloss or pkl_path.stem, frames=frames)


class MediaPipeToSMPLX:
    """Convert MediaPipe poses to approximate SMPL-X parameters."""

    # MediaPipe to SMPL-X body joint mapping (approximate)
    # MediaPipe has 33 pose landmarks, SMPL-X has 22 body joints
    BODY_MAPPING = {
        # MediaPipe idx -> SMPL-X joint name
        11: 'left_shoulder',
        12: 'right_shoulder',
        13: 'left_elbow',
        14: 'right_elbow',
        15: 'left_wrist',
        16: 'right_wrist',
        23: 'left_hip',
        24: 'right_hip',
        25: 'left_knee',
        26: 'right_knee',
        27: 'left_ankle',
        28: 'right_ankle',
    }

    def __init__(self):
        """Initialize converter."""
        pass

    def convert_frame(self, frame_pose: FramePose) -> SMPLXFrame:
        """Convert a MediaPipe frame to SMPL-X parameters.

        Note: This is an approximate conversion that captures the general
        body pose. Full accuracy requires optimization-based fitting.
        """
        smplx_frame = SMPLXFrame()

        if frame_pose.pose_landmarks:
            # Extract key joint positions
            landmarks = frame_pose.pose_landmarks

            # Compute approximate body pose from joint angles
            body_pose = self._compute_body_pose(landmarks)
            smplx_frame.body_pose = body_pose

            # Compute root orientation
            smplx_frame.root_pose = self._compute_root_pose(landmarks)

        # Convert hand poses
        if frame_pose.left_hand_landmarks:
            smplx_frame.left_hand_pose = self._compute_hand_pose(
                frame_pose.left_hand_landmarks
            )

        if frame_pose.right_hand_landmarks:
            smplx_frame.right_hand_pose = self._compute_hand_pose(
                frame_pose.right_hand_landmarks
            )

        return smplx_frame

    def _compute_body_pose(self, landmarks) -> np.ndarray:
        """Compute SMPL-X body pose from MediaPipe landmarks."""
        # SMPL-X body has 21 joints (excluding root) × 3 axis-angle = 63 params
        body_pose = np.zeros(63)

        # Get key positions
        def get_pos(idx):
            if idx < len(landmarks):
                lm = landmarks[idx]
                return np.array([lm.x - 0.5, -(lm.y - 0.5), -lm.z])
            return np.zeros(3)

        # Compute approximate joint angles from positions
        # This is a simplified approach - full fitting would use optimization

        # Left arm: compute elbow and shoulder angles
        left_shoulder = get_pos(11)
        left_elbow = get_pos(13)
        left_wrist = get_pos(15)

        if np.linalg.norm(left_elbow - left_shoulder) > 0.01:
            upper_arm = left_elbow - left_shoulder
            forearm = left_wrist - left_elbow

            # Shoulder rotation (joints 16, 17, 18 in SMPL-X correspond to left arm)
            shoulder_angle = self._direction_to_axis_angle(upper_arm)
            body_pose[48:51] = shoulder_angle  # Left shoulder

            # Elbow angle
            if np.linalg.norm(forearm) > 0.01:
                elbow_angle = self._compute_elbow_angle(upper_arm, forearm)
                body_pose[51:54] = elbow_angle  # Left elbow

        # Right arm
        right_shoulder = get_pos(12)
        right_elbow = get_pos(14)
        right_wrist = get_pos(16)

        if np.linalg.norm(right_elbow - right_shoulder) > 0.01:
            upper_arm = right_elbow - right_shoulder
            forearm = right_wrist - right_elbow

            shoulder_angle = self._direction_to_axis_angle(upper_arm)
            body_pose[45:48] = shoulder_angle  # Right shoulder

            if np.linalg.norm(forearm) > 0.01:
                elbow_angle = self._compute_elbow_angle(upper_arm, forearm)
                body_pose[54:57] = elbow_angle  # Right elbow

        return body_pose

    def _compute_root_pose(self, landmarks) -> np.ndarray:
        """Compute root orientation from landmarks."""
        root_pose = np.zeros(3)

        # Get hip and shoulder positions to estimate torso orientation
        def get_pos(idx):
            if idx < len(landmarks):
                lm = landmarks[idx]
                return np.array([lm.x - 0.5, -(lm.y - 0.5), -lm.z])
            return np.zeros(3)

        left_hip = get_pos(23)
        right_hip = get_pos(24)
        left_shoulder = get_pos(11)
        right_shoulder = get_pos(12)

        # Hip direction
        hip_dir = right_hip - left_hip
        if np.linalg.norm(hip_dir) > 0.01:
            hip_dir = hip_dir / np.linalg.norm(hip_dir)
            # Y rotation based on hip direction
            root_pose[1] = np.arctan2(hip_dir[2], hip_dir[0])

        return root_pose

    def _compute_hand_pose(self, landmarks) -> np.ndarray:
        """Compute hand pose from hand landmarks."""
        # SMPL-X uses PCA for hand poses, 45 dims for full pose
        hand_pose = np.zeros(45)

        # Simple approach: estimate finger curling based on fingertip distances
        # Full accuracy would require proper IK or learning-based methods

        def get_pos(idx):
            if idx < len(landmarks):
                lm = landmarks[idx]
                return np.array([lm.x, lm.y, lm.z])
            return np.zeros(3)

        # Get wrist and fingertips
        wrist = get_pos(0)

        # For each finger, estimate curl based on tip-to-wrist distance
        finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        finger_bases = [1, 5, 9, 13, 17]

        for i, (tip_idx, base_idx) in enumerate(zip(finger_tips, finger_bases)):
            tip = get_pos(tip_idx)
            base = get_pos(base_idx)

            # Distance from tip to base vs expected extended length
            actual_dist = np.linalg.norm(tip - base)
            # Estimate curl amount
            curl = 1.0 - min(actual_dist / 0.15, 1.0)

            # Set approximate joint angles for this finger
            # Each finger has ~3 joints
            start_idx = i * 9
            if start_idx + 9 <= 45:
                hand_pose[start_idx:start_idx + 3] = curl * 0.5  # Base
                hand_pose[start_idx + 3:start_idx + 6] = curl * 0.8  # Mid
                hand_pose[start_idx + 6:start_idx + 9] = curl * 0.5  # Tip

        return hand_pose

    def _direction_to_axis_angle(self, direction: np.ndarray) -> np.ndarray:
        """Convert a direction vector to axis-angle rotation."""
        if np.linalg.norm(direction) < 1e-6:
            return np.zeros(3)

        direction = direction / np.linalg.norm(direction)
        default = np.array([0, -1, 0])  # Default arm direction (down)

        # Compute rotation from default to target direction
        axis = np.cross(default, direction)
        if np.linalg.norm(axis) < 1e-6:
            return np.zeros(3)

        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(default, direction), -1, 1))

        return axis * angle

    def _compute_elbow_angle(
        self,
        upper_arm: np.ndarray,
        forearm: np.ndarray
    ) -> np.ndarray:
        """Compute elbow bend angle."""
        if np.linalg.norm(upper_arm) < 1e-6 or np.linalg.norm(forearm) < 1e-6:
            return np.zeros(3)

        upper_arm_norm = upper_arm / np.linalg.norm(upper_arm)
        forearm_norm = forearm / np.linalg.norm(forearm)

        # Elbow bend angle
        cos_angle = np.clip(np.dot(upper_arm_norm, forearm_norm), -1, 1)
        angle = np.arccos(cos_angle)

        # Elbow bends primarily around local X axis
        return np.array([angle, 0, 0])

    def convert_sequence(self, pose_sequence: PoseSequence) -> SMPLXSequence:
        """Convert a full MediaPipe sequence to SMPL-X."""
        frames = [self.convert_frame(fp) for fp in pose_sequence.frames]
        return SMPLXSequence(
            gloss=pose_sequence.gloss,
            frames=frames,
            fps=pose_sequence.fps or 30.0
        )


class AvatarSMPLXRenderer:
    """Render SMPL-X avatar sequences to video."""

    def __init__(self, settings: Optional[SMPLXRenderSettings] = None):
        if not SMPLX_AVAILABLE:
            raise ImportError(
                "smplx and torch required. Run: pip install smplx torch"
            )

        self.settings = settings or SMPLXRenderSettings()
        self._load_model()
        self.converter = MediaPipeToSMPLX()

    def _load_model(self):
        """Load the SMPL-X model."""
        model_path = Path(self.settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"SMPL-X model path not found: {model_path}\n"
                "Download from: https://smpl-x.is.tue.mpg.de/"
            )

        self.model = smplx.create(
            model_path=str(model_path),
            model_type='smplx',
            gender=self.settings.gender,
            use_pca=False,  # Use full hand pose
            flat_hand_mean=True,
        )
        self.faces = self.model.faces

    def _smplx_frame_to_vertices(self, frame: SMPLXFrame) -> np.ndarray:
        """Convert SMPL-X parameters to mesh vertices."""
        with torch.no_grad():
            output = self.model(
                global_orient=torch.tensor(frame.root_pose).float().unsqueeze(0),
                body_pose=torch.tensor(frame.body_pose).float().unsqueeze(0),
                left_hand_pose=torch.tensor(frame.left_hand_pose).float().unsqueeze(0),
                right_hand_pose=torch.tensor(frame.right_hand_pose).float().unsqueeze(0),
                jaw_pose=torch.tensor(frame.jaw_pose).float().unsqueeze(0),
                betas=torch.tensor(frame.betas).float().unsqueeze(0),
                expression=torch.tensor(frame.expression).float().unsqueeze(0),
                transl=torch.tensor(frame.transl).float().unsqueeze(0),
            )

        vertices = output.vertices.detach().numpy()[0]
        return vertices

    def render_smplx_frame(self, frame: SMPLXFrame) -> np.ndarray:
        """Render a single SMPL-X frame."""
        vertices = self._smplx_frame_to_vertices(frame)
        return self._render_mesh(vertices)

    def render_frame(self, frame_pose: FramePose) -> np.ndarray:
        """Render a MediaPipe frame by converting to SMPL-X."""
        smplx_frame = self.converter.convert_frame(frame_pose)
        return self.render_smplx_frame(smplx_frame)

    def _render_mesh(self, vertices: np.ndarray) -> np.ndarray:
        """Render mesh vertices to image using matplotlib."""
        # Transform coordinates for proper viewing:
        # SMPL-X: X=right, Y=up, Z=forward (towards camera)
        # Matplotlib 3D: X=right, Y=depth, Z=up
        vertices_transformed = np.zeros_like(vertices)
        vertices_transformed[:, 0] = vertices[:, 0]   # X stays (left-right)
        vertices_transformed[:, 1] = -vertices[:, 2]  # Y = -Z (depth, flip for front)
        vertices_transformed[:, 2] = vertices[:, 1]   # Z = Y (up)
        vertices = vertices_transformed

        # Center using bounding box midpoint (not mean) for proper centering
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        center = (mins + maxs) / 2
        vertices = vertices - center

        fig = plt.figure(
            figsize=(self.settings.width / self.settings.dpi,
                     self.settings.height / self.settings.dpi),
            dpi=self.settings.dpi,
            facecolor=self.settings.background_color
        )
        ax = fig.add_subplot(111, projection='3d',
                            facecolor=self.settings.background_color)

        # Use all faces for solid mesh appearance
        triangles = vertices[self.faces]

        # Calculate simple shading based on face normals (manual lighting)
        # Compute face normals
        v0 = triangles[:, 0]
        v1 = triangles[:, 1]
        v2 = triangles[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        norm_lengths[norm_lengths == 0] = 1
        normals = normals / norm_lengths

        # Light direction (from camera)
        light_dir = np.array([0, -1, 0.5])
        light_dir = light_dir / np.linalg.norm(light_dir)

        # Calculate lighting intensity
        intensity = np.dot(normals, light_dir)
        intensity = np.clip(intensity, 0.2, 1.0)  # Ambient + diffuse

        # Base color
        hex_color = self.settings.mesh_color.lstrip('#')
        base_rgb = np.array([int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4)])

        # Apply shading to each face
        face_colors = np.outer(intensity, base_rgb)
        face_colors = np.clip(face_colors, 0, 1)

        mesh = Poly3DCollection(
            triangles,
            facecolors=face_colors,
            edgecolors='none',
            linewidths=0,
        )
        ax.add_collection3d(mesh)

        # Set tight axis limits - fit body closely
        # Use different extents for each axis to frame body tightly
        x_ext = (maxs[0] - mins[0]) / 2 * 1.05
        y_ext = (maxs[1] - mins[1]) / 2 * 1.05
        z_ext = (maxs[2] - mins[2]) / 2 * 1.05

        # Make all axes equal for proper aspect ratio, use max extent
        max_ext = max(x_ext, z_ext) * 1.0  # Prioritize width and height
        ax.set_xlim([-max_ext, max_ext])
        ax.set_ylim([-y_ext * 2, y_ext * 2])  # More depth range
        ax.set_zlim([-z_ext, z_ext])

        ax.view_init(elev=self.settings.elev, azim=self.settings.azim)
        ax.set_axis_off()
        ax.grid(False)

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png',
                    facecolor=self.settings.background_color,
                    bbox_inches='tight', pad_inches=0)
        buf.seek(0)

        from PIL import Image
        img = Image.open(buf)
        img = img.convert('RGB')

        # Use fixed crop region if provided, otherwise no crop
        img_array = np.array(img)

        if hasattr(self, '_fixed_crop') and self._fixed_crop is not None:
            cmin, rmin, cmax, rmax = self._fixed_crop
            img = img.crop((cmin, rmin, cmax, rmax))

        img = img.resize((self.settings.width, self.settings.height),
                        Image.Resampling.LANCZOS)
        frame_array = np.array(img)

        plt.close(fig)
        buf.close()

        return frame_array

    def render_sequence(self, pose_sequence: PoseSequence) -> List[np.ndarray]:
        """Render a MediaPipe pose sequence."""
        frames = []
        total = len(pose_sequence.frames)
        for i, frame_pose in enumerate(pose_sequence.frames):
            if i % 5 == 0:
                print(f"  SMPL-X frame {i+1}/{total}...", end='\r')
            rendered = self.render_frame(frame_pose)
            frames.append(rendered)
        print(f"  Rendered {total} SMPL-X frames    ")
        return frames

    def render_smplx_sequence(self, smplx_sequence: SMPLXSequence) -> List[np.ndarray]:
        """Render an SMPL-X sequence directly with fixed frame (waist-up view)."""
        total = len(smplx_sequence.frames)

        # First pass: compute upper body bounds across all frames
        print(f"  Computing frame bounds...", end='\r')
        sample_indices = list(range(0, total, max(1, total // 10)))
        all_upper_widths = []
        all_upper_heights = []
        torso_centers = []

        for i in sample_indices:
            vertices = self._smplx_frame_to_vertices(smplx_sequence.frames[i])
            # Apply coordinate transform
            vertices_t = np.zeros_like(vertices)
            vertices_t[:, 0] = vertices[:, 0]
            vertices_t[:, 1] = -vertices[:, 2]
            vertices_t[:, 2] = vertices[:, 1]

            # Find upper body (top 60% by height) - waist up
            z_min, z_max = vertices_t[:, 2].min(), vertices_t[:, 2].max()
            body_height = z_max - z_min
            waist_z = z_min + body_height * 0.4  # Waist is ~40% up from feet
            upper_mask = vertices_t[:, 2] >= waist_z

            upper_verts = vertices_t[upper_mask]
            upper_width = upper_verts[:, 0].max() - upper_verts[:, 0].min()
            upper_height = upper_verts[:, 2].max() - upper_verts[:, 2].min()
            all_upper_widths.append(upper_width)
            all_upper_heights.append(upper_height)

            # Track torso center (stable anchor point - around chest)
            chest_z = z_min + body_height * 0.7
            chest_mask = (vertices_t[:, 2] >= chest_z - 0.1) & (vertices_t[:, 2] <= chest_z + 0.1)
            if chest_mask.sum() > 0:
                torso_centers.append(vertices_t[chest_mask].mean(axis=0))

        # Use average torso center as stable anchor
        self._torso_anchor = np.mean(torso_centers, axis=0) if torso_centers else None

        # Compute extents for upper body framing
        max_width = max(all_upper_widths)
        max_height = max(all_upper_heights)

        x_extent = max(max_width * 0.7, 0.5)  # Width with padding
        z_extent = max(max_height * 0.6, 0.5)  # Height with padding

        self._fixed_extents = (x_extent, z_extent)

        # Render all frames
        frames = []
        for i, frame in enumerate(smplx_sequence.frames):
            if i % 5 == 0:
                print(f"  SMPL-X frame {i+1}/{total}...", end='\r')
            vertices = self._smplx_frame_to_vertices(frame)
            rendered = self._render_upper_body(vertices)
            frames.append(rendered)

        print(f"  Rendered {total} SMPL-X frames    ")
        self._fixed_extents = None
        self._torso_anchor = None
        return frames

    def _render_upper_body(self, vertices: np.ndarray) -> np.ndarray:
        """Render upper body (waist-up) with stable torso anchoring."""
        x_extent, z_extent = self._fixed_extents

        # Transform coordinates
        vertices_transformed = np.zeros_like(vertices)
        vertices_transformed[:, 0] = vertices[:, 0]
        vertices_transformed[:, 1] = -vertices[:, 2]
        vertices_transformed[:, 2] = vertices[:, 1]
        vertices = vertices_transformed

        # Find body bounds and waist position
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        body_height = z_max - z_min
        waist_z = z_min + body_height * 0.4

        # Anchor by torso (chest area) - keeps upper body stable
        chest_z = z_min + body_height * 0.7
        chest_mask = (vertices[:, 2] >= chest_z - 0.1) & (vertices[:, 2] <= chest_z + 0.1)
        if chest_mask.sum() > 0 and self._torso_anchor is not None:
            current_chest = vertices[chest_mask].mean(axis=0)
            # Only anchor X position (left-right), not vertical
            x_offset = current_chest[0] - self._torso_anchor[0]
            vertices[:, 0] -= x_offset

        # Center horizontally and in depth
        x_center = (vertices[:, 0].min() + vertices[:, 0].max()) / 2
        vertices[:, 0] -= x_center
        vertices[:, 1] -= vertices[:, 1].mean()

        # Position so upper body is centered in frame
        upper_mask = vertices[:, 2] >= waist_z
        upper_verts = vertices[upper_mask]
        upper_z_center = (upper_verts[:, 2].min() + upper_verts[:, 2].max()) / 2
        vertices[:, 2] -= upper_z_center

        # Calculate figure size
        fig_height = self.settings.height / self.settings.dpi
        fig_width = self.settings.width / self.settings.dpi

        fig = plt.figure(
            figsize=(fig_width, fig_height),
            dpi=self.settings.dpi,
            facecolor=self.settings.background_color
        )
        ax = fig.add_subplot(111, projection='3d',
                            facecolor=self.settings.background_color)

        # Only render faces that are part of upper body (waist and above)
        # Get face centers and filter by Z position
        face_centers_z = vertices[self.faces].mean(axis=1)[:, 2]
        upper_face_mask = face_centers_z >= (waist_z - upper_z_center - 0.05)
        upper_faces = self.faces[upper_face_mask]

        triangles = vertices[upper_faces]

        # Calculate shading
        v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        norm_lengths[norm_lengths == 0] = 1
        normals = normals / norm_lengths

        light_dir = np.array([0, -1, 0.5])
        light_dir = light_dir / np.linalg.norm(light_dir)
        intensity = np.clip(np.dot(normals, light_dir), 0.2, 1.0)

        hex_color = self.settings.mesh_color.lstrip('#')
        base_rgb = np.array([int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4)])
        face_colors = np.clip(np.outer(intensity, base_rgb), 0, 1)

        mesh = Poly3DCollection(triangles, facecolors=face_colors,
                               edgecolors='none', linewidths=0)
        ax.add_collection3d(mesh)

        # FIXED axis limits for stable framing
        ax.set_xlim([-x_extent, x_extent])
        ax.set_ylim([-x_extent * 2, x_extent * 2])
        ax.set_zlim([-z_extent, z_extent])

        ax.view_init(elev=self.settings.elev, azim=self.settings.azim)
        ax.set_axis_off()
        ax.grid(False)
        ax.set_position([0, 0, 1, 1])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=self.settings.background_color,
                    pad_inches=0)
        buf.seek(0)

        from PIL import Image
        img = Image.open(buf)
        img = img.convert('RGB')
        img = img.resize((self.settings.width, self.settings.height),
                        Image.Resampling.LANCZOS)
        frame_array = np.array(img)

        plt.close(fig)
        buf.close()

        return frame_array

    def render_to_video(
        self,
        pose_sequence: PoseSequence,
        output_path: Path,
        fps: Optional[float] = None
    ) -> Path:
        """Render MediaPipe sequence to SMPL-X video."""
        fps = fps or pose_sequence.fps or 30.0
        print(f"Rendering {pose_sequence.gloss} with SMPL-X...")
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
                bg_color = tuple(
                    int(self.settings.background_color.lstrip('#')[j:j+2], 16)
                    for j in (0, 2, 4)
                )
                bg = np.full(
                    (self.settings.height, self.settings.width, 3),
                    bg_color,
                    dtype=np.uint8
                )
                for _ in range(transition_frames):
                    all_frames.append(bg)

        fps = pose_sequences[0].fps if pose_sequences else 30.0
        return self._export_frames(np.array(all_frames), output_path, fps)

    def _export_frames(
        self,
        frames: np.ndarray,
        output_path: Path,
        fps: float
    ) -> Path:
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


class PyRenderSMPLXRenderer:
    """High-quality SMPL-X renderer using PyRender."""

    def __init__(self, settings: Optional[SMPLXRenderSettings] = None, use_texture: bool = True):
        if not SMPLX_AVAILABLE:
            raise ImportError("smplx and torch required")

        try:
            import pyrender
            import trimesh
            from PIL import Image
            self.pyrender = pyrender
            self.trimesh = trimesh
            self.PIL_Image = Image
        except ImportError:
            raise ImportError("pyrender and trimesh required. Run: pip install pyrender trimesh")

        self.settings = settings or SMPLXRenderSettings()
        self.use_texture = use_texture
        self._load_model()

        # Load texture if available
        self.texture_image = None
        self.uv_coords = None
        if use_texture:
            self._load_texture()

        # Nice warm skin tone
        self.skin_color = [0.85, 0.65, 0.55, 1.0]  # Warm peachy skin

        # Set up offscreen renderer
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.settings.width,
            viewport_height=self.settings.height
        )

    def _load_model(self):
        """Load the SMPL-X model (female for textured rendering)."""
        model_path = Path(self.settings.model_path)
        # Use female model for textured rendering
        gender = 'female' if self.use_texture else self.settings.gender
        self.model = smplx.create(
            model_path=str(model_path),
            model_type='smplx',
            gender=gender,
            use_pca=False,
            flat_hand_mean=True,
        )
        self.faces = self.model.faces

    def _load_texture(self):
        """Load the textured reference mesh and sample vertex colors."""
        texture_dir = Path(self.settings.model_path).parent / 'SMPL_texture_samples' / 'Textured_Mesh_samples'
        uv_obj_path = texture_dir / 'SMPLX-female.obj'
        texture_path = texture_dir / 'f_01_alb.002.png'

        if uv_obj_path.exists() and texture_path.exists():
            print(f"  Loading textured mesh: {uv_obj_path.name}")

            # Load the complete textured mesh - trimesh unrolls it to have 1:1 vertex:UV mapping
            self.textured_mesh = self.trimesh.load(
                str(uv_obj_path),
                process=False,
                force='mesh'
            )

            # The mesh has 42064 vertices (unrolled for UV seams) but SMPL-X has 10475
            # Find the mapping: which SMPL-X vertex each textured mesh vertex corresponds to
            ref_vertices = self.textured_mesh.vertices
            unique_positions, self.vertex_mapping = np.unique(
                ref_vertices.round(decimals=6), axis=0, return_inverse=True
            )

            print(f"  Textured mesh: {len(ref_vertices)} verts -> {len(unique_positions)} unique positions")
            print(f"  SMPL-X model: {self.model.get_num_verts()} verts")

            if len(unique_positions) == self.model.get_num_verts():
                print(f"  Vertex mapping OK!")

                # Load texture image and sample vertex colors from UVs
                texture_img = self.PIL_Image.open(texture_path)
                texture_array = np.array(texture_img)
                tex_h, tex_w = texture_array.shape[:2]

                # Get UV coordinates (one per textured mesh vertex)
                uvs = self.textured_mesh.visual.uv

                # Sample colors for each of the 42064 textured mesh vertices
                textured_vertex_colors = np.zeros((len(uvs), 4), dtype=np.uint8)
                for i, (u, v) in enumerate(uvs):
                    px = int(np.clip(u * (tex_w - 1), 0, tex_w - 1))
                    py = int(np.clip((1.0 - v) * (tex_h - 1), 0, tex_h - 1))
                    textured_vertex_colors[i, :3] = texture_array[py, px, :3]
                    textured_vertex_colors[i, 3] = 255

                # Average colors for vertices that share the same SMPL-X position
                # This gives us colors for the 10475 SMPL-X vertices
                self.smplx_vertex_colors = np.zeros((len(unique_positions), 4), dtype=np.float32)
                counts = np.zeros(len(unique_positions))
                for i, smplx_idx in enumerate(self.vertex_mapping):
                    self.smplx_vertex_colors[smplx_idx] += textured_vertex_colors[i].astype(np.float32)
                    counts[smplx_idx] += 1
                self.smplx_vertex_colors /= counts[:, np.newaxis]
                self.smplx_vertex_colors = self.smplx_vertex_colors.astype(np.uint8)

                print(f"  Sampled vertex colors for {len(self.smplx_vertex_colors)} vertices")
                self.texture_loaded = True
            else:
                print(f"  WARNING: Vertex count mismatch!")
                self.texture_loaded = False
                self.use_texture = False
        else:
            print(f"  Texture files not found")
            self.use_texture = False
            self.texture_loaded = False
            self.textured_mesh = None
            self.vertex_mapping = None
            self.smplx_vertex_colors = None

    def _smplx_frame_to_vertices(self, frame: SMPLXFrame) -> np.ndarray:
        """Convert SMPL-X parameters to mesh vertices with locked position."""
        from scipy.spatial.transform import Rotation

        # Constrain hand poses to prevent unnatural finger bending
        left_hand = self._constrain_hand_pose(frame.left_hand_pose)
        right_hand = self._constrain_hand_pose(frame.right_hand_pose)

        # Fix root orientation - keep avatar upright facing camera
        # Zero out root pose to prevent tilting, only keep Y rotation (turning)
        root_pose = frame.root_pose
        root_magnitude = np.linalg.norm(root_pose)

        if root_magnitude > 0.1:
            # Extract just the Y rotation (left-right turn) and discard X/Z tilt
            original_rot = Rotation.from_rotvec(root_pose)
            euler = original_rot.as_euler('xyz')
            # Keep only Y rotation, zero out X and Z to prevent tilt
            corrected_euler = [0, euler[1], 0]
            corrected_rot = Rotation.from_euler('xyz', corrected_euler)
            corrected_root = corrected_rot.as_rotvec()
        else:
            corrected_root = np.zeros(3)  # Fully upright

        # Use body pose as-is (constraint was causing issues)
        body_pose = frame.body_pose

        # ALWAYS zero out translation
        transl = np.zeros(3)

        with torch.no_grad():
            output = self.model(
                global_orient=torch.tensor(corrected_root).float().unsqueeze(0),
                body_pose=torch.tensor(body_pose).float().unsqueeze(0),
                left_hand_pose=torch.tensor(left_hand).float().unsqueeze(0),
                right_hand_pose=torch.tensor(right_hand).float().unsqueeze(0),
                jaw_pose=torch.tensor(frame.jaw_pose).float().unsqueeze(0),
                betas=torch.tensor(frame.betas).float().unsqueeze(0),
                expression=torch.tensor(frame.expression).float().unsqueeze(0),
                transl=torch.tensor(transl).float().unsqueeze(0),
            )
        return output.vertices.detach().numpy()[0]

    def _keep_hands_forward(self, body_pose: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
        """Ensure minimum forward shoulder position to keep hands visible.

        Uses exact indices from 13Hacks working implementation:
        - L_Shoulder at 48:51 with [X=forward, Y=twist, Z=outward]
        - R_Shoulder at 51:54
        """
        pose = body_pose.copy()

        # From 13Hacks neutral pose that works:
        # L_Shoulder [0.3, 0.0, 0.2] = forward raise, slight outward
        # R_Shoulder [0.3, 0.0, -0.2] = forward raise, slight outward

        # Ensure minimum forward raise (X component >= 0.3)
        if pose[48] < 0.3:  # Left shoulder X
            pose[48] = max(pose[48], 0.3)
        if pose[51] < 0.3:  # Right shoulder X
            pose[51] = max(pose[51], 0.3)

        # Ensure slight outward position
        if pose[50] < 0.1:  # Left shoulder Z (positive = outward for left)
            pose[50] = max(pose[50], 0.1)
        if pose[53] > -0.1:  # Right shoulder Z (negative = outward for right)
            pose[53] = min(pose[53], -0.1)

        return pose

    def _ensure_arms_forward(self, body_pose: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
        """Ensure hands are ALWAYS in front of body by iteratively adjusting shoulders.

        Keeps adjusting until hands are verified to be in front.
        """
        pose = body_pose.copy()

        for iteration in range(8):  # Up to 8 iterations
            # Check current hand positions
            with torch.no_grad():
                output = self.model(
                    global_orient=torch.tensor(root_pose).float().unsqueeze(0),
                    body_pose=torch.tensor(pose).float().unsqueeze(0),
                    left_hand_pose=torch.zeros(1, 45),
                    right_hand_pose=torch.zeros(1, 45),
                    betas=torch.zeros(1, 10),
                    expression=torch.zeros(1, 10),
                )
                joints = output.joints.detach().numpy()[0]

            # Get torso front - use maximum Z of spine/chest area
            spine_z = max(joints[3, 2], joints[6, 2], joints[9, 2])  # spine1, spine2, spine3

            # Wrist and elbow positions
            left_wrist_z = joints[20, 2]
            right_wrist_z = joints[21, 2]
            left_elbow_z = joints[18, 2]
            right_elbow_z = joints[19, 2]

            # Check if any arm part is behind torso
            left_behind = min(left_wrist_z, left_elbow_z) < spine_z
            right_behind = min(right_wrist_z, right_elbow_z) < spine_z

            if not left_behind and not right_behind:
                break  # Both arms are in front, done

            # Push arms forward more aggressively
            if left_behind:
                deficit = spine_z - min(left_wrist_z, left_elbow_z)
                # Increase shoulder forward flexion (X rotation at index 48)
                pose[48] += 0.4 + deficit * 2
                # Also bend elbow more (X rotation at index 54)
                pose[54] += 0.2

            if right_behind:
                deficit = spine_z - min(right_wrist_z, right_elbow_z)
                pose[51] += 0.4 + deficit * 2
                pose[57] += 0.2

        return pose

    def _get_neutral_pose(self, reference_frame: SMPLXFrame) -> SMPLXFrame:
        """Create a neutral-active signing pose - arms relaxed at sides, hands visible.

        Based on 13Hacks implementation:
        - Arms relaxed but slightly raised
        - Hands in front at waist level
        - Fingers relaxed (slight natural curl)

        SMPL-X body_pose joint indices (0-indexed from body_pose start):
        16: L_Shoulder (indices 48:51), 17: R_Shoulder (51:54)
        18: L_Elbow (54:57), 19: R_Elbow (57:60)
        """
        body_pose = np.zeros(63)

        # Shoulders: slight forward raise, slight outward
        # This keeps arms naturally at sides but visible from front
        body_pose[48:51] = [0.3, 0.0, 0.2]   # Left shoulder: forward, outward
        body_pose[51:54] = [0.3, 0.0, -0.2]  # Right shoulder: forward, outward

        # Elbows: bent inward so hands are in front of body
        body_pose[54:57] = [0.0, 0.0, 0.4]   # Left elbow bent inward
        body_pose[57:60] = [0.0, 0.0, -0.4]  # Right elbow bent inward

        # Relaxed hand pose with natural slight curl
        hand_pose = np.zeros(45)
        for finger in range(5):
            base = finger * 9  # 3 joints x 3 dims per finger
            hand_pose[base:base+3] = [0.1, 0.0, 0.0]      # First joint
            hand_pose[base+3:base+6] = [0.05, 0.0, 0.0]   # Second joint
            hand_pose[base+6:base+9] = [0.02, 0.0, 0.0]   # Third joint

        return SMPLXFrame(
            root_pose=reference_frame.root_pose.copy(),  # Match reference orientation!
            body_pose=body_pose,
            left_hand_pose=hand_pose.copy(),
            right_hand_pose=hand_pose.copy(),
            jaw_pose=np.zeros(3),
            betas=reference_frame.betas.copy(),
            expression=np.zeros(10),
            transl=np.zeros(3),
        )

    def _constrain_arm_pose(self, body_pose: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
        """Constrain arm poses to ALWAYS keep hands in front of body.

        Iteratively adjusts shoulder and elbow until hands are in front.
        Sign language interpreters NEVER have hands behind their body.

        Joint indices in body_pose (using 13Hacks convention):
        - L_Shoulder: 48:51, R_Shoulder: 51:54
        - L_Elbow: 54:57, R_Elbow: 57:60
        """
        pose = body_pose.copy()

        # Iterate to ensure constraints are met
        for iteration in range(5):
            # Run forward pass to get joint positions
            with torch.no_grad():
                output = self.model(
                    global_orient=torch.tensor(root_pose).float().unsqueeze(0),
                    body_pose=torch.tensor(pose).float().unsqueeze(0),
                    left_hand_pose=torch.zeros(1, 45),
                    right_hand_pose=torch.zeros(1, 45),
                    betas=torch.zeros(1, 10),
                    expression=torch.zeros(1, 10),
                )
                joints = output.joints.detach().numpy()[0]

            # Get body reference points
            spine3 = joints[9]      # Chest
            pelvis = joints[0]      # Pelvis
            left_shoulder_joint = joints[16]
            right_shoulder_joint = joints[17]

            # The "front plane" - hands must be in front of this
            body_front_z = max(spine3[2], pelvis[2], left_shoulder_joint[2], right_shoulder_joint[2])

            # Get arm positions
            left_wrist = joints[20]
            right_wrist = joints[21]
            left_elbow = joints[18]
            right_elbow = joints[19]

            # Check violations - how far behind the body are the hands?
            left_wrist_behind = body_front_z - left_wrist[2]
            right_wrist_behind = body_front_z - right_wrist[2]
            left_elbow_behind = body_front_z - left_elbow[2]
            right_elbow_behind = body_front_z - right_elbow[2]

            # If everything is in front, we're done
            max_violation = max(left_wrist_behind, right_wrist_behind,
                               left_elbow_behind, right_elbow_behind)
            if max_violation <= 0.02:  # Small tolerance
                break

            # Fix left arm if behind (indices 48:51 for shoulder, 54:57 for elbow)
            if left_wrist_behind > 0.02 or left_elbow_behind > 0.02:
                violation = max(left_wrist_behind, left_elbow_behind)
                # Shoulder: [X=forward, Y=twist, Z=abduction]
                pose[48] += violation * 2.0 + 0.2  # More forward raise
                pose[50] += violation * 0.5 + 0.1  # More outward (positive Z for left)
                # Elbow: bend inward
                pose[56] += violation * 0.5 + 0.1  # Elbow Z rotation

            # Fix right arm if behind (indices 51:54 for shoulder, 57:60 for elbow)
            if right_wrist_behind > 0.02 or right_elbow_behind > 0.02:
                violation = max(right_wrist_behind, right_elbow_behind)
                pose[51] += violation * 2.0 + 0.2  # More forward raise
                pose[53] -= violation * 0.5 + 0.1  # More outward (negative Z for right)
                pose[59] -= violation * 0.5 + 0.1  # Elbow Z rotation

        # Final check: prevent hands from crossing through each other
        with torch.no_grad():
            output = self.model(
                global_orient=torch.tensor(root_pose).float().unsqueeze(0),
                body_pose=torch.tensor(pose).float().unsqueeze(0),
                left_hand_pose=torch.zeros(1, 45),
                right_hand_pose=torch.zeros(1, 45),
                betas=torch.zeros(1, 10),
                expression=torch.zeros(1, 10),
            )
            joints = output.joints.detach().numpy()[0]

        left_wrist = joints[20]
        right_wrist = joints[21]
        hand_distance = np.linalg.norm(left_wrist - right_wrist)

        if hand_distance < 0.08:  # Hands too close/intersecting
            pose[50] += 0.2  # Push left arm outward (Z)
            pose[53] -= 0.2  # Push right arm outward (Z)

        return pose

    def _constrain_hand_pose(self, hand_pose: np.ndarray) -> np.ndarray:
        """Constrain hand pose to prevent unnatural finger positions.

        Fingers should NEVER bend backwards (hyperextension).
        This is anatomically impossible and looks very wrong.
        """
        hand = hand_pose.copy()

        # SMPL-X hand pose: 15 joints x 3 axis-angle = 45 params
        # Joint order: wrist, then 3 joints per finger (thumb, index, middle, ring, pinky)
        # Each joint has 3 axis-angle values [x, y, z]

        for i in range(15):  # 15 finger joints
            joint = hand[i*3:(i+1)*3]

            # STRICT constraint: NO backwards bending (hyperextension)
            # Negative X rotation = backwards bend = NOT ALLOWED
            joint[0] = max(joint[0], -0.1)  # Almost no backwards bend allowed

            # Cap forward curl to reasonable amount
            joint[0] = min(joint[0], 2.0)

            # Strictly limit side-to-side and twist (these should be minimal)
            joint[1] = np.clip(joint[1], -0.3, 0.3)
            joint[2] = np.clip(joint[2], -0.3, 0.3)

            # Cap overall rotation magnitude
            magnitude = np.linalg.norm(joint)
            if magnitude > 1.2:
                joint = joint * (1.2 / magnitude)

            hand[i*3:(i+1)*3] = joint

        return hand

    def render_smplx_sequence(self, smplx_sequence: SMPLXSequence) -> List[np.ndarray]:
        """Render SMPL-X sequence with PyRender - simplified, no modifications."""
        all_frames = list(smplx_sequence.frames)
        total = len(all_frames)

        print(f"  Computing fixed frame reference...", end='\r')

        # Use first frame for camera reference
        ref_vertices = self._smplx_frame_to_vertices(all_frames[0])
        y_min, y_max = ref_vertices[:, 1].min(), ref_vertices[:, 1].max()
        body_height = y_max - y_min
        self._waist_y = y_min + body_height * 0.4

        upper_mask = ref_vertices[:, 1] >= self._waist_y
        upper_verts = ref_vertices[upper_mask]

        self._fixed_x_center = (ref_vertices[:, 0].min() + ref_vertices[:, 0].max()) / 2
        self._fixed_z_center = (ref_vertices[:, 2].min() + ref_vertices[:, 2].max()) / 2
        self._fixed_y_center = (upper_verts[:, 1].min() + upper_verts[:, 1].max()) / 2

        x_range = upper_verts[:, 0].max() - upper_verts[:, 0].min()
        y_range = upper_verts[:, 1].max() - upper_verts[:, 1].min()
        self._camera_distance = max(x_range, y_range) * 2.5

        rendered_frames = []
        for i, frame in enumerate(all_frames):
            if i % 5 == 0:
                print(f"  Rendering frame {i+1}/{total}...", end='\r')
            rendered = self._render_frame_pyrender(frame)
            rendered_frames.append(rendered)

        print(f"  Rendered {total} frames              ")
        return rendered_frames

    def _blend_frames(self, start_frame: SMPLXFrame, end_frame: SMPLXFrame, num_frames: int) -> List[SMPLXFrame]:
        """Blend smoothly between two SMPL-X frames using ease-in-out interpolation."""
        frames = []

        for i in range(num_frames):
            # Ease-in-out interpolation (sine curve)
            t = (i + 1) / (num_frames + 1)
            t_smooth = (1 - np.cos(np.pi * t)) / 2  # Smooth ease-in-out

            frame = SMPLXFrame(
                root_pose=self._lerp(start_frame.root_pose, end_frame.root_pose, t_smooth),
                body_pose=self._lerp(start_frame.body_pose, end_frame.body_pose, t_smooth),
                left_hand_pose=self._lerp(start_frame.left_hand_pose, end_frame.left_hand_pose, t_smooth),
                right_hand_pose=self._lerp(start_frame.right_hand_pose, end_frame.right_hand_pose, t_smooth),
                jaw_pose=self._lerp(start_frame.jaw_pose, end_frame.jaw_pose, t_smooth),
                betas=self._lerp(start_frame.betas, end_frame.betas, t_smooth),
                expression=self._lerp(start_frame.expression, end_frame.expression, t_smooth),
                transl=self._lerp(start_frame.transl, end_frame.transl, t_smooth),
            )
            frames.append(frame)

        return frames

    def _lerp(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation between two arrays."""
        return np.array(a) * (1 - t) + np.array(b) * t

    def _render_frame_pyrender(self, frame: SMPLXFrame) -> np.ndarray:
        """Render a single frame using PyRender with FIXED camera position."""
        # Get posed SMPL-X vertices (10,475 vertices)
        smplx_vertices = self._smplx_frame_to_vertices(frame)

        # Use FIXED center values (same for ALL frames = no camera movement)
        smplx_vertices[:, 0] -= self._fixed_x_center
        smplx_vertices[:, 2] -= self._fixed_z_center
        smplx_vertices[:, 1] -= self._fixed_y_center

        waist_y_adjusted = self._waist_y - self._fixed_y_center

        # Filter to upper body faces
        face_centers_y = smplx_vertices[self.faces].mean(axis=1)[:, 1]
        upper_face_mask = face_centers_y >= (waist_y_adjusted - 0.02)
        upper_faces = self.faces[upper_face_mask]

        # Create mesh with clean skin material
        mesh = self.trimesh.Trimesh(
            vertices=smplx_vertices,
            faces=upper_faces,
            process=False
        )
        material = self.pyrender.MetallicRoughnessMaterial(
            baseColorFactor=self.skin_color,
            metallicFactor=0.0,
            roughnessFactor=0.5,
        )
        mesh_pyrender = self.pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

        # Create scene
        scene = self.pyrender.Scene(
            bg_color=[0.12, 0.12, 0.18, 1.0],  # Dark blue-gray background
            ambient_light=[0.35, 0.35, 0.35]
        )
        scene.add(mesh_pyrender)

        # Add lights - 3-point lighting setup
        # Key light (main, from front-right-top)
        key_light = self.pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=2.5)
        key_pose = np.eye(4)
        key_pose[:3, :3] = self._rotation_matrix([-0.4, 0.3, 0])
        scene.add(key_light, pose=key_pose)

        # Fill light (softer, from front-left)
        fill_light = self.pyrender.DirectionalLight(color=[0.95, 0.95, 1.0], intensity=1.2)
        fill_pose = np.eye(4)
        fill_pose[:3, :3] = self._rotation_matrix([-0.2, -0.3, 0])
        scene.add(fill_light, pose=fill_pose)

        # Back/rim light
        rim_light = self.pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        rim_pose = np.eye(4)
        rim_pose[:3, :3] = self._rotation_matrix([0.5, 3.14, 0])
        scene.add(rim_light, pose=rim_pose)

        # Camera - positioned in front looking at mesh
        # Camera looks down -Z, so we position it at +Z and it looks toward origin
        camera = self.pyrender.PerspectiveCamera(yfov=np.pi / 5.5)  # ~33 degree FOV
        camera_pose = np.eye(4)
        camera_pose[2, 3] = self._camera_distance  # Move camera back in Z
        scene.add(camera, pose=camera_pose)

        # Render
        color, _ = self.renderer.render(scene)

        return color

    def _create_hair(self, vertices: np.ndarray) -> list:
        """Create simple hair geometry for feminine appearance."""
        hair_meshes = []

        try:
            # Find the top of the head - vertices with highest Y values
            y_coords = vertices[:, 1]
            head_top_y = y_coords.max()

            # Get vertices near the top of head (scalp area)
            # These are vertices in the top 8% of the head height
            head_height = y_coords.max() - y_coords.min()
            scalp_threshold = head_top_y - head_height * 0.08
            scalp_mask = y_coords >= scalp_threshold
            scalp_verts = vertices[scalp_mask]

            if len(scalp_verts) < 10:
                return hair_meshes

            # Calculate scalp center and bounds
            scalp_center = scalp_verts.mean(axis=0)

            # Hair color - dark brown
            hair_color = [0.15, 0.10, 0.08, 1.0]

            # Create hair as a rounded cap on top of the head
            # Use an ellipsoid that sits on the scalp
            hair_cap = self.trimesh.creation.icosphere(subdivisions=3, radius=0.11)

            # Scale to make it more like a hair cap (wider, flatter)
            hair_cap.vertices[:, 0] *= 1.1  # Wider in X
            hair_cap.vertices[:, 2] *= 1.0  # Normal depth in Z
            hair_cap.vertices[:, 1] *= 0.6  # Flatter in Y (height)

            # Position on top of head
            hair_cap.vertices[:, 0] += scalp_center[0]
            hair_cap.vertices[:, 1] += head_top_y + 0.01  # Slightly above head top
            hair_cap.vertices[:, 2] += scalp_center[2] - 0.02  # Slightly back

            # Remove bottom half of the sphere (keep only the cap)
            # Keep only vertices above a certain Y threshold
            cap_center_y = head_top_y + 0.01

            hair_material = self.pyrender.MetallicRoughnessMaterial(
                baseColorFactor=hair_color,
                metallicFactor=0.0,
                roughnessFactor=0.8,  # Matte hair
            )

            hair_mesh = self.pyrender.Mesh.from_trimesh(hair_cap, material=hair_material, smooth=True)
            hair_meshes.append(hair_mesh)

            # Add side hair pieces (covers ears/sides)
            for side in [-1, 1]:  # Left and right
                side_hair = self.trimesh.creation.cylinder(radius=0.04, height=0.15)

                # Rotate to hang down
                rotation = self.trimesh.transformations.rotation_matrix(
                    np.pi / 2, [1, 0, 0]  # Rotate to vertical
                )
                side_hair.apply_transform(rotation)

                # Position at sides of head
                side_hair.vertices[:, 0] += scalp_center[0] + side * 0.08
                side_hair.vertices[:, 1] += head_top_y - 0.08  # Hang down from top
                side_hair.vertices[:, 2] += scalp_center[2] - 0.02

                side_mesh = self.pyrender.Mesh.from_trimesh(side_hair, material=hair_material, smooth=True)
                hair_meshes.append(side_mesh)

            # Add back hair (longer, hangs down back)
            back_hair = self.trimesh.creation.box(extents=[0.14, 0.20, 0.04])
            back_hair.vertices[:, 0] += scalp_center[0]
            back_hair.vertices[:, 1] += head_top_y - 0.12
            back_hair.vertices[:, 2] += scalp_center[2] - 0.08  # Behind head

            back_mesh = self.pyrender.Mesh.from_trimesh(back_hair, material=hair_material, smooth=True)
            hair_meshes.append(back_mesh)

        except Exception as e:
            pass  # Skip hair if there's an error

        return hair_meshes

    def _sample_texture_colors(self, vertices: np.ndarray) -> np.ndarray:
        """Sample colors from texture based on UV coordinates."""
        n_verts = len(vertices)
        colors = np.ones((n_verts, 4), dtype=np.uint8) * 200  # Default gray

        if self.uv_coords is None or self.texture_image is None:
            return colors

        tex_h, tex_w = self.texture_image.shape[:2]
        n_uvs = len(self.uv_coords)

        # Sample texture at each UV coordinate
        for i in range(min(n_uvs, n_verts)):
            u, v = self.uv_coords[i]
            # UV coords are 0-1, convert to pixel coords
            # V is typically flipped in texture space
            px = int(np.clip(u * (tex_w - 1), 0, tex_w - 1))
            py = int(np.clip((1.0 - v) * (tex_h - 1), 0, tex_h - 1))

            pixel = self.texture_image[py, px]
            colors[i, :3] = pixel[:3] if len(pixel) >= 3 else pixel
            colors[i, 3] = 255

        return colors

    def _rotation_matrix(self, angles):
        """Create rotation matrix from euler angles [x, y, z]."""
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

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
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
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
