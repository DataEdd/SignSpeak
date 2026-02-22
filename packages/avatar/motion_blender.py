"""
Natural Motion Blending for Sign Language Avatar

Implements human-like motion transitions between signs using:
1. SLERP (Spherical Linear Interpolation) for rotations
2. Easing functions for natural acceleration curves
3. Co-articulation for fluid sign-to-sign transitions
4. Overlapping action for kinematic wave effect
5. Momentum preservation for physics-based motion

Based on lessons from 13Hacks/SignBridge implementation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

try:
    from scipy.spatial.transform import Rotation, Slerp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class EasingType(Enum):
    """Easing function types for natural motion."""
    LINEAR = "linear"
    SINE = "sine"          # Smooth, natural feel
    CUBIC = "cubic"        # Moderate acceleration (Hermite smoothstep)
    QUINTIC = "quintic"    # Pronounced acceleration (Perlin smootherstep)
    ELASTIC = "elastic"    # Overshoot with bounce


@dataclass
class BlendSettings:
    """Settings for motion blending."""
    transition_frames: int = 12          # Frames for sign-to-sign transition
    easing_type: EasingType = EasingType.SINE
    use_slerp: bool = True               # Spherical interpolation for rotations
    use_overlapping: bool = True         # Joint cascade wave
    use_momentum: bool = True            # Velocity preservation
    momentum_carry: float = 0.7          # How much velocity carries through
    overshoot_percent: float = 0.025     # 2.5% overshoot for emphasis
    settling_frames: int = 4             # Frames to settle after overshoot

    # Joint cascade offsets (frames delay from root)
    joint_offsets: dict = None

    def __post_init__(self):
        if self.joint_offsets is None:
            # Proximal to distal cascade
            self.joint_offsets = {
                'root': 0,
                'spine': 1,
                'shoulders': 2,
                'elbows': 4,
                'wrists': 6,
                'fingers': 8,
            }


# =============================================================================
# EASING FUNCTIONS
# =============================================================================

def ease_linear(t: float) -> float:
    """Linear interpolation (robotic, baseline)."""
    return t


def ease_sine(t: float) -> float:
    """Sinusoidal ease-in-out (smooth, natural)."""
    return (1 - np.cos(np.pi * t)) / 2


def ease_cubic(t: float) -> float:
    """Cubic Hermite smoothstep."""
    return 3 * t**2 - 2 * t**3


def ease_quintic(t: float) -> float:
    """Quintic smootherstep (Perlin's improved version)."""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def ease_elastic(t: float, overshoot: float = 0.1) -> float:
    """Elastic ease with overshoot and settle."""
    if t < 0.8:
        # Main motion with overshoot
        progress = t / 0.8
        base = ease_sine(progress)
        return base * (1 + overshoot)
    else:
        # Settling phase
        settle_t = (t - 0.8) / 0.2
        return 1.0 + overshoot * (1 - settle_t) * np.cos(settle_t * np.pi)


def get_easing_function(easing_type: EasingType):
    """Get the easing function for a given type."""
    return {
        EasingType.LINEAR: ease_linear,
        EasingType.SINE: ease_sine,
        EasingType.CUBIC: ease_cubic,
        EasingType.QUINTIC: ease_quintic,
        EasingType.ELASTIC: ease_elastic,
    }[easing_type]


# =============================================================================
# SLERP INTERPOLATION
# =============================================================================

def axis_angle_to_rotation(axis_angle: np.ndarray) -> 'Rotation':
    """Convert axis-angle to scipy Rotation."""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for SLERP")
    return Rotation.from_rotvec(axis_angle)


def rotation_to_axis_angle(rotation: 'Rotation') -> np.ndarray:
    """Convert scipy Rotation to axis-angle."""
    return rotation.as_rotvec()


def slerp_axis_angle(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two axis-angle rotations.

    Args:
        start: Starting rotation (3D axis-angle)
        end: Ending rotation (3D axis-angle)
        t: Interpolation factor [0, 1]

    Returns:
        Interpolated rotation (3D axis-angle)
    """
    if not SCIPY_AVAILABLE:
        # Fallback to linear interpolation
        return start + t * (end - start)

    try:
        r_start = Rotation.from_rotvec(start)
        r_end = Rotation.from_rotvec(end)

        # Create SLERP interpolator
        times = [0, 1]
        rotations = Rotation.concatenate([r_start, r_end])
        slerp = Slerp(times, rotations)

        # Interpolate
        result = slerp(t)
        return result.as_rotvec()
    except Exception:
        # Fallback to linear
        return start + t * (end - start)


def slerp_pose(start_pose: np.ndarray, end_pose: np.ndarray, t: float) -> np.ndarray:
    """
    SLERP interpolation for full body pose (multiple joints).

    Assumes pose is array of axis-angle rotations: [joint1_xyz, joint2_xyz, ...]
    """
    result = np.zeros_like(start_pose)
    n_joints = len(start_pose) // 3

    for i in range(n_joints):
        idx = i * 3
        start_rot = start_pose[idx:idx+3]
        end_rot = end_pose[idx:idx+3]
        result[idx:idx+3] = slerp_axis_angle(start_rot, end_rot, t)

    return result


# =============================================================================
# MOTION BLENDING
# =============================================================================

def blend_frames(
    start_frame: dict,
    end_frame: dict,
    num_frames: int,
    settings: BlendSettings = None
) -> List[dict]:
    """
    Generate smooth transition frames between two poses.

    Args:
        start_frame: Starting SMPL-X parameters dict
        end_frame: Ending SMPL-X parameters dict
        num_frames: Number of transition frames to generate
        settings: Blending settings

    Returns:
        List of interpolated frame dicts
    """
    settings = settings or BlendSettings()
    ease_fn = get_easing_function(settings.easing_type)

    frames = []
    for i in range(num_frames):
        # Base interpolation factor
        t = (i + 1) / (num_frames + 1)

        # Apply easing
        t_eased = ease_fn(t)

        # Create interpolated frame
        frame = {}

        # Interpolate each pose component
        for key in ['root_pose', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose']:
            if key in start_frame and key in end_frame:
                start_val = np.array(start_frame[key])
                end_val = np.array(end_frame[key])

                if settings.use_slerp and SCIPY_AVAILABLE:
                    frame[key] = slerp_pose(start_val, end_val, t_eased)
                else:
                    frame[key] = start_val + t_eased * (end_val - start_val)

        # Linear interpolation for shape/expression
        for key in ['betas', 'expression', 'transl']:
            if key in start_frame and key in end_frame:
                start_val = np.array(start_frame[key])
                end_val = np.array(end_frame[key])
                frame[key] = start_val + t_eased * (end_val - start_val)

        frames.append(frame)

    return frames


def apply_overlapping_action(
    frames: List[dict],
    settings: BlendSettings = None
) -> List[dict]:
    """
    Apply overlapping action - joints move in sequence (proximal to distal).

    This creates a natural "wave" effect where shoulders move first,
    then elbows, then wrists, then fingers.
    """
    if not settings or not settings.use_overlapping:
        return frames

    if len(frames) < 3:
        return frames

    # SMPL-X body pose joint indices (approximate mapping)
    # Body pose: 21 joints × 3 = 63 params
    joint_groups = {
        'spine': list(range(0, 9)),        # Joints 0-2 (spine)
        'shoulders': list(range(36, 42)),   # Joints 12-13 (shoulders)
        'elbows': list(range(42, 48)),      # Joints 14-15 (elbows)
        'wrists': list(range(48, 54)),      # Joints 16-17 (wrists)
    }

    result = []
    for frame_idx, frame in enumerate(frames):
        new_frame = frame.copy()

        if 'body_pose' in frame:
            body_pose = np.array(frame['body_pose']).copy()

            # Apply offset for each joint group
            for group_name, joint_indices in joint_groups.items():
                offset = settings.joint_offsets.get(group_name, 0)

                if offset > 0 and frame_idx >= offset:
                    # Use the pose from earlier frame for this joint group
                    source_idx = frame_idx - offset
                    if source_idx >= 0 and source_idx < len(frames):
                        source_pose = np.array(frames[source_idx]['body_pose'])
                        for idx in joint_indices:
                            if idx < len(body_pose):
                                body_pose[idx] = source_pose[idx]

            new_frame['body_pose'] = body_pose

        # Apply to hands as well
        finger_offset = settings.joint_offsets.get('fingers', 0)
        if finger_offset > 0 and frame_idx >= finger_offset:
            source_idx = frame_idx - finger_offset
            if source_idx >= 0 and source_idx < len(frames):
                for hand_key in ['left_hand_pose', 'right_hand_pose']:
                    if hand_key in frame and hand_key in frames[source_idx]:
                        new_frame[hand_key] = np.array(frames[source_idx][hand_key])

        result.append(new_frame)

    return result


# =============================================================================
# SEQUENCE BLENDING
# =============================================================================

def blend_sign_sequence(
    sign_frames: List[List],
    settings: BlendSettings = None
) -> List:
    """
    Blend multiple signs into a continuous sequence with natural transitions.

    Args:
        sign_frames: List of sign frame sequences [[sign1_frames], [sign2_frames], ...]
        settings: Blending settings

    Returns:
        Single blended sequence of frames (as dicts)
    """
    settings = settings or BlendSettings()

    if not sign_frames:
        return []

    if len(sign_frames) == 1:
        # Convert all to dicts for consistency
        return [frame_to_dict(f) for f in sign_frames[0]]

    result = []

    for i, sign in enumerate(sign_frames):
        if not sign:
            continue

        # Add sign frames (convert to dicts)
        for frame in sign:
            result.append(frame_to_dict(frame))

        # Add transition to next sign (if not last)
        if i < len(sign_frames) - 1:
            next_sign = sign_frames[i + 1]
            if next_sign:
                # Get end of current sign and start of next
                end_frame = frame_to_dict(sign[-1])
                start_frame = frame_to_dict(next_sign[0])

                # Generate transition frames
                transition = blend_frames(
                    end_frame,
                    start_frame,
                    settings.transition_frames,
                    settings
                )

                # Apply overlapping action to transition
                if settings.use_overlapping:
                    transition = apply_overlapping_action(transition, settings)

                # Add transition frames (already dicts)
                result.extend(transition)

    return result


def frame_to_dict(frame) -> dict:
    """Convert SMPLXFrame to dictionary."""
    if isinstance(frame, dict):
        return frame
    return {
        'root_pose': np.array(frame.root_pose),
        'body_pose': np.array(frame.body_pose),
        'left_hand_pose': np.array(frame.left_hand_pose),
        'right_hand_pose': np.array(frame.right_hand_pose),
        'jaw_pose': np.array(frame.jaw_pose),
        'betas': np.array(frame.betas),
        'expression': np.array(frame.expression),
        'transl': np.array(frame.transl),
    }


def dict_to_frame(d):
    """Convert dictionary to SMPLXFrame-like object."""
    from .renderer_smplx import SMPLXFrame

    # If already an SMPLXFrame, return as-is
    if isinstance(d, SMPLXFrame):
        return d

    return SMPLXFrame(
        root_pose=d.get('root_pose', np.zeros(3)),
        body_pose=d.get('body_pose', np.zeros(63)),
        left_hand_pose=d.get('left_hand_pose', np.zeros(45)),
        right_hand_pose=d.get('right_hand_pose', np.zeros(45)),
        jaw_pose=d.get('jaw_pose', np.zeros(3)),
        betas=d.get('betas', np.zeros(10)),
        expression=d.get('expression', np.zeros(10)),
        transl=d.get('transl', np.zeros(3)),
    )


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

class MotionBlender:
    """High-level motion blending interface."""

    def __init__(self, settings: BlendSettings = None):
        self.settings = settings or BlendSettings()

    def blend_sequences(
        self,
        sequences: List['SMPLXSequence']
    ) -> 'SMPLXSequence':
        """
        Blend multiple SMPLXSequences into one with natural transitions.
        """
        from .renderer_smplx import SMPLXSequence

        if not sequences:
            return None

        if len(sequences) == 1:
            return sequences[0]

        # Extract frames from each sequence
        all_sign_frames = [
            [frame_to_dict(f) for f in seq.frames]
            for seq in sequences
        ]

        # Blend with transitions
        blended_dicts = blend_sign_sequence(all_sign_frames, self.settings)

        # Convert back to SMPLXFrames
        blended_frames = [dict_to_frame(d) for d in blended_dicts]

        # Create new sequence
        glosses = [seq.gloss for seq in sequences]
        return SMPLXSequence(
            gloss=" ".join(glosses),
            frames=blended_frames,
            fps=sequences[0].fps
        )
