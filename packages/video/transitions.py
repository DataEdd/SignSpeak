"""
Transitions - Smooth transitions between sign video clips.
"""

from enum import Enum
from typing import Optional

import cv2
import numpy as np


class TransitionType(Enum):
    """Available transition types."""
    CUT = "cut"
    CROSSFADE = "crossfade"
    BLEND = "blend"  # Frame interpolation


def apply_transition(
    clip_a_frames: np.ndarray,
    clip_b_frames: np.ndarray,
    transition_type: TransitionType,
    duration_frames: int,
    fps: float = 30.0
) -> np.ndarray:
    """
    Apply a transition between two clips.

    Args:
        clip_a_frames: Frames from the outgoing clip
        clip_b_frames: Frames from the incoming clip
        transition_type: Type of transition to apply
        duration_frames: Number of frames for the transition
        fps: Frames per second (for blend interpolation)

    Returns:
        Combined frames with transition applied
    """
    if transition_type == TransitionType.CUT:
        return _apply_cut(clip_a_frames, clip_b_frames)
    elif transition_type == TransitionType.CROSSFADE:
        return _apply_crossfade(clip_a_frames, clip_b_frames, duration_frames)
    elif transition_type == TransitionType.BLEND:
        return _apply_blend(clip_a_frames, clip_b_frames, duration_frames)
    else:
        raise ValueError(f"Unknown transition type: {transition_type}")


def _apply_cut(clip_a_frames: np.ndarray, clip_b_frames: np.ndarray) -> np.ndarray:
    """Hard cut - simply concatenate clips."""
    return np.concatenate([clip_a_frames, clip_b_frames], axis=0)


def _apply_crossfade(
    clip_a_frames: np.ndarray,
    clip_b_frames: np.ndarray,
    duration_frames: int
) -> np.ndarray:
    """
    Crossfade transition - blend opacity between clips.

    The last `duration_frames` of clip_a overlap with the first
    `duration_frames` of clip_b.
    """
    if len(clip_a_frames) == 0:
        return clip_b_frames
    if len(clip_b_frames) == 0:
        return clip_a_frames

    # Ensure we have enough frames
    duration_frames = min(
        duration_frames,
        len(clip_a_frames),
        len(clip_b_frames)
    )

    if duration_frames <= 0:
        return np.concatenate([clip_a_frames, clip_b_frames], axis=0)

    # Split clips
    clip_a_main = clip_a_frames[:-duration_frames]
    clip_a_overlap = clip_a_frames[-duration_frames:]
    clip_b_overlap = clip_b_frames[:duration_frames]
    clip_b_main = clip_b_frames[duration_frames:]

    # Create crossfade
    transition_frames = []
    for i in range(duration_frames):
        alpha = i / (duration_frames - 1) if duration_frames > 1 else 1.0
        blended = cv2.addWeighted(
            clip_a_overlap[i].astype(np.float32),
            1.0 - alpha,
            clip_b_overlap[i].astype(np.float32),
            alpha,
            0
        ).astype(np.uint8)
        transition_frames.append(blended)

    transition_array = np.array(transition_frames)

    # Combine all parts
    parts = []
    if len(clip_a_main) > 0:
        parts.append(clip_a_main)
    parts.append(transition_array)
    if len(clip_b_main) > 0:
        parts.append(clip_b_main)

    return np.concatenate(parts, axis=0)


def _apply_blend(
    clip_a_frames: np.ndarray,
    clip_b_frames: np.ndarray,
    duration_frames: int
) -> np.ndarray:
    """
    Blend transition - frame interpolation for smoother motion.

    Uses optical flow to interpolate between frames.
    """
    if len(clip_a_frames) == 0:
        return clip_b_frames
    if len(clip_b_frames) == 0:
        return clip_a_frames

    duration_frames = min(
        duration_frames,
        len(clip_a_frames),
        len(clip_b_frames)
    )

    if duration_frames <= 0:
        return np.concatenate([clip_a_frames, clip_b_frames], axis=0)

    # Get transition boundary frames
    last_a = clip_a_frames[-1]
    first_b = clip_b_frames[0]

    # Generate interpolated frames
    interpolated = _interpolate_frames(last_a, first_b, duration_frames)

    # Combine: all of clip_a except last, interpolated frames, all of clip_b except first
    parts = []
    if len(clip_a_frames) > 1:
        parts.append(clip_a_frames[:-1])
    parts.append(interpolated)
    if len(clip_b_frames) > 1:
        parts.append(clip_b_frames[1:])

    return np.concatenate(parts, axis=0)


def _interpolate_frames(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    num_frames: int
) -> np.ndarray:
    """
    Interpolate frames between two keyframes using optical flow.

    Falls back to simple linear interpolation if optical flow fails.
    """
    try:
        return _optical_flow_interpolation(frame_a, frame_b, num_frames)
    except Exception:
        return _linear_interpolation(frame_a, frame_b, num_frames)


def _linear_interpolation(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    num_frames: int
) -> np.ndarray:
    """Simple linear interpolation between frames."""
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1) if num_frames > 1 else 1.0
        blended = cv2.addWeighted(
            frame_a.astype(np.float32),
            1.0 - alpha,
            frame_b.astype(np.float32),
            alpha,
            0
        ).astype(np.uint8)
        frames.append(blended)
    return np.array(frames)


def _optical_flow_interpolation(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    num_frames: int
) -> np.ndarray:
    """
    Optical flow-based frame interpolation.

    Uses Farneback dense optical flow for motion estimation.
    """
    # Convert to grayscale for flow calculation
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    h, w = frame_a.shape[:2]

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 1.0

        # Warp frame_a forward
        map_x_forward = x_coords + flow[..., 0] * t
        map_y_forward = y_coords + flow[..., 1] * t
        warped_a = cv2.remap(
            frame_a,
            map_x_forward,
            map_y_forward,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # Warp frame_b backward
        map_x_backward = x_coords - flow[..., 0] * (1 - t)
        map_y_backward = y_coords - flow[..., 1] * (1 - t)
        warped_b = cv2.remap(
            frame_b,
            map_x_backward,
            map_y_backward,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        # Blend warped frames
        blended = cv2.addWeighted(
            warped_a.astype(np.float32),
            1.0 - t,
            warped_b.astype(np.float32),
            t,
            0
        ).astype(np.uint8)

        frames.append(blended)

    return np.array(frames)


def ms_to_frames(duration_ms: float, fps: float) -> int:
    """Convert milliseconds to frame count."""
    return max(1, int((duration_ms / 1000) * fps))


def frames_to_ms(num_frames: int, fps: float) -> float:
    """Convert frame count to milliseconds."""
    return (num_frames / fps) * 1000
