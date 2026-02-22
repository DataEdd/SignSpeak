"""Extract pose data from sign language videos using MediaPipe Holistic."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


@dataclass
class Landmark:
    """A single 3D landmark point."""
    x: float
    y: float
    z: float
    visibility: float = 1.0

    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y, "z": self.z, "v": self.visibility}

    @classmethod
    def from_dict(cls, d: Dict) -> "Landmark":
        return cls(x=d["x"], y=d["y"], z=d["z"], visibility=d.get("v", 1.0))


@dataclass
class FramePose:
    """Pose data for a single frame."""
    frame_idx: int
    pose_landmarks: Optional[List[Landmark]] = None  # 33 landmarks
    face_landmarks: Optional[List[Landmark]] = None  # 468 landmarks
    left_hand_landmarks: Optional[List[Landmark]] = None  # 21 landmarks
    right_hand_landmarks: Optional[List[Landmark]] = None  # 21 landmarks

    def to_dict(self) -> Dict:
        return {
            "frame": self.frame_idx,
            "pose": [lm.to_dict() for lm in self.pose_landmarks] if self.pose_landmarks else None,
            "face": [lm.to_dict() for lm in self.face_landmarks] if self.face_landmarks else None,
            "left_hand": [lm.to_dict() for lm in self.left_hand_landmarks] if self.left_hand_landmarks else None,
            "right_hand": [lm.to_dict() for lm in self.right_hand_landmarks] if self.right_hand_landmarks else None,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FramePose":
        return cls(
            frame_idx=d["frame"],
            pose_landmarks=[Landmark.from_dict(lm) for lm in d["pose"]] if d.get("pose") else None,
            face_landmarks=[Landmark.from_dict(lm) for lm in d["face"]] if d.get("face") else None,
            left_hand_landmarks=[Landmark.from_dict(lm) for lm in d["left_hand"]] if d.get("left_hand") else None,
            right_hand_landmarks=[Landmark.from_dict(lm) for lm in d["right_hand"]] if d.get("right_hand") else None,
        )

    @property
    def has_body(self) -> bool:
        return self.pose_landmarks is not None

    @property
    def has_hands(self) -> bool:
        return self.left_hand_landmarks is not None or self.right_hand_landmarks is not None


@dataclass
class PoseSequence:
    """A sequence of poses for a sign."""
    gloss: str
    fps: float
    frames: List[FramePose] = field(default_factory=list)
    source: str = "mediapipe"  # "mediapipe" or "signavatars"

    def to_dict(self) -> Dict:
        return {
            "gloss": self.gloss,
            "fps": self.fps,
            "source": self.source,
            "num_frames": len(self.frames),
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PoseSequence":
        return cls(
            gloss=d["gloss"],
            fps=d["fps"],
            source=d.get("source", "mediapipe"),
            frames=[FramePose.from_dict(f) for f in d["frames"]],
        )

    def save(self, path: Path) -> None:
        """Save pose sequence to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: Path) -> "PoseSequence":
        """Load pose sequence from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @property
    def duration_ms(self) -> float:
        return (len(self.frames) / self.fps) * 1000 if self.fps > 0 else 0

    def get_pose_array(self) -> np.ndarray:
        """Get pose landmarks as numpy array (num_frames, 33, 3)."""
        poses = []
        for frame in self.frames:
            if frame.pose_landmarks:
                poses.append([[lm.x, lm.y, lm.z] for lm in frame.pose_landmarks])
            else:
                poses.append([[0, 0, 0]] * 33)
        return np.array(poses)

    def get_hand_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Get hand landmarks as numpy arrays (num_frames, 21, 3) each."""
        left_hands = []
        right_hands = []
        for frame in self.frames:
            if frame.left_hand_landmarks:
                left_hands.append([[lm.x, lm.y, lm.z] for lm in frame.left_hand_landmarks])
            else:
                left_hands.append([[0, 0, 0]] * 21)
            if frame.right_hand_landmarks:
                right_hands.append([[lm.x, lm.y, lm.z] for lm in frame.right_hand_landmarks])
            else:
                right_hands.append([[0, 0, 0]] * 21)
        return np.array(left_hands), np.array(right_hands)


class PoseExtractor:
    """Extract poses from videos using MediaPipe Holistic."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not installed. Run: pip install mediapipe")

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

    def _landmarks_to_list(self, landmarks) -> Optional[List[Landmark]]:
        """Convert MediaPipe landmarks to list of Landmark objects."""
        if landmarks is None:
            return None
        return [
            Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=getattr(lm, "visibility", 1.0),
            )
            for lm in landmarks.landmark
        ]

    def extract_from_video(self, video_path: Path, gloss: str = "") -> PoseSequence:
        """Extract pose sequence from a video file."""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.holistic.process(image_rgb)

            # Extract landmarks
            frame_pose = FramePose(
                frame_idx=frame_idx,
                pose_landmarks=self._landmarks_to_list(results.pose_landmarks),
                face_landmarks=self._landmarks_to_list(results.face_landmarks),
                left_hand_landmarks=self._landmarks_to_list(results.left_hand_landmarks),
                right_hand_landmarks=self._landmarks_to_list(results.right_hand_landmarks),
            )
            frames.append(frame_pose)
            frame_idx += 1

        cap.release()

        return PoseSequence(
            gloss=gloss or video_path.stem,
            fps=fps,
            frames=frames,
            source="mediapipe",
        )

    def extract_from_sign_dir(self, sign_dir: Path) -> PoseSequence:
        """Extract poses from a sign directory (containing video.mp4 and sign.json)."""
        # Find video file
        video_path = None
        for ext in [".mp4", ".webm", ".mov", ".avi"]:
            candidate = sign_dir / f"video{ext}"
            if candidate.exists():
                video_path = candidate
                break

        if video_path is None:
            raise FileNotFoundError(f"No video file found in: {sign_dir}")

        # Get gloss from directory name or sign.json
        gloss = sign_dir.name
        sign_json = sign_dir / "sign.json"
        if sign_json.exists():
            with open(sign_json) as f:
                data = json.load(f)
                gloss = data.get("gloss", gloss)

        return self.extract_from_video(video_path, gloss)

    def close(self):
        """Release MediaPipe resources."""
        self.holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def extract_all_signs(
    signs_dir: Path,
    output_dir: Optional[Path] = None,
    subdirs: List[str] = ["verified"],
    overwrite: bool = False,
) -> Dict[str, Path]:
    """
    Extract poses from all signs in a directory.

    Args:
        signs_dir: Base signs directory
        output_dir: Where to save pose files (default: alongside videos)
        subdirs: Subdirectories to process
        overwrite: Whether to overwrite existing pose files

    Returns:
        Dict mapping gloss to pose file path
    """
    results = {}

    with PoseExtractor() as extractor:
        for subdir in subdirs:
            search_path = signs_dir / subdir
            if not search_path.exists():
                continue

            for sign_dir in sorted(search_path.iterdir()):
                if not sign_dir.is_dir():
                    continue

                gloss = sign_dir.name

                # Determine output path
                if output_dir:
                    pose_path = output_dir / f"{gloss}.json"
                else:
                    pose_path = sign_dir / "poses.json"

                # Skip if exists and not overwriting
                if pose_path.exists() and not overwrite:
                    results[gloss] = pose_path
                    continue

                try:
                    pose_seq = extractor.extract_from_sign_dir(sign_dir)
                    pose_seq.save(pose_path)
                    results[gloss] = pose_path
                    print(f"  {gloss}: {len(pose_seq.frames)} frames")
                except Exception as e:
                    print(f"  {gloss}: ERROR - {e}")

    return results
