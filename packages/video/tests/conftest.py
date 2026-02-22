"""
Shared fixtures for video package tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def sample_frames():
    """Create sample video frames (10 frames, 100x100, RGB)."""
    return np.random.randint(0, 255, (10, 100, 100, 3), dtype=np.uint8)


@pytest.fixture
def small_frames():
    """Create small video frames for fast tests (5 frames, 50x50, RGB)."""
    return np.random.randint(0, 255, (5, 50, 50, 3), dtype=np.uint8)


@pytest.fixture
def single_frame():
    """Create a single frame (100x100, RGB)."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def temp_signs_dir():
    """Create temporary signs directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        (path / "verified").mkdir()
        (path / "pending").mkdir()
        (path / "imported").mkdir()
        (path / "rejected").mkdir()
        yield path


@pytest.fixture
def sample_video_clip(sample_frames):
    """Create a sample VideoClip for testing."""
    from packages.video.clip_manager import VideoClip

    return VideoClip(
        gloss="TEST",
        frames=sample_frames,
        fps=30.0,
        metadata={"source": "test"}
    )


@pytest.fixture
def empty_video_clip():
    """Create an empty VideoClip for testing edge cases."""
    from packages.video.clip_manager import VideoClip

    return VideoClip(
        gloss="EMPTY",
        frames=np.empty((0, 100, 100, 3), dtype=np.uint8),
        fps=30.0,
        metadata={}
    )


@pytest.fixture
def mock_cv2_video_capture(mocker):
    """Mock cv2.VideoCapture for video loading tests."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0  # FPS

    # Return 5 frames then stop
    frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
    read_returns = [(True, frame) for frame in frames] + [(False, None)]
    mock_cap.read.side_effect = read_returns

    mocker.patch('cv2.VideoCapture', return_value=mock_cap)
    return mock_cap


@pytest.fixture
def mock_ffmpeg(mocker):
    """Mock subprocess calls for ffmpeg."""
    mock_run = mocker.patch('subprocess.run')
    mock_run.return_value = MagicMock(returncode=0, stdout=b'', stderr=b'')

    mock_popen = mocker.patch('subprocess.Popen')
    process = MagicMock()
    process.returncode = 0
    process.stdin = MagicMock()
    process.communicate.return_value = (b'', b'')
    mock_popen.return_value = process

    return mock_run, mock_popen


@pytest.fixture
def populated_signs_dir(temp_signs_dir):
    """Create signs directory with sample sign data."""
    import json

    # Create a verified sign
    hello_dir = temp_signs_dir / "verified" / "HELLO"
    hello_dir.mkdir(parents=True)

    metadata = {
        "gloss": "HELLO",
        "english": ["hello", "hi"],
        "category": "greeting",
        "timing": {
            "sign_start_ms": 100,
            "sign_end_ms": 900
        }
    }

    with open(hello_dir / "sign.json", "w") as f:
        json.dump(metadata, f)

    # Create a fake video file (just for path detection)
    (hello_dir / "video.mp4").write_bytes(b"fake video content")

    # Create a pending sign
    world_dir = temp_signs_dir / "pending" / "WORLD"
    world_dir.mkdir(parents=True)
    (world_dir / "sign.json").write_text('{"gloss": "WORLD"}')
    (world_dir / "video.mp4").write_bytes(b"fake video content")

    return temp_signs_dir
