"""
Exporter - Output videos in various formats (MP4, WebM, GIF).
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class ExportFormat(Enum):
    """Supported export formats."""
    MP4 = "mp4"
    WEBM = "webm"
    GIF = "gif"


@dataclass
class ExportSettings:
    """Settings for video export."""
    format: ExportFormat = ExportFormat.MP4
    resolution: Optional[Tuple[int, int]] = None  # (width, height), None = keep original
    fps: float = 30.0
    quality: int = 23  # CRF for H.264/VP9 (lower = better, 0-51)
    use_hwaccel: bool = False  # Use hardware acceleration
    hwaccel_type: str = "auto"  # "cuda", "videotoolbox", "vaapi", "auto"

    # GIF-specific settings
    gif_fps: float = 15.0
    gif_colors: int = 256
    gif_dither: bool = True


class VideoExporter:
    """Export video frames to various formats using ffmpeg."""

    def __init__(self, settings: Optional[ExportSettings] = None):
        self.settings = settings or ExportSettings()
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify ffmpeg is available."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    def export(
        self,
        frames: np.ndarray,
        output_path: str,
        fps: Optional[float] = None,
        settings: Optional[ExportSettings] = None
    ) -> Path:
        """
        Export frames to a video file.

        Args:
            frames: Video frames array (num_frames, height, width, channels)
            output_path: Output file path
            fps: Frames per second (overrides settings)
            settings: Export settings (overrides instance settings)

        Returns:
            Path to the exported file
        """
        settings = settings or self.settings
        fps = fps or settings.fps
        output_path = Path(output_path)

        # Determine format from extension if not specified
        ext = output_path.suffix.lower().lstrip(".")
        if ext in ["mp4", "m4v"]:
            export_format = ExportFormat.MP4
        elif ext in ["webm"]:
            export_format = ExportFormat.WEBM
        elif ext in ["gif"]:
            export_format = ExportFormat.GIF
        else:
            export_format = settings.format
            output_path = output_path.with_suffix(f".{export_format.value}")

        # Resize if needed
        if settings.resolution:
            frames = self._resize_frames(frames, settings.resolution)

        if export_format == ExportFormat.GIF:
            return self._export_gif(frames, output_path, settings)
        else:
            return self._export_video(frames, output_path, export_format, fps, settings)

    def _resize_frames(
        self,
        frames: np.ndarray,
        resolution: Tuple[int, int]
    ) -> np.ndarray:
        """Resize frames to target resolution."""
        width, height = resolution
        return np.array([
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            for frame in frames
        ])

    def _export_video(
        self,
        frames: np.ndarray,
        output_path: Path,
        export_format: ExportFormat,
        fps: float,
        settings: ExportSettings
    ) -> Path:
        """Export to MP4 or WebM using ffmpeg pipe."""
        height, width = frames.shape[1:3]

        # Build ffmpeg command
        cmd = ["ffmpeg", "-y"]  # -y overwrites output

        # Hardware acceleration
        if settings.use_hwaccel:
            hwaccel = self._get_hwaccel_args(settings.hwaccel_type)
            cmd.extend(hwaccel)

        # Input settings
        cmd.extend([
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-"  # Read from pipe
        ])

        # Output settings based on format
        if export_format == ExportFormat.MP4:
            encoder = self._get_h264_encoder(settings)
            cmd.extend([
                "-c:v", encoder,
                "-crf", str(settings.quality),
                "-preset", "medium",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart"
            ])
        elif export_format == ExportFormat.WEBM:
            cmd.extend([
                "-c:v", "libvpx-vp9",
                "-crf", str(settings.quality),
                "-b:v", "0",
                "-pix_fmt", "yuv420p"
            ])

        cmd.append(str(output_path))

        # Run ffmpeg
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Write frames to pipe
        try:
            for frame in frames:
                process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass  # ffmpeg may close early on error
        finally:
            if process.stdin:
                process.stdin.close()

        # Wait for process and capture stderr
        stderr = process.stderr.read()
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

        return output_path

    def _export_gif(
        self,
        frames: np.ndarray,
        output_path: Path,
        settings: ExportSettings
    ) -> Path:
        """Export to GIF using ffmpeg with palette generation."""
        height, width = frames.shape[1:3]

        # Create temporary file for raw video
        with tempfile.NamedTemporaryFile(suffix=".rgb", delete=False) as tmp:
            tmp_path = tmp.name
            for frame in frames:
                tmp.write(frame.tobytes())

        try:
            # Generate palette
            palette_path = tempfile.mktemp(suffix=".png")

            palette_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", str(settings.gif_fps),
                "-i", tmp_path,
                "-vf", f"fps={settings.gif_fps},palettegen=max_colors={settings.gif_colors}",
                palette_path
            ]

            subprocess.run(palette_cmd, capture_output=True, check=True)

            # Create GIF with palette
            dither = "dither=bayer:bayer_scale=5" if settings.gif_dither else "dither=none"

            gif_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", str(settings.gif_fps),
                "-i", tmp_path,
                "-i", palette_path,
                "-lavfi", f"fps={settings.gif_fps} [x]; [x][1:v] paletteuse={dither}",
                str(output_path)
            ]

            result = subprocess.run(gif_cmd, capture_output=True)

            if result.returncode != 0:
                raise RuntimeError(f"GIF export failed: {result.stderr.decode()}")

        finally:
            # Cleanup
            os.unlink(tmp_path)
            if os.path.exists(palette_path):
                os.unlink(palette_path)

        return output_path

    def _get_hwaccel_args(self, hwaccel_type: str) -> list:
        """Get hardware acceleration arguments for ffmpeg."""
        if hwaccel_type == "auto":
            # Try to detect available acceleration
            hwaccel_type = self._detect_hwaccel()

        if hwaccel_type == "cuda":
            return ["-hwaccel", "cuda"]
        elif hwaccel_type == "videotoolbox":
            return ["-hwaccel", "videotoolbox"]
        elif hwaccel_type == "vaapi":
            return ["-hwaccel", "vaapi"]

        return []

    def _detect_hwaccel(self) -> str:
        """Detect available hardware acceleration."""
        # Check for NVIDIA CUDA
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return "cuda"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Check for macOS VideoToolbox
        import platform
        if platform.system() == "Darwin":
            return "videotoolbox"

        # Check for VAAPI (Linux)
        if os.path.exists("/dev/dri"):
            return "vaapi"

        return ""

    def _get_h264_encoder(self, settings: ExportSettings) -> str:
        """Get the best available H.264 encoder."""
        if settings.use_hwaccel:
            hwaccel = settings.hwaccel_type
            if hwaccel == "auto":
                hwaccel = self._detect_hwaccel()

            if hwaccel == "cuda":
                return "h264_nvenc"
            elif hwaccel == "videotoolbox":
                return "h264_videotoolbox"
            elif hwaccel == "vaapi":
                return "h264_vaapi"

        return "libx264"


def export_frames(
    frames: np.ndarray,
    output_path: str,
    fps: float = 30.0,
    format: ExportFormat = ExportFormat.MP4,
    quality: int = 23
) -> Path:
    """
    Convenience function to export frames to video.

    Args:
        frames: Video frames array
        output_path: Output file path
        fps: Frames per second
        format: Export format
        quality: Quality setting (0-51, lower is better)

    Returns:
        Path to exported file
    """
    settings = ExportSettings(format=format, quality=quality)
    exporter = VideoExporter(settings)
    return exporter.export(frames, output_path, fps=fps)
