"""Tests for exporter module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from packages.video.exporter import (
    ExportFormat,
    ExportSettings,
    VideoExporter,
    export_frames,
)


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_mp4_value(self):
        """Test MP4 enum value is correct."""
        assert ExportFormat.MP4.value == "mp4"

    def test_webm_value(self):
        """Test WEBM enum value is correct."""
        assert ExportFormat.WEBM.value == "webm"

    def test_gif_value(self):
        """Test GIF enum value is correct."""
        assert ExportFormat.GIF.value == "gif"


class TestExportSettings:
    """Tests for ExportSettings dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        settings = ExportSettings()

        assert settings.format == ExportFormat.MP4
        assert settings.resolution is None
        assert settings.fps == 30.0
        assert settings.quality == 23
        assert settings.use_hwaccel is False
        assert settings.hwaccel_type == "auto"

    def test_gif_specific_settings(self):
        """Test GIF-specific default settings."""
        settings = ExportSettings()

        assert settings.gif_fps == 15.0
        assert settings.gif_colors == 256
        assert settings.gif_dither is True

    def test_hwaccel_settings(self):
        """Test hardware acceleration settings can be customized."""
        settings = ExportSettings(use_hwaccel=True, hwaccel_type="cuda")

        assert settings.use_hwaccel is True
        assert settings.hwaccel_type == "cuda"

    def test_custom_resolution(self):
        """Test custom resolution setting."""
        settings = ExportSettings(resolution=(1280, 720))

        assert settings.resolution == (1280, 720)


class TestVideoExporterInit:
    """Tests for VideoExporter initialization."""

    def test_init_with_default_settings(self, mock_ffmpeg):
        """Test initialization with default settings."""
        exporter = VideoExporter()

        assert exporter.settings.format == ExportFormat.MP4
        assert exporter.settings.fps == 30.0

    def test_init_with_custom_settings(self, mock_ffmpeg):
        """Test initialization with custom settings."""
        settings = ExportSettings(format=ExportFormat.WEBM, fps=60.0)
        exporter = VideoExporter(settings)

        assert exporter.settings.format == ExportFormat.WEBM
        assert exporter.settings.fps == 60.0


class TestCheckFfmpeg:
    """Tests for ffmpeg availability checking."""

    def test_check_ffmpeg_raises_when_missing(self):
        """Test RuntimeError is raised when ffmpeg is not found."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()

            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                VideoExporter()

    def test_check_ffmpeg_raises_on_failure(self):
        """Test RuntimeError is raised when ffmpeg command fails."""
        import subprocess
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")

            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                VideoExporter()

    def test_check_ffmpeg_passes_when_available(self, mock_ffmpeg):
        """Test no error when ffmpeg is available."""
        exporter = VideoExporter()
        assert exporter is not None


class TestExportFormatInference:
    """Tests for export format inference from file extension."""

    def test_export_infers_mp4_from_mp4_extension(self, mock_ffmpeg, sample_frames):
        """Test MP4 format is inferred from .mp4 extension."""
        mock_run, mock_popen = mock_ffmpeg
        exporter = VideoExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            exporter.export(sample_frames, str(output_path))

            # Verify ffmpeg was called with H.264 encoder
            call_args = mock_popen.call_args[0][0]
            assert "-c:v" in call_args
            assert "libx264" in call_args

    def test_export_infers_mp4_from_m4v_extension(self, mock_ffmpeg, sample_frames):
        """Test MP4 format is inferred from .m4v extension."""
        mock_run, mock_popen = mock_ffmpeg
        exporter = VideoExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.m4v"
            exporter.export(sample_frames, str(output_path))

            call_args = mock_popen.call_args[0][0]
            assert "libx264" in call_args

    def test_export_infers_webm_from_webm_extension(self, mock_ffmpeg, sample_frames):
        """Test WEBM format is inferred from .webm extension."""
        mock_run, mock_popen = mock_ffmpeg
        exporter = VideoExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.webm"
            exporter.export(sample_frames, str(output_path))

            call_args = mock_popen.call_args[0][0]
            assert "libvpx-vp9" in call_args

    def test_export_infers_gif_from_gif_extension(self, mock_ffmpeg, sample_frames):
        """Test GIF format is inferred from .gif extension."""
        mock_run, mock_popen = mock_ffmpeg
        exporter = VideoExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.gif"
            exporter.export(sample_frames, str(output_path))

            # GIF export uses subprocess.run for palette generation
            assert mock_run.call_count >= 2  # Initial check + palette + gif

    def test_export_uses_settings_format_for_unknown_extension(self, mock_ffmpeg, sample_frames):
        """Test settings format is used for unknown extensions."""
        mock_run, mock_popen = mock_ffmpeg
        settings = ExportSettings(format=ExportFormat.WEBM)
        exporter = VideoExporter(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.xyz"
            result = exporter.export(sample_frames, str(output_path))

            # Should add .webm extension
            assert result.suffix == ".webm"

    def test_export_adds_correct_extension_for_unknown(self, mock_ffmpeg, sample_frames):
        """Test correct extension is added for unknown extension."""
        mock_run, mock_popen = mock_ffmpeg
        settings = ExportSettings(format=ExportFormat.MP4)
        exporter = VideoExporter(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.unknown"
            result = exporter.export(sample_frames, str(output_path))

            assert result.suffix == ".mp4"


class TestExportResizing:
    """Tests for frame resizing during export."""

    def test_export_calls_resize_frames_when_resolution_specified(self, mock_ffmpeg, sample_frames):
        """Test _resize_frames is called when resolution is specified."""
        mock_run, mock_popen = mock_ffmpeg
        settings = ExportSettings(resolution=(200, 200))
        exporter = VideoExporter(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            with patch.object(exporter, '_resize_frames', wraps=exporter._resize_frames) as mock_resize:
                exporter.export(sample_frames, str(output_path))
                mock_resize.assert_called_once()


class TestResizeFrames:
    """Tests for _resize_frames method."""

    def test_resize_frames_resizes_all_frames(self, mock_ffmpeg, sample_frames):
        """Test all frames are resized to target resolution."""
        exporter = VideoExporter()
        resized = exporter._resize_frames(sample_frames, (200, 150))

        assert resized.shape == (10, 150, 200, 3)

    def test_resize_frames_preserves_frame_count(self, mock_ffmpeg, sample_frames):
        """Test frame count is preserved after resize."""
        exporter = VideoExporter()
        original_count = len(sample_frames)
        resized = exporter._resize_frames(sample_frames, (50, 50))

        assert len(resized) == original_count


class TestExportVideo:
    """Tests for _export_video method."""

    def test_export_video_builds_correct_ffmpeg_command_for_mp4(self, mock_ffmpeg, sample_frames):
        """Test correct ffmpeg command is built for MP4 export."""
        mock_run, mock_popen = mock_ffmpeg
        exporter = VideoExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            exporter._export_video(
                sample_frames, output_path, ExportFormat.MP4, 30.0, ExportSettings()
            )

            call_args = mock_popen.call_args[0][0]
            assert "ffmpeg" in call_args
            assert "-c:v" in call_args
            assert "libx264" in call_args
            assert "-crf" in call_args
            assert "-pix_fmt" in call_args
            assert "yuv420p" in call_args
            assert "-movflags" in call_args

    def test_export_video_builds_correct_ffmpeg_command_for_webm(self, mock_ffmpeg, sample_frames):
        """Test correct ffmpeg command is built for WEBM export."""
        mock_run, mock_popen = mock_ffmpeg
        exporter = VideoExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.webm"
            exporter._export_video(
                sample_frames, output_path, ExportFormat.WEBM, 30.0, ExportSettings()
            )

            call_args = mock_popen.call_args[0][0]
            assert "libvpx-vp9" in call_args
            assert "-b:v" in call_args
            assert "0" in call_args

    def test_export_video_includes_hwaccel_args_when_enabled(self, mock_ffmpeg, sample_frames):
        """Test hardware acceleration arguments are included when enabled."""
        mock_run, mock_popen = mock_ffmpeg
        settings = ExportSettings(use_hwaccel=True, hwaccel_type="cuda")
        exporter = VideoExporter(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            exporter._export_video(sample_frames, output_path, ExportFormat.MP4, 30.0, settings)

            call_args = mock_popen.call_args[0][0]
            assert "-hwaccel" in call_args
            assert "cuda" in call_args

    def test_export_video_raises_on_ffmpeg_failure(self, mock_ffmpeg, sample_frames):
        """Test RuntimeError is raised when ffmpeg fails."""
        mock_run, mock_popen = mock_ffmpeg

        # Make process return non-zero exit code
        process = mock_popen.return_value
        process.returncode = 1
        process.communicate.return_value = (b'', b'Error: something went wrong')

        exporter = VideoExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            with pytest.raises(RuntimeError, match="ffmpeg failed"):
                exporter._export_video(
                    sample_frames, output_path, ExportFormat.MP4, 30.0, ExportSettings()
                )


class TestExportGif:
    """Tests for _export_gif method."""

    def test_export_gif_creates_palette_and_final_gif(self, mock_ffmpeg, sample_frames):
        """Test GIF export creates palette and final GIF."""
        mock_run, mock_popen = mock_ffmpeg
        exporter = VideoExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.gif"
            exporter._export_gif(sample_frames, output_path, ExportSettings())

            # Should call subprocess.run twice (palette generation + gif creation)
            # Plus the initial ffmpeg check
            assert mock_run.call_count >= 2

    def test_export_gif_cleans_up_temp_files(self, mock_ffmpeg, sample_frames):
        """Test temporary files are cleaned up after GIF export."""
        mock_run, mock_popen = mock_ffmpeg
        exporter = VideoExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.gif"

            # Track temp files
            with patch('tempfile.NamedTemporaryFile') as mock_tmp:
                mock_file = MagicMock()
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_file.name = str(Path(tmpdir) / "temp.rgb")
                mock_file.write = MagicMock()
                mock_tmp.return_value = mock_file

                with patch('os.unlink') as mock_unlink:
                    with patch('os.path.exists', return_value=True):
                        exporter._export_gif(sample_frames, output_path, ExportSettings())

                    # Verify cleanup was called
                    assert mock_unlink.call_count >= 1

    def test_export_gif_respects_gif_dither_setting_enabled(self, mock_ffmpeg, sample_frames):
        """Test GIF export respects dither setting when enabled."""
        mock_run, mock_popen = mock_ffmpeg
        settings = ExportSettings(gif_dither=True)
        exporter = VideoExporter(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.gif"
            exporter._export_gif(sample_frames, output_path, settings)

            # Find the GIF creation call (second call after palette)
            gif_call = mock_run.call_args_list[-1]
            call_args = gif_call[0][0]

            # Look for the lavfi filter with dither setting
            lavfi_idx = call_args.index("-lavfi") if "-lavfi" in call_args else -1
            if lavfi_idx >= 0:
                filter_str = call_args[lavfi_idx + 1]
                assert "dither=bayer" in filter_str

    def test_export_gif_respects_gif_dither_setting_disabled(self, mock_ffmpeg, sample_frames):
        """Test GIF export respects dither setting when disabled."""
        mock_run, mock_popen = mock_ffmpeg
        settings = ExportSettings(gif_dither=False)
        exporter = VideoExporter(settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.gif"
            exporter._export_gif(sample_frames, output_path, settings)

            gif_call = mock_run.call_args_list[-1]
            call_args = gif_call[0][0]

            lavfi_idx = call_args.index("-lavfi") if "-lavfi" in call_args else -1
            if lavfi_idx >= 0:
                filter_str = call_args[lavfi_idx + 1]
                assert "dither=none" in filter_str


class TestGetHwaccelArgs:
    """Tests for _get_hwaccel_args method."""

    def test_get_hwaccel_args_returns_cuda_args(self, mock_ffmpeg):
        """Test CUDA hardware acceleration arguments are returned."""
        exporter = VideoExporter()
        args = exporter._get_hwaccel_args("cuda")

        assert args == ["-hwaccel", "cuda"]

    def test_get_hwaccel_args_returns_videotoolbox_args(self, mock_ffmpeg):
        """Test VideoToolbox hardware acceleration arguments are returned."""
        exporter = VideoExporter()
        args = exporter._get_hwaccel_args("videotoolbox")

        assert args == ["-hwaccel", "videotoolbox"]

    def test_get_hwaccel_args_returns_vaapi_args(self, mock_ffmpeg):
        """Test VAAPI hardware acceleration arguments are returned."""
        exporter = VideoExporter()
        args = exporter._get_hwaccel_args("vaapi")

        assert args == ["-hwaccel", "vaapi"]

    def test_get_hwaccel_args_returns_empty_for_auto_with_nothing_detected(self, mock_ffmpeg):
        """Test empty list is returned when auto detection finds nothing."""
        exporter = VideoExporter()

        with patch.object(exporter, '_detect_hwaccel', return_value=""):
            args = exporter._get_hwaccel_args("auto")

        assert args == []


class TestDetectHwaccel:
    """Tests for _detect_hwaccel method."""

    def test_detect_hwaccel_returns_cuda_when_nvidia_smi_succeeds(self, mock_ffmpeg):
        """Test CUDA is detected when nvidia-smi succeeds."""
        exporter = VideoExporter()

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = exporter._detect_hwaccel()

        assert result == "cuda"

    def test_detect_hwaccel_returns_videotoolbox_on_darwin(self, mock_ffmpeg):
        """Test VideoToolbox is detected on macOS (Darwin)."""
        exporter = VideoExporter()

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()  # nvidia-smi not found

            with patch('platform.system', return_value="Darwin"):
                result = exporter._detect_hwaccel()

        assert result == "videotoolbox"

    def test_detect_hwaccel_returns_vaapi_when_dev_dri_exists(self, mock_ffmpeg):
        """Test VAAPI is detected when /dev/dri exists."""
        exporter = VideoExporter()

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()  # nvidia-smi not found

            with patch('platform.system', return_value="Linux"):
                with patch('os.path.exists', return_value=True):
                    result = exporter._detect_hwaccel()

        assert result == "vaapi"

    def test_detect_hwaccel_returns_empty_when_nothing_available(self, mock_ffmpeg):
        """Test empty string is returned when no hardware acceleration is available."""
        exporter = VideoExporter()

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()  # nvidia-smi not found

            with patch('platform.system', return_value="Linux"):
                with patch('os.path.exists', return_value=False):
                    result = exporter._detect_hwaccel()

        assert result == ""


class TestGetH264Encoder:
    """Tests for _get_h264_encoder method."""

    def test_get_h264_encoder_returns_libx264_without_hwaccel(self, mock_ffmpeg):
        """Test libx264 is returned when hardware acceleration is disabled."""
        exporter = VideoExporter()
        settings = ExportSettings(use_hwaccel=False)

        encoder = exporter._get_h264_encoder(settings)

        assert encoder == "libx264"

    def test_get_h264_encoder_returns_h264_nvenc_for_cuda(self, mock_ffmpeg):
        """Test h264_nvenc is returned for CUDA acceleration."""
        exporter = VideoExporter()
        settings = ExportSettings(use_hwaccel=True, hwaccel_type="cuda")

        encoder = exporter._get_h264_encoder(settings)

        assert encoder == "h264_nvenc"

    def test_get_h264_encoder_returns_h264_videotoolbox_for_videotoolbox(self, mock_ffmpeg):
        """Test h264_videotoolbox is returned for VideoToolbox acceleration."""
        exporter = VideoExporter()
        settings = ExportSettings(use_hwaccel=True, hwaccel_type="videotoolbox")

        encoder = exporter._get_h264_encoder(settings)

        assert encoder == "h264_videotoolbox"

    def test_get_h264_encoder_returns_h264_vaapi_for_vaapi(self, mock_ffmpeg):
        """Test h264_vaapi is returned for VAAPI acceleration."""
        exporter = VideoExporter()
        settings = ExportSettings(use_hwaccel=True, hwaccel_type="vaapi")

        encoder = exporter._get_h264_encoder(settings)

        assert encoder == "h264_vaapi"

    def test_get_h264_encoder_uses_detect_for_auto(self, mock_ffmpeg):
        """Test auto hwaccel type triggers detection."""
        exporter = VideoExporter()
        settings = ExportSettings(use_hwaccel=True, hwaccel_type="auto")

        with patch.object(exporter, '_detect_hwaccel', return_value="cuda"):
            encoder = exporter._get_h264_encoder(settings)

        assert encoder == "h264_nvenc"


class TestExportFramesFunction:
    """Tests for export_frames convenience function."""

    def test_basic_usage(self, mock_ffmpeg, sample_frames):
        """Test basic usage of export_frames function."""
        mock_run, mock_popen = mock_ffmpeg

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"
            result = export_frames(sample_frames, str(output_path))

            assert result == output_path

    def test_passes_options_correctly(self, mock_ffmpeg, sample_frames):
        """Test options are passed correctly to VideoExporter."""
        mock_run, mock_popen = mock_ffmpeg

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.webm"
            result = export_frames(
                sample_frames,
                str(output_path),
                fps=60.0,
                format=ExportFormat.WEBM,
                quality=18
            )

            # Verify the WEBM encoder was used
            call_args = mock_popen.call_args[0][0]
            assert "libvpx-vp9" in call_args

            # Verify quality setting was passed
            crf_idx = call_args.index("-crf")
            assert call_args[crf_idx + 1] == "18"


class TestExportRouting:
    """Tests for export method routing to correct export function."""

    def test_export_routes_to_export_gif_for_gif_format(self, mock_ffmpeg, sample_frames):
        """Test export routes to _export_gif for GIF format."""
        mock_run, mock_popen = mock_ffmpeg
        exporter = VideoExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.gif"

            with patch.object(exporter, '_export_gif', wraps=exporter._export_gif) as mock_gif:
                exporter.export(sample_frames, str(output_path))
                mock_gif.assert_called_once()

    def test_export_routes_to_export_video_for_mp4(self, mock_ffmpeg, sample_frames):
        """Test export routes to _export_video for MP4 format."""
        mock_run, mock_popen = mock_ffmpeg
        exporter = VideoExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            with patch.object(exporter, '_export_video', wraps=exporter._export_video) as mock_video:
                exporter.export(sample_frames, str(output_path))
                mock_video.assert_called_once()

    def test_export_routes_to_export_video_for_webm(self, mock_ffmpeg, sample_frames):
        """Test export routes to _export_video for WEBM format."""
        mock_run, mock_popen = mock_ffmpeg
        exporter = VideoExporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.webm"

            with patch.object(exporter, '_export_video', wraps=exporter._export_video) as mock_video:
                exporter.export(sample_frames, str(output_path))
                mock_video.assert_called_once()
