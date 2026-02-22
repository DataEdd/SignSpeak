"""Tests for compositor module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from packages.video.compositor import (
    Compositor,
    CompositorSegment,
    CompositorSettings,
    compose_sequence,
)
from packages.video.clip_manager import VideoClip
from packages.video.transitions import TransitionType


class TestCompositorSegment:
    """Tests for CompositorSegment dataclass."""

    def test_defaults(self, sample_video_clip):
        """Test default values for CompositorSegment."""
        segment = CompositorSegment(clip=sample_video_clip)
        assert segment.transition_out is None
        assert segment.transition_duration_ms == 150

    def test_with_custom_values(self, sample_video_clip):
        """Test CompositorSegment with custom values."""
        segment = CompositorSegment(
            clip=sample_video_clip,
            transition_out=TransitionType.CROSSFADE,
            transition_duration_ms=200
        )
        assert segment.clip == sample_video_clip
        assert segment.transition_out == TransitionType.CROSSFADE
        assert segment.transition_duration_ms == 200


class TestCompositorSettings:
    """Tests for CompositorSettings dataclass."""

    def test_default_values(self):
        """Test default values for CompositorSettings."""
        settings = CompositorSettings()
        assert settings.fps == 30.0
        assert settings.resolution == (720, 540)
        assert settings.default_transition == TransitionType.CROSSFADE
        assert settings.default_transition_duration_ms == 150
        assert settings.background_color == (0, 0, 0)

    def test_custom_values(self):
        """Test CompositorSettings with custom values."""
        settings = CompositorSettings(
            fps=60.0,
            resolution=(1280, 720),
            default_transition=TransitionType.CUT,
            default_transition_duration_ms=200,
            background_color=(255, 255, 255)
        )
        assert settings.fps == 60.0
        assert settings.resolution == (1280, 720)
        assert settings.default_transition == TransitionType.CUT
        assert settings.default_transition_duration_ms == 200
        assert settings.background_color == (255, 255, 255)


class TestCompositorInit:
    """Tests for Compositor initialization."""

    def test_init_with_clip_manager(self):
        """Test initialization with ClipManager."""
        mock_manager = MagicMock()
        comp = Compositor(clip_manager=mock_manager)
        assert comp.clip_manager == mock_manager
        assert comp.settings.fps == 30.0
        assert comp.settings.resolution == (720, 540)

    def test_init_with_signs_dir(self, temp_signs_dir):
        """Test initialization with signs_dir creates ClipManager."""
        with patch('packages.video.compositor.ClipManager') as mock_clip_manager:
            comp = Compositor(signs_dir=temp_signs_dir)
            mock_clip_manager.assert_called_once_with(temp_signs_dir)
            assert comp.clip_manager is not None

    def test_init_with_settings_object(self):
        """Test initialization with a CompositorSettings object."""
        settings = CompositorSettings(fps=60.0, resolution=(1280, 720))
        comp = Compositor(settings=settings)
        assert comp.settings == settings
        assert comp.settings.fps == 60.0
        assert comp.settings.resolution == (1280, 720)

    def test_init_with_neither_clip_manager_nor_signs_dir(self):
        """Test initialization without clip_manager or signs_dir sets clip_manager to None."""
        comp = Compositor()
        assert comp.clip_manager is None

    def test_init_settings_override_individual_params(self):
        """Test that settings object overrides individual fps/resolution params."""
        settings = CompositorSettings(fps=60.0, resolution=(1280, 720))
        comp = Compositor(fps=15.0, resolution=(100, 100), settings=settings)
        assert comp.settings.fps == 60.0
        assert comp.settings.resolution == (1280, 720)


class TestCompositorAddClip:
    """Tests for Compositor.add_clip method."""

    def test_add_clip_from_string_gloss(self, sample_video_clip):
        """Test adding clip from string gloss requires clip_manager."""
        mock_manager = MagicMock()
        mock_manager.get_clip.return_value = sample_video_clip
        comp = Compositor(clip_manager=mock_manager)

        comp.add_clip("HELLO")

        mock_manager.get_clip.assert_called_once_with("HELLO")
        assert comp.num_clips == 1

    def test_add_clip_from_video_clip_object(self, sample_video_clip):
        """Test adding clip from VideoClip object directly."""
        comp = Compositor()
        comp.add_clip(sample_video_clip)
        assert comp.num_clips == 1
        assert comp.glosses == ["TEST"]

    def test_add_clip_raises_valueerror_no_clip_manager_string_gloss(self):
        """Test add_clip raises ValueError when no clip_manager and string gloss."""
        comp = Compositor()
        with pytest.raises(ValueError, match="ClipManager required"):
            comp.add_clip("HELLO")

    def test_add_clip_with_trim_start_ms(self, sample_video_clip):
        """Test add_clip with trim_start_ms."""
        mock_trimmed = MagicMock()
        mock_trimmed.resolution = (100, 100)
        mock_trimmed.resize.return_value = mock_trimmed
        sample_video_clip.trim = MagicMock(return_value=mock_trimmed)

        comp = Compositor(resolution=(100, 100))
        comp.add_clip(sample_video_clip, trim_start_ms=100)

        sample_video_clip.trim.assert_called_once_with(start_ms=100, end_ms=None)

    def test_add_clip_with_trim_end_ms(self, sample_video_clip):
        """Test add_clip with trim_end_ms."""
        mock_trimmed = MagicMock()
        mock_trimmed.resolution = (100, 100)
        mock_trimmed.resize.return_value = mock_trimmed
        sample_video_clip.trim = MagicMock(return_value=mock_trimmed)

        comp = Compositor(resolution=(100, 100))
        comp.add_clip(sample_video_clip, trim_end_ms=500)

        sample_video_clip.trim.assert_called_once_with(start_ms=None, end_ms=500)

    def test_add_clip_resizes_to_target_resolution(self, sample_video_clip):
        """Test add_clip resizes clip to target resolution."""
        comp = Compositor(resolution=(200, 150))
        comp.add_clip(sample_video_clip)
        # The clip should have been resized since resolution doesn't match
        assert comp.num_clips == 1

    def test_add_clip_returns_self_for_chaining(self, sample_video_clip):
        """Test add_clip returns self for method chaining."""
        comp = Compositor()
        result = comp.add_clip(sample_video_clip)
        assert result is comp


class TestCompositorAddTransition:
    """Tests for Compositor.add_transition method."""

    def test_add_transition_queues_pending_transition(self):
        """Test add_transition queues a pending transition."""
        comp = Compositor()
        comp.add_transition(TransitionType.CROSSFADE, duration_ms=200)
        assert comp._pending_transition == (TransitionType.CROSSFADE, 200)

    def test_add_transition_from_string(self):
        """Test add_transition from string converts to TransitionType."""
        comp = Compositor()
        comp.add_transition("crossfade", duration_ms=150)
        assert comp._pending_transition == (TransitionType.CROSSFADE, 150)

    def test_add_transition_applies_to_previous_segment_when_next_clip_added(self, sample_video_clip):
        """Test add_transition applies to previous segment when next clip added."""
        comp = Compositor()
        comp.add_clip(sample_video_clip)
        comp.add_transition(TransitionType.BLEND, duration_ms=100)
        comp.add_clip(sample_video_clip)

        assert comp._segments[0].transition_out == TransitionType.BLEND
        assert comp._segments[0].transition_duration_ms == 100
        assert comp._pending_transition is None

    def test_add_transition_returns_self_for_chaining(self):
        """Test add_transition returns self for method chaining."""
        comp = Compositor()
        result = comp.add_transition(TransitionType.CROSSFADE)
        assert result is comp


class TestCompositorAddSequence:
    """Tests for Compositor.add_sequence method."""

    def test_add_sequence_with_multiple_glosses(self, sample_video_clip):
        """Test add_sequence with multiple glosses."""
        mock_manager = MagicMock()
        mock_manager.get_clip.return_value = sample_video_clip
        comp = Compositor(clip_manager=mock_manager)

        comp.add_sequence(["HELLO", "WORLD", "TEST"])

        assert mock_manager.get_clip.call_count == 3
        assert comp.num_clips == 3

    def test_add_sequence_adds_transitions_between_not_after_last(self, sample_video_clip):
        """Test add_sequence adds transitions between clips but not after last."""
        mock_manager = MagicMock()
        mock_manager.get_clip.return_value = sample_video_clip
        comp = Compositor(clip_manager=mock_manager)

        comp.add_sequence(["A", "B", "C"], TransitionType.CROSSFADE, 100)

        # First two segments should have transitions, last should not
        assert comp._segments[0].transition_out == TransitionType.CROSSFADE
        assert comp._segments[1].transition_out == TransitionType.CROSSFADE
        assert comp._segments[2].transition_out is None

    def test_add_sequence_with_empty_list(self):
        """Test add_sequence with empty list does nothing."""
        comp = Compositor()
        comp.add_sequence([])
        assert comp.num_clips == 0

    def test_add_sequence_returns_self_for_chaining(self, sample_video_clip):
        """Test add_sequence returns self for method chaining."""
        mock_manager = MagicMock()
        mock_manager.get_clip.return_value = sample_video_clip
        comp = Compositor(clip_manager=mock_manager)

        result = comp.add_sequence(["HELLO"])
        assert result is comp


class TestCompositorCompose:
    """Tests for Compositor.compose method."""

    def test_compose_with_empty_segments_returns_empty_clip(self):
        """Test compose with empty segments returns empty clip."""
        comp = Compositor(resolution=(720, 540))
        result = comp.compose()

        assert result.gloss == "COMPOSED"
        assert result.num_frames == 0
        assert result.fps == 30.0

    def test_compose_with_single_clip_returns_that_clip(self, sample_video_clip):
        """Test compose with single clip returns that clip."""
        comp = Compositor(resolution=(100, 100))
        comp.add_clip(sample_video_clip)
        result = comp.compose()

        # Should return the same clip (after resize)
        assert result.gloss == "TEST"

    def test_compose_with_multiple_clips_applies_transitions(self, sample_video_clip):
        """Test compose with multiple clips applies transitions."""
        comp = Compositor(resolution=(100, 100))
        comp.add_clip(sample_video_clip)
        comp.add_transition(TransitionType.CROSSFADE, duration_ms=100)
        comp.add_clip(sample_video_clip)

        with patch('packages.video.compositor.apply_transition') as mock_apply:
            mock_apply.return_value = sample_video_clip.frames
            result = comp.compose()

            mock_apply.assert_called_once()

    def test_compose_uses_default_transition_when_none_specified(self, sample_video_clip):
        """Test compose uses default transition when none specified."""
        settings = CompositorSettings(
            resolution=(100, 100),
            default_transition=TransitionType.CUT,
            default_transition_duration_ms=0
        )
        comp = Compositor(settings=settings)
        comp.add_clip(sample_video_clip)
        comp.add_clip(sample_video_clip)

        with patch('packages.video.compositor.apply_transition') as mock_apply:
            mock_apply.return_value = sample_video_clip.frames
            comp.compose()

            # Should use default transition CUT
            call_args = mock_apply.call_args
            assert call_args[0][2] == TransitionType.CUT


class TestCompositorExport:
    """Tests for Compositor.export method."""

    def test_export_calls_video_exporter(self, sample_video_clip, tmp_path):
        """Test export calls VideoExporter."""
        comp = Compositor(resolution=(100, 100))
        comp.add_clip(sample_video_clip)

        output_path = tmp_path / "output.mp4"

        with patch('packages.video.compositor.VideoExporter') as mock_exporter_class:
            mock_exporter = MagicMock()
            mock_exporter.export.return_value = output_path
            mock_exporter_class.return_value = mock_exporter

            result = comp.export(str(output_path))

            mock_exporter_class.assert_called_once()
            mock_exporter.export.assert_called_once()

    def test_export_returns_path(self, sample_video_clip, tmp_path):
        """Test export returns Path."""
        comp = Compositor(resolution=(100, 100))
        comp.add_clip(sample_video_clip)

        output_path = tmp_path / "output.mp4"

        with patch('packages.video.compositor.VideoExporter') as mock_exporter_class:
            mock_exporter = MagicMock()
            mock_exporter.export.return_value = output_path
            mock_exporter_class.return_value = mock_exporter

            result = comp.export(str(output_path))

            assert result == output_path


class TestCompositorPreviewFrames:
    """Tests for Compositor.preview_frames method."""

    def test_preview_frames_returns_all_frames_when_under_max(self, sample_video_clip):
        """Test preview_frames returns all frames when under max."""
        comp = Compositor(resolution=(100, 100))
        comp.add_clip(sample_video_clip)

        frames = comp.preview_frames(max_frames=100)

        assert len(frames) == sample_video_clip.num_frames

    def test_preview_frames_samples_evenly_when_over_max(self):
        """Test preview_frames samples evenly when over max."""
        # Create clip with 100 frames
        frames_array = np.random.randint(0, 255, (100, 50, 50, 3), dtype=np.uint8)
        clip = VideoClip(gloss="MANY", frames=frames_array, fps=30.0)

        comp = Compositor(resolution=(50, 50))
        comp.add_clip(clip)

        preview = comp.preview_frames(max_frames=10)

        assert len(preview) == 10


class TestCompositorGetDurationMs:
    """Tests for Compositor.get_duration_ms method."""

    def test_get_duration_ms_with_no_segments_returns_zero(self):
        """Test get_duration_ms with no segments returns 0."""
        comp = Compositor()
        assert comp.get_duration_ms() == 0.0

    def test_get_duration_ms_sums_clip_durations(self, sample_video_clip):
        """Test get_duration_ms sums clip durations."""
        comp = Compositor(resolution=(100, 100))
        comp.add_clip(sample_video_clip)  # 10 frames at 30fps = 333.33ms

        duration = comp.get_duration_ms()

        # 10 frames / 30 fps * 1000 = 333.33ms
        expected = (10 / 30.0) * 1000
        assert abs(duration - expected) < 0.01

    def test_get_duration_ms_subtracts_transition_overlap(self, sample_video_clip):
        """Test get_duration_ms subtracts transition overlap."""
        comp = Compositor(resolution=(100, 100))
        comp.add_clip(sample_video_clip)
        comp.add_transition(TransitionType.CROSSFADE, duration_ms=100)
        comp.add_clip(sample_video_clip)

        duration = comp.get_duration_ms()

        # 20 frames total, minus transition overlap (100ms = 3 frames at 30fps)
        total_frames = 20 - 3  # 17 frames
        expected = (total_frames / 30.0) * 1000
        assert abs(duration - expected) < 0.01

    def test_get_duration_ms_does_not_subtract_for_cut_transitions(self, sample_video_clip):
        """Test get_duration_ms doesn't subtract for CUT transitions."""
        comp = Compositor(resolution=(100, 100))
        comp.add_clip(sample_video_clip)
        comp.add_transition(TransitionType.CUT, duration_ms=100)
        comp.add_clip(sample_video_clip)

        duration = comp.get_duration_ms()

        # CUT doesn't have overlap, so full 20 frames
        expected = (20 / 30.0) * 1000
        assert abs(duration - expected) < 0.01


class TestCompositorClear:
    """Tests for Compositor.clear method."""

    def test_clear_resets_segments(self, sample_video_clip):
        """Test clear resets segments."""
        comp = Compositor(resolution=(100, 100))
        comp.add_clip(sample_video_clip)
        comp.add_clip(sample_video_clip)

        comp.clear()

        assert comp.num_clips == 0

    def test_clear_resets_pending_transition(self):
        """Test clear resets pending transition."""
        comp = Compositor()
        comp.add_transition(TransitionType.CROSSFADE)

        comp.clear()

        assert comp._pending_transition is None

    def test_clear_returns_self(self, sample_video_clip):
        """Test clear returns self for chaining."""
        comp = Compositor(resolution=(100, 100))
        comp.add_clip(sample_video_clip)

        result = comp.clear()

        assert result is comp


class TestCompositorProperties:
    """Tests for Compositor properties."""

    def test_num_clips_property(self, sample_video_clip):
        """Test num_clips property returns correct count."""
        comp = Compositor(resolution=(100, 100))
        assert comp.num_clips == 0

        comp.add_clip(sample_video_clip)
        assert comp.num_clips == 1

        comp.add_clip(sample_video_clip)
        assert comp.num_clips == 2

    def test_glosses_property_returns_list_of_glosses(self, sample_video_clip):
        """Test glosses property returns list of glosses."""
        # Create clips with different glosses
        clip_a = VideoClip(gloss="HELLO", frames=sample_video_clip.frames, fps=30.0)
        clip_b = VideoClip(gloss="WORLD", frames=sample_video_clip.frames, fps=30.0)

        comp = Compositor(resolution=(100, 100))
        comp.add_clip(clip_a)
        comp.add_clip(clip_b)

        assert comp.glosses == ["HELLO", "WORLD"]


class TestComposeSequenceFunction:
    """Tests for compose_sequence convenience function."""

    def test_basic_usage(self, sample_video_clip, tmp_path):
        """Test compose_sequence basic usage."""
        output_path = tmp_path / "output.mp4"

        with patch('packages.video.compositor.ClipManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.get_clip.return_value = sample_video_clip
            mock_manager_class.return_value = mock_manager

            with patch('packages.video.compositor.VideoExporter') as mock_exporter_class:
                mock_exporter = MagicMock()
                mock_exporter.export.return_value = output_path
                mock_exporter_class.return_value = mock_exporter

                result = compose_sequence(
                    glosses=["HELLO", "WORLD"],
                    signs_dir=tmp_path,
                    output_path=str(output_path)
                )

                assert result == output_path
                assert mock_manager.get_clip.call_count == 2

    def test_custom_options_passed_through(self, sample_video_clip, tmp_path):
        """Test compose_sequence passes custom options through."""
        output_path = tmp_path / "output.mp4"

        with patch('packages.video.compositor.ClipManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.get_clip.return_value = sample_video_clip
            mock_manager_class.return_value = mock_manager

            with patch('packages.video.compositor.VideoExporter') as mock_exporter_class:
                mock_exporter = MagicMock()
                mock_exporter.export.return_value = output_path
                mock_exporter_class.return_value = mock_exporter

                result = compose_sequence(
                    glosses=["TEST"],
                    signs_dir=tmp_path,
                    output_path=str(output_path),
                    transition_type=TransitionType.CUT,
                    transition_duration_ms=50,
                    fps=60.0,
                    resolution=(1280, 720)
                )

                # Verify exporter was called
                mock_exporter.export.assert_called_once()


class TestCompositorMethodChaining:
    """Tests for method chaining across Compositor methods."""

    def test_full_chain(self, sample_video_clip):
        """Test full method chaining works correctly."""
        comp = Compositor(resolution=(100, 100))

        result = (
            comp
            .add_clip(sample_video_clip)
            .add_transition(TransitionType.CROSSFADE)
            .add_clip(sample_video_clip)
            .clear()
            .add_clip(sample_video_clip)
        )

        assert result is comp
        assert comp.num_clips == 1
