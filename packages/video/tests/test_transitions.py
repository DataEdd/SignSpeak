"""Tests for transitions module."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from packages.video.transitions import (
    TransitionType,
    apply_transition,
    _apply_cut,
    _apply_crossfade,
    _apply_blend,
    _interpolate_frames,
    _linear_interpolation,
    _optical_flow_interpolation,
    ms_to_frames,
    frames_to_ms,
)


class TestTransitionType:
    """Tests for TransitionType enum."""

    def test_cut_value(self):
        """Test CUT enum value equals 'cut'."""
        assert TransitionType.CUT.value == "cut"

    def test_crossfade_value(self):
        """Test CROSSFADE enum value equals 'crossfade'."""
        assert TransitionType.CROSSFADE.value == "crossfade"

    def test_blend_value(self):
        """Test BLEND enum value equals 'blend'."""
        assert TransitionType.BLEND.value == "blend"


class TestApplyTransition:
    """Tests for apply_transition dispatcher function."""

    def test_dispatches_to_cut(self, small_frames):
        """Test apply_transition dispatches to _apply_cut for CUT type."""
        clip_a = small_frames[:3]
        clip_b = small_frames[3:]

        with patch("packages.video.transitions._apply_cut") as mock_cut:
            mock_cut.return_value = np.concatenate([clip_a, clip_b])
            result = apply_transition(clip_a, clip_b, TransitionType.CUT, 2)
            mock_cut.assert_called_once()
            # Verify correct arguments passed
            np.testing.assert_array_equal(mock_cut.call_args[0][0], clip_a)
            np.testing.assert_array_equal(mock_cut.call_args[0][1], clip_b)

    def test_dispatches_to_crossfade(self, small_frames):
        """Test apply_transition dispatches to _apply_crossfade for CROSSFADE type."""
        clip_a = small_frames[:3]
        clip_b = small_frames[3:]

        with patch("packages.video.transitions._apply_crossfade") as mock_crossfade:
            mock_crossfade.return_value = small_frames
            result = apply_transition(clip_a, clip_b, TransitionType.CROSSFADE, 2)
            mock_crossfade.assert_called_once()
            np.testing.assert_array_equal(mock_crossfade.call_args[0][0], clip_a)
            np.testing.assert_array_equal(mock_crossfade.call_args[0][1], clip_b)
            assert mock_crossfade.call_args[0][2] == 2

    def test_dispatches_to_blend(self, small_frames):
        """Test apply_transition dispatches to _apply_blend for BLEND type."""
        clip_a = small_frames[:3]
        clip_b = small_frames[3:]

        with patch("packages.video.transitions._apply_blend") as mock_blend:
            mock_blend.return_value = small_frames
            result = apply_transition(clip_a, clip_b, TransitionType.BLEND, 2)
            mock_blend.assert_called_once()
            np.testing.assert_array_equal(mock_blend.call_args[0][0], clip_a)
            np.testing.assert_array_equal(mock_blend.call_args[0][1], clip_b)
            assert mock_blend.call_args[0][2] == 2

    def test_raises_for_unknown_type(self, small_frames):
        """Test apply_transition raises ValueError for unknown transition type."""
        clip_a = small_frames[:3]
        clip_b = small_frames[3:]

        with pytest.raises(ValueError, match="Unknown transition type"):
            apply_transition(clip_a, clip_b, "invalid_type", 2)


class TestApplyCut:
    """Tests for _apply_cut function."""

    def test_concatenates_frames(self):
        """Test cut transition concatenates frames correctly."""
        clip_a = np.zeros((3, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((4, 10, 10, 3), dtype=np.uint8) * 255

        result = _apply_cut(clip_a, clip_b)

        assert result.shape[0] == 7  # 3 + 4
        np.testing.assert_array_equal(result[:3], clip_a)
        np.testing.assert_array_equal(result[3:], clip_b)

    def test_handles_empty_clip_a(self):
        """Test cut transition handles empty clip_a."""
        clip_a = np.empty((0, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((4, 10, 10, 3), dtype=np.uint8) * 255

        result = _apply_cut(clip_a, clip_b)

        assert result.shape[0] == 4
        np.testing.assert_array_equal(result, clip_b)

    def test_handles_empty_clip_b(self):
        """Test cut transition handles empty clip_b."""
        clip_a = np.zeros((3, 10, 10, 3), dtype=np.uint8)
        clip_b = np.empty((0, 10, 10, 3), dtype=np.uint8)

        result = _apply_cut(clip_a, clip_b)

        assert result.shape[0] == 3
        np.testing.assert_array_equal(result, clip_a)


class TestApplyCrossfade:
    """Tests for _apply_crossfade function."""

    def test_returns_clip_b_when_clip_a_empty(self):
        """Test crossfade returns clip_b when clip_a is empty."""
        clip_a = np.empty((0, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((4, 10, 10, 3), dtype=np.uint8) * 128

        result = _apply_crossfade(clip_a, clip_b, 2)

        np.testing.assert_array_equal(result, clip_b)

    def test_returns_clip_a_when_clip_b_empty(self):
        """Test crossfade returns clip_a when clip_b is empty."""
        clip_a = np.ones((4, 10, 10, 3), dtype=np.uint8) * 128
        clip_b = np.empty((0, 10, 10, 3), dtype=np.uint8)

        result = _apply_crossfade(clip_a, clip_b, 2)

        np.testing.assert_array_equal(result, clip_a)

    def test_clamps_duration_to_min_of_clip_lengths(self):
        """Test crossfade clamps duration to minimum of clip lengths."""
        clip_a = np.zeros((2, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((5, 10, 10, 3), dtype=np.uint8) * 255

        # Request 10 frames but only 2 available in clip_a
        result = _apply_crossfade(clip_a, clip_b, 10)

        # Result length should be clip_a + clip_b - min(duration, len_a, len_b)
        # = 2 + 5 - 2 = 5
        assert result.shape[0] == 5

    def test_returns_concatenation_when_duration_zero(self):
        """Test crossfade returns simple concatenation when duration <= 0."""
        clip_a = np.zeros((3, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((4, 10, 10, 3), dtype=np.uint8) * 255

        result = _apply_crossfade(clip_a, clip_b, 0)

        assert result.shape[0] == 7
        np.testing.assert_array_equal(result[:3], clip_a)
        np.testing.assert_array_equal(result[3:], clip_b)

    def test_blends_frames_correctly(self):
        """Test crossfade blends frames with correct alpha progression."""
        # Use solid colors for predictable blending
        clip_a = np.zeros((3, 10, 10, 3), dtype=np.uint8)  # Black
        clip_b = np.ones((3, 10, 10, 3), dtype=np.uint8) * 254  # Near-white

        result = _apply_crossfade(clip_a, clip_b, 2)

        # Result should have 4 frames (3 + 3 - 2)
        assert result.shape[0] == 4

        # First frame of transition (index 1) should be mostly black (alpha=0)
        # Last frame of transition (index 2) should be mostly white (alpha=1)
        # Due to alpha = i/(duration-1), first blend is alpha=0, second is alpha=1
        assert np.mean(result[1]) < 10  # Should be mostly black
        assert np.mean(result[2]) > 240  # Should be mostly white

    def test_result_length_equals_clip_a_plus_clip_b_minus_duration(self):
        """Test crossfade result length equals clip_a + clip_b - duration_frames."""
        clip_a = np.zeros((5, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((4, 10, 10, 3), dtype=np.uint8) * 255

        result = _apply_crossfade(clip_a, clip_b, 3)

        # 5 + 4 - 3 = 6
        assert result.shape[0] == 6

    def test_handles_duration_one(self):
        """Test crossfade handles duration_frames = 1."""
        clip_a = np.zeros((3, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((3, 10, 10, 3), dtype=np.uint8) * 255

        result = _apply_crossfade(clip_a, clip_b, 1)

        # 3 + 3 - 1 = 5
        assert result.shape[0] == 5
        # When duration=1, alpha = i/(1-1) = i/0 which is handled as alpha=1.0
        # So the single transition frame should be clip_b's first frame
        np.testing.assert_array_equal(result[2], clip_b[0])

    def test_handles_single_frame_clips(self):
        """Test crossfade handles single frame clips."""
        clip_a = np.zeros((1, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((1, 10, 10, 3), dtype=np.uint8) * 255

        result = _apply_crossfade(clip_a, clip_b, 1)

        # 1 + 1 - 1 = 1
        assert result.shape[0] == 1


class TestApplyBlend:
    """Tests for _apply_blend function."""

    def test_returns_clip_b_when_clip_a_empty(self):
        """Test blend returns clip_b when clip_a is empty."""
        clip_a = np.empty((0, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((4, 10, 10, 3), dtype=np.uint8) * 128

        result = _apply_blend(clip_a, clip_b, 2)

        np.testing.assert_array_equal(result, clip_b)

    def test_returns_clip_a_when_clip_b_empty(self):
        """Test blend returns clip_a when clip_b is empty."""
        clip_a = np.ones((4, 10, 10, 3), dtype=np.uint8) * 128
        clip_b = np.empty((0, 10, 10, 3), dtype=np.uint8)

        result = _apply_blend(clip_a, clip_b, 2)

        np.testing.assert_array_equal(result, clip_a)

    def test_clamps_duration(self):
        """Test blend clamps duration to minimum of clip lengths."""
        clip_a = np.zeros((2, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((5, 10, 10, 3), dtype=np.uint8) * 255

        result = _apply_blend(clip_a, clip_b, 10)

        # Duration clamped to 2
        # Result = (len_a - 1) + duration + (len_b - 1) = 1 + 2 + 4 = 7
        assert result.shape[0] == 7

    def test_returns_concatenation_when_duration_zero(self):
        """Test blend returns simple concatenation when duration <= 0."""
        clip_a = np.zeros((3, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((4, 10, 10, 3), dtype=np.uint8) * 255

        result = _apply_blend(clip_a, clip_b, 0)

        assert result.shape[0] == 7
        np.testing.assert_array_equal(result[:3], clip_a)
        np.testing.assert_array_equal(result[3:], clip_b)

    def test_calls_interpolate_frames(self):
        """Test blend calls _interpolate_frames with correct arguments."""
        clip_a = np.zeros((3, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((4, 10, 10, 3), dtype=np.uint8) * 255

        with patch("packages.video.transitions._interpolate_frames") as mock_interp:
            mock_interp.return_value = np.zeros((2, 10, 10, 3), dtype=np.uint8)
            result = _apply_blend(clip_a, clip_b, 2)
            mock_interp.assert_called_once()
            # Should pass last frame of clip_a and first frame of clip_b
            np.testing.assert_array_equal(mock_interp.call_args[0][0], clip_a[-1])
            np.testing.assert_array_equal(mock_interp.call_args[0][1], clip_b[0])
            assert mock_interp.call_args[0][2] == 2

    def test_result_removes_last_of_clip_a_and_first_of_clip_b(self):
        """Test blend result removes last frame of clip_a and first frame of clip_b."""
        clip_a = np.zeros((3, 10, 10, 3), dtype=np.uint8)
        clip_a[0] = 10  # Mark first
        clip_a[1] = 20  # Mark second
        clip_a[2] = 30  # Mark last (should be removed)

        clip_b = np.ones((4, 10, 10, 3), dtype=np.uint8) * 255
        clip_b[0] = 100  # Mark first (should be removed)
        clip_b[1] = 150  # Mark second
        clip_b[2] = 200  # Mark third
        clip_b[3] = 250  # Mark fourth

        with patch("packages.video.transitions._interpolate_frames") as mock_interp:
            mock_interp.return_value = np.ones((2, 10, 10, 3), dtype=np.uint8) * 128
            result = _apply_blend(clip_a, clip_b, 2)

        # Should have: clip_a[:-1] + interpolated + clip_b[1:]
        # = 2 + 2 + 3 = 7 frames
        assert result.shape[0] == 7
        # First two frames should be from clip_a (excluding last)
        assert np.mean(result[0]) == 10
        assert np.mean(result[1]) == 20
        # Middle two frames are interpolated
        assert np.mean(result[2]) == 128
        assert np.mean(result[3]) == 128
        # Last three frames should be from clip_b (excluding first)
        assert np.mean(result[4]) == 150
        assert np.mean(result[5]) == 200
        assert np.mean(result[6]) == 250

    def test_result_length_calculation(self):
        """Test blend result length is (len_a - 1) + duration + (len_b - 1)."""
        clip_a = np.zeros((5, 10, 10, 3), dtype=np.uint8)
        clip_b = np.ones((4, 10, 10, 3), dtype=np.uint8) * 255

        with patch("packages.video.transitions._interpolate_frames") as mock_interp:
            mock_interp.return_value = np.zeros((3, 10, 10, 3), dtype=np.uint8)
            result = _apply_blend(clip_a, clip_b, 3)

        # (5-1) + 3 + (4-1) = 4 + 3 + 3 = 10
        assert result.shape[0] == 10


class TestInterpolateFrames:
    """Tests for _interpolate_frames function."""

    def test_tries_optical_flow_first(self, single_frame):
        """Test _interpolate_frames tries optical flow interpolation first."""
        frame_a = single_frame
        frame_b = np.roll(single_frame, 5, axis=0)

        with patch("packages.video.transitions._optical_flow_interpolation") as mock_optical:
            mock_optical.return_value = np.zeros((3, 100, 100, 3), dtype=np.uint8)
            result = _interpolate_frames(frame_a, frame_b, 3)
            mock_optical.assert_called_once()

    def test_falls_back_to_linear_on_exception(self, single_frame):
        """Test _interpolate_frames falls back to linear interpolation on exception."""
        frame_a = single_frame
        frame_b = np.roll(single_frame, 5, axis=0)

        with patch("packages.video.transitions._optical_flow_interpolation") as mock_optical:
            with patch("packages.video.transitions._linear_interpolation") as mock_linear:
                mock_optical.side_effect = Exception("Optical flow failed")
                mock_linear.return_value = np.zeros((3, 100, 100, 3), dtype=np.uint8)
                result = _interpolate_frames(frame_a, frame_b, 3)
                mock_linear.assert_called_once()


class TestLinearInterpolation:
    """Tests for _linear_interpolation function."""

    def test_single_frame_case(self):
        """Test linear interpolation with single frame returns frame_b."""
        frame_a = np.zeros((10, 10, 3), dtype=np.uint8)
        frame_b = np.ones((10, 10, 3), dtype=np.uint8) * 255

        result = _linear_interpolation(frame_a, frame_b, 1)

        assert result.shape == (1, 10, 10, 3)
        # When num_frames=1, alpha=1.0, so result should be frame_b
        np.testing.assert_array_equal(result[0], frame_b)

    def test_multiple_frames_case(self):
        """Test linear interpolation with multiple frames."""
        frame_a = np.zeros((10, 10, 3), dtype=np.uint8)
        frame_b = np.ones((10, 10, 3), dtype=np.uint8) * 255

        result = _linear_interpolation(frame_a, frame_b, 5)

        assert result.shape == (5, 10, 10, 3)

    def test_alpha_progression_from_zero_to_one(self):
        """Test linear interpolation alpha progresses from 0 to 1."""
        frame_a = np.zeros((10, 10, 3), dtype=np.uint8)
        frame_b = np.ones((10, 10, 3), dtype=np.uint8) * 200

        result = _linear_interpolation(frame_a, frame_b, 5)

        # First frame: alpha=0, should be frame_a (black)
        assert np.mean(result[0]) < 5

        # Last frame: alpha=1, should be frame_b (200)
        assert np.mean(result[4]) > 195

        # Middle frames should progress
        means = [np.mean(result[i]) for i in range(5)]
        assert means == sorted(means)  # Should be monotonically increasing

    def test_output_dtype_is_uint8(self):
        """Test linear interpolation output dtype is uint8."""
        frame_a = np.zeros((10, 10, 3), dtype=np.uint8)
        frame_b = np.ones((10, 10, 3), dtype=np.uint8) * 255

        result = _linear_interpolation(frame_a, frame_b, 3)

        assert result.dtype == np.uint8

    def test_output_shape(self):
        """Test linear interpolation output shape is (num_frames, h, w, 3)."""
        frame_a = np.zeros((20, 30, 3), dtype=np.uint8)
        frame_b = np.ones((20, 30, 3), dtype=np.uint8) * 255

        result = _linear_interpolation(frame_a, frame_b, 7)

        assert result.shape == (7, 20, 30, 3)


class TestOpticalFlowInterpolation:
    """Tests for _optical_flow_interpolation function."""

    def test_calculates_optical_flow(self):
        """Test optical flow interpolation calculates optical flow."""
        frame_a = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        frame_b = np.roll(frame_a, 2, axis=1)  # Shift horizontally

        with patch.object(cv2, "calcOpticalFlowFarneback") as mock_flow:
            mock_flow.return_value = np.zeros((50, 50, 2), dtype=np.float32)
            with patch.object(cv2, "remap") as mock_remap:
                mock_remap.return_value = frame_a
                result = _optical_flow_interpolation(frame_a, frame_b, 3)
                mock_flow.assert_called_once()

    def test_output_shape_is_correct(self):
        """Test optical flow interpolation output shape is correct."""
        frame_a = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        frame_b = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        result = _optical_flow_interpolation(frame_a, frame_b, 5)

        assert result.shape == (5, 50, 50, 3)

    def test_warps_frames_based_on_flow(self):
        """Test optical flow interpolation warps frames using cv2.remap."""
        frame_a = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        frame_b = np.roll(frame_a, 2, axis=1)

        with patch.object(cv2, "remap", wraps=cv2.remap) as mock_remap:
            result = _optical_flow_interpolation(frame_a, frame_b, 3)
            # remap should be called 2 times per frame (forward and backward)
            # 3 frames * 2 = 6 calls
            assert mock_remap.call_count == 6

    def test_blends_warped_frames(self):
        """Test optical flow interpolation blends warped frames."""
        frame_a = np.zeros((50, 50, 3), dtype=np.uint8)
        frame_b = np.ones((50, 50, 3), dtype=np.uint8) * 200

        result = _optical_flow_interpolation(frame_a, frame_b, 5)

        # First frame should be closer to frame_a, last closer to frame_b
        assert np.mean(result[0]) < np.mean(result[4])


class TestMsToFrames:
    """Tests for ms_to_frames utility function."""

    def test_basic_conversion(self):
        """Test 100ms at 30fps = 3 frames."""
        result = ms_to_frames(100, 30.0)
        assert result == 3

    def test_minimum_is_one(self):
        """Test ms_to_frames minimum is 1 frame."""
        result = ms_to_frames(1, 30.0)  # Very small duration
        assert result == 1

    def test_with_zero_returns_one(self):
        """Test ms_to_frames with zero duration returns 1."""
        result = ms_to_frames(0, 30.0)
        assert result == 1

    def test_exact_second(self):
        """Test 1000ms at 30fps = 30 frames."""
        result = ms_to_frames(1000, 30.0)
        assert result == 30

    def test_different_fps(self):
        """Test conversion at different fps values."""
        assert ms_to_frames(1000, 60.0) == 60
        assert ms_to_frames(1000, 24.0) == 24


class TestFramesToMs:
    """Tests for frames_to_ms utility function."""

    def test_basic_conversion(self):
        """Test 30 frames at 30fps = 1000ms."""
        result = frames_to_ms(30, 30.0)
        assert result == 1000.0

    def test_with_one_frame(self):
        """Test 1 frame at 30fps = ~33.33ms."""
        result = frames_to_ms(1, 30.0)
        assert abs(result - 33.333) < 0.01

    def test_different_fps(self):
        """Test conversion at different fps values."""
        assert frames_to_ms(60, 60.0) == 1000.0
        assert frames_to_ms(24, 24.0) == 1000.0


class TestRoundtripConversion:
    """Tests for roundtrip consistency between ms and frames."""

    def test_roundtrip_consistency(self):
        """Test ms -> frames -> ms is approximately equal."""
        original_ms = 500.0
        fps = 30.0

        frames = ms_to_frames(original_ms, fps)
        result_ms = frames_to_ms(frames, fps)

        # Should be within one frame's worth of time
        frame_duration_ms = 1000.0 / fps
        assert abs(result_ms - original_ms) <= frame_duration_ms

    def test_roundtrip_at_various_durations(self):
        """Test roundtrip consistency at various duration values."""
        fps = 30.0
        frame_duration_ms = 1000.0 / fps

        for original_ms in [100, 250, 500, 1000, 2000]:
            frames = ms_to_frames(original_ms, fps)
            result_ms = frames_to_ms(frames, fps)
            assert abs(result_ms - original_ms) <= frame_duration_ms
