"""Tests for the verify command."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.main import app
from packages.database import Sign, SignStatus, VideoInfo


runner = CliRunner()


class TestVerifyCommand:
    """Tests for the verify command."""

    def test_verify_with_args(self, mock_store, mock_pending_sign, mock_verifier):
        """Test verifying a sign with all arguments provided."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_store.get_sign.return_value = mock_pending_sign

            verified_sign = Sign(
                gloss="WORLD",
                english=["world"],
                status=SignStatus.VERIFIED,
                quality_score=5,
                verified_by="john",
                video=VideoInfo(file="video.mp4"),
            )
            mock_verifier.verify.return_value = verified_sign

            result = runner.invoke(
                app,
                ["verify", "WORLD", "--score", "5", "--by", "john"],
            )

            assert result.exit_code == 0
            assert "Verified WORLD" in result.output
            assert "score 5/5" in result.output
            mock_verifier.verify.assert_called_once_with("WORLD", 5, "john")

    def test_verify_sign_not_found(self, mock_store, mock_verifier):
        """Test error when sign doesn't exist."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_store.get_sign.return_value = None

            result = runner.invoke(
                app,
                ["verify", "NONEXISTENT", "--score", "5", "--by", "john"],
            )

            assert result.exit_code == 1
            assert "not found" in result.output
            mock_verifier.verify.assert_not_called()

    def test_verify_already_verified(self, mock_store, mock_sign, mock_verifier):
        """Test message when sign is already verified."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_store.get_sign.return_value = mock_sign  # mock_sign has VERIFIED status

            result = runner.invoke(
                app,
                ["verify", "HELLO", "--score", "5", "--by", "john"],
            )

            assert result.exit_code == 0
            assert "already verified" in result.output
            mock_verifier.verify.assert_not_called()

    def test_verify_with_reject_flag(self, mock_store, mock_pending_sign, mock_verifier):
        """Test rejection flow with --reject flag."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_store.get_sign.return_value = mock_pending_sign

            rejected_sign = Sign(
                gloss="WORLD",
                english=["world"],
                status=SignStatus.REJECTED,
                video=VideoInfo(file="video.mp4"),
            )
            mock_verifier.reject.return_value = rejected_sign

            result = runner.invoke(
                app,
                ["verify", "WORLD", "--reject", "--reason", "Poor quality", "--by", "john"],
            )

            assert result.exit_code == 0
            assert "Rejected WORLD" in result.output
            mock_verifier.reject.assert_called_once_with("WORLD", "Poor quality", "john")

    def test_verify_prompts_for_missing_score(self, mock_store, mock_pending_sign, mock_verifier):
        """Test that verify prompts for score when not provided."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_store.get_sign.return_value = mock_pending_sign

            verified_sign = Sign(
                gloss="WORLD",
                english=["world"],
                status=SignStatus.VERIFIED,
                quality_score=4,
                verified_by="john",
                video=VideoInfo(file="video.mp4"),
            )
            mock_verifier.verify.return_value = verified_sign

            # Provide score and name via stdin
            result = runner.invoke(
                app,
                ["verify", "WORLD", "--by", "john"],
                input="4\n",  # Score input
            )

            assert result.exit_code == 0
            assert "Verified WORLD" in result.output

    def test_verify_prompts_for_missing_by(self, mock_store, mock_pending_sign, mock_verifier):
        """Test that verify prompts for verifier name when not provided."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_store.get_sign.return_value = mock_pending_sign

            verified_sign = Sign(
                gloss="WORLD",
                english=["world"],
                status=SignStatus.VERIFIED,
                quality_score=5,
                verified_by="jane",
                video=VideoInfo(file="video.mp4"),
            )
            mock_verifier.verify.return_value = verified_sign

            # Provide name via stdin
            result = runner.invoke(
                app,
                ["verify", "WORLD", "--score", "5"],
                input="jane\n",  # Name input
            )

            assert result.exit_code == 0
            assert "Verified WORLD" in result.output

    def test_verify_interactive_mode_empty_queue(self, mock_store, mock_verifier):
        """Test interactive mode when no signs are pending verification."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_verifier.get_verification_queue.return_value = []

            result = runner.invoke(app, ["verify", "--interactive"])

            assert result.exit_code == 0
            assert "No signs awaiting verification" in result.output

    def test_verify_error_handling(self, mock_store, mock_pending_sign, mock_verifier):
        """Test error handling when verifier raises ValueError."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_store.get_sign.return_value = mock_pending_sign
            mock_verifier.verify.side_effect = ValueError("Invalid score")

            result = runner.invoke(
                app,
                ["verify", "WORLD", "--score", "5", "--by", "john"],
            )

            assert result.exit_code == 1
            assert "Error" in result.output

    def test_verify_normalizes_gloss_to_uppercase(self, mock_store, mock_pending_sign, mock_verifier):
        """Test that gloss is normalized to uppercase."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_store.get_sign.return_value = mock_pending_sign

            verified_sign = Sign(
                gloss="WORLD",
                english=["world"],
                status=SignStatus.VERIFIED,
                quality_score=5,
                verified_by="john",
                video=VideoInfo(file="video.mp4"),
            )
            mock_verifier.verify.return_value = verified_sign

            result = runner.invoke(
                app,
                ["verify", "world", "--score", "5", "--by", "john"],
            )

            assert result.exit_code == 0
            mock_store.get_sign.assert_called_with("WORLD")

    def test_verify_reject_prompts_for_reason(self, mock_store, mock_pending_sign, mock_verifier):
        """Test that reject prompts for reason when not provided."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_store.get_sign.return_value = mock_pending_sign

            rejected_sign = Sign(
                gloss="WORLD",
                english=["world"],
                status=SignStatus.REJECTED,
                video=VideoInfo(file="video.mp4"),
            )
            mock_verifier.reject.return_value = rejected_sign

            # Provide reason and name via stdin
            result = runner.invoke(
                app,
                ["verify", "WORLD", "--reject"],
                input="Poor quality video\njohn\n",
            )

            assert result.exit_code == 0
            assert "Rejected WORLD" in result.output

    def test_verify_shows_quality_issues(self, mock_store, mock_pending_sign, mock_verifier):
        """Test that quality issues are displayed during verification."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_store.get_sign.return_value = mock_pending_sign

            # Mock quality check with issues
            quality_result = MagicMock()
            quality_result.passed = False
            quality_result.score = 2
            quality_result.issues = ["Video too short", "Poor lighting"]
            quality_result.suggestions = ["Re-record with better lighting"]
            mock_verifier.check_sign_quality.return_value = quality_result

            verified_sign = Sign(
                gloss="WORLD",
                english=["world"],
                status=SignStatus.VERIFIED,
                quality_score=3,
                verified_by="john",
                video=VideoInfo(file="video.mp4"),
            )
            mock_verifier.verify.return_value = verified_sign

            # Provide score and name
            result = runner.invoke(
                app,
                ["verify", "WORLD"],
                input="3\njohn\n",
            )

            assert result.exit_code == 0
            assert "Issues found" in result.output
            assert "Video too short" in result.output

    def test_verify_interactive_with_queue(self, mock_store, mock_pending_sign, mock_verifier):
        """Test interactive mode processes verification queue."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_verifier.get_verification_queue.return_value = [mock_pending_sign]

            verified_sign = Sign(
                gloss="WORLD",
                english=["world"],
                status=SignStatus.VERIFIED,
                quality_score=5,
                verified_by="john",
                video=VideoInfo(file="video.mp4"),
            )
            mock_verifier.verify.return_value = verified_sign

            # Provide action, score, and name
            result = runner.invoke(
                app,
                ["verify", "--interactive"],
                input="verify\n5\njohn\n",
            )

            assert result.exit_code == 0
            assert "1 signs in verification queue" in result.output
            assert "Verified WORLD" in result.output

    def test_verify_interactive_skip_action(self, mock_store, mock_pending_sign, mock_verifier):
        """Test skip action in interactive mode."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_verifier.get_verification_queue.return_value = [mock_pending_sign]

            result = runner.invoke(
                app,
                ["verify", "--interactive"],
                input="skip\n",
            )

            assert result.exit_code == 0
            mock_verifier.verify.assert_not_called()
            mock_verifier.reject.assert_not_called()

    def test_verify_interactive_quit_action(self, mock_store, mock_pending_sign, mock_verifier):
        """Test quit action in interactive mode."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_verifier.get_verification_queue.return_value = [mock_pending_sign]

            result = runner.invoke(
                app,
                ["verify", "--interactive"],
                input="quit\n",
            )

            assert result.exit_code == 0
            mock_verifier.verify.assert_not_called()

    def test_verify_shows_summary_stats(self, mock_store, mock_verifier):
        """Test that verification summary stats are shown after interactive mode."""
        with patch("src.commands.verify.get_store") as mock_get_store, \
             patch("src.commands.verify.get_verifier") as mock_get_verifier:
            mock_get_store.return_value = mock_store
            mock_get_verifier.return_value = mock_verifier
            mock_verifier.get_verification_queue.return_value = []

            result = runner.invoke(app, ["verify", "--interactive"])

            assert result.exit_code == 0
            # Empty queue shows message, no stats
            assert "No signs awaiting verification" in result.output
