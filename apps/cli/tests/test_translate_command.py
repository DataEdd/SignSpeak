"""Tests for the translate command."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.main import app
from packages.database import Sign, SignStatus, VideoInfo


runner = CliRunner()


class TestTranslateCommand:
    """Tests for the translate command."""

    def test_translate_shows_glosses(self, mock_store, mock_translation_result):
        """Test that glosses are displayed by default."""
        with patch("src.commands.translate.get_store") as mock_get_store, \
             patch("src.commands.translate.get_signs_dir") as mock_get_signs_dir, \
             patch("src.commands.translate.translate_text") as mock_translate:
            mock_get_store.return_value = mock_store
            mock_get_signs_dir.return_value = Path("/tmp/signs")
            mock_translate.return_value = mock_translation_result

            result = runner.invoke(
                app,
                ["translate", "Hello, how are you?"],
            )

            assert result.exit_code == 0
            assert "Glosses:" in result.output
            assert "HELLO" in result.output
            assert "HOW" in result.output
            assert "YOU" in result.output

    def test_translate_with_output(self, mock_store, mock_sign, mock_translation_result):
        """Test video generation when -o is specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            with patch("src.commands.translate.get_store") as mock_get_store, \
                 patch("src.commands.translate.get_signs_dir") as mock_get_signs_dir, \
                 patch("src.commands.translate.translate_text") as mock_translate, \
                 patch("src.commands.translate.compose_sequence") as mock_compose:
                mock_get_store.return_value = mock_store
                mock_get_signs_dir.return_value = Path(tmpdir)
                mock_translate.return_value = mock_translation_result

                # Mock get_verified_sign to return sign for HELLO and YOU
                def get_verified(gloss):
                    if gloss in ["HELLO", "YOU"]:
                        return mock_sign
                    return None
                mock_store.get_verified_sign.side_effect = get_verified

                mock_compose.return_value = str(output_path)

                result = runner.invoke(
                    app,
                    ["translate", "Hello, how are you?", "-o", str(output_path)],
                    input="y\n",  # Confirm low coverage
                )

                assert result.exit_code == 0
                assert "Saved to" in result.output
                mock_compose.assert_called_once()

    def test_translate_no_glosses_flag(self, mock_store, mock_translation_result):
        """Test that --no-glosses hides glosses output."""
        with patch("src.commands.translate.get_store") as mock_get_store, \
             patch("src.commands.translate.get_signs_dir") as mock_get_signs_dir, \
             patch("src.commands.translate.translate_text") as mock_translate:
            mock_get_store.return_value = mock_store
            mock_get_signs_dir.return_value = Path("/tmp/signs")
            mock_translate.return_value = mock_translation_result

            result = runner.invoke(
                app,
                ["translate", "Hello", "--no-glosses"],
            )

            assert result.exit_code == 0
            assert "Glosses:" not in result.output

    def test_translate_with_speed_option_slow(self, mock_store, mock_sign, mock_translation_result):
        """Test --speed slow option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            with patch("src.commands.translate.get_store") as mock_get_store, \
                 patch("src.commands.translate.get_signs_dir") as mock_get_signs_dir, \
                 patch("src.commands.translate.translate_text") as mock_translate, \
                 patch("src.commands.translate.compose_sequence") as mock_compose:
                mock_get_store.return_value = mock_store
                mock_get_signs_dir.return_value = Path(tmpdir)

                # High coverage result
                high_coverage_result = MagicMock()
                high_coverage_result.glosses = ["HELLO"]
                high_coverage_result.validation = MagicMock()
                high_coverage_result.validation.coverage = 1.0
                high_coverage_result.validation.missing = []
                mock_translate.return_value = high_coverage_result

                mock_store.get_verified_sign.return_value = mock_sign
                mock_compose.return_value = str(output_path)

                result = runner.invoke(
                    app,
                    ["translate", "Hello", "-o", str(output_path), "--speed", "slow"],
                )

                assert result.exit_code == 0
                call_kwargs = mock_compose.call_args[1]
                assert call_kwargs["transition_duration_ms"] == 250

    def test_translate_with_speed_option_fast(self, mock_store, mock_sign, mock_translation_result):
        """Test --speed fast option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            with patch("src.commands.translate.get_store") as mock_get_store, \
                 patch("src.commands.translate.get_signs_dir") as mock_get_signs_dir, \
                 patch("src.commands.translate.translate_text") as mock_translate, \
                 patch("src.commands.translate.compose_sequence") as mock_compose:
                mock_get_store.return_value = mock_store
                mock_get_signs_dir.return_value = Path(tmpdir)

                high_coverage_result = MagicMock()
                high_coverage_result.glosses = ["HELLO"]
                high_coverage_result.validation = MagicMock()
                high_coverage_result.validation.coverage = 1.0
                high_coverage_result.validation.missing = []
                mock_translate.return_value = high_coverage_result

                mock_store.get_verified_sign.return_value = mock_sign
                mock_compose.return_value = str(output_path)

                result = runner.invoke(
                    app,
                    ["translate", "Hello", "-o", str(output_path), "--speed", "fast"],
                )

                assert result.exit_code == 0
                call_kwargs = mock_compose.call_args[1]
                assert call_kwargs["transition_duration_ms"] == 100

    def test_translate_with_format_option(self, mock_store, mock_sign):
        """Test --format option for output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.webm"

            with patch("src.commands.translate.get_store") as mock_get_store, \
                 patch("src.commands.translate.get_signs_dir") as mock_get_signs_dir, \
                 patch("src.commands.translate.translate_text") as mock_translate, \
                 patch("src.commands.translate.compose_sequence") as mock_compose:
                mock_get_store.return_value = mock_store
                mock_get_signs_dir.return_value = Path(tmpdir)

                high_coverage_result = MagicMock()
                high_coverage_result.glosses = ["HELLO"]
                high_coverage_result.validation = MagicMock()
                high_coverage_result.validation.coverage = 1.0
                high_coverage_result.validation.missing = []
                mock_translate.return_value = high_coverage_result

                mock_store.get_verified_sign.return_value = mock_sign
                mock_compose.return_value = str(output_path)

                result = runner.invoke(
                    app,
                    ["translate", "Hello", "-o", str(output_path), "--format", "webm"],
                )

                assert result.exit_code == 0
                assert "Saved to" in result.output

    def test_translate_low_coverage_warning(self, mock_store, mock_sign):
        """Test warning when coverage is below 50%."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            with patch("src.commands.translate.get_store") as mock_get_store, \
                 patch("src.commands.translate.get_signs_dir") as mock_get_signs_dir, \
                 patch("src.commands.translate.translate_text") as mock_translate, \
                 patch("src.commands.translate.compose_sequence") as mock_compose:
                mock_get_store.return_value = mock_store
                mock_get_signs_dir.return_value = Path(tmpdir)

                low_coverage_result = MagicMock()
                low_coverage_result.glosses = ["HELLO", "UNKNOWN", "WORD"]
                low_coverage_result.validation = MagicMock()
                low_coverage_result.validation.coverage = 0.33
                low_coverage_result.validation.missing = ["UNKNOWN", "WORD"]
                mock_translate.return_value = low_coverage_result

                mock_store.get_verified_sign.side_effect = lambda g: mock_sign if g == "HELLO" else None
                mock_compose.return_value = str(output_path)

                # Confirm to continue
                result = runner.invoke(
                    app,
                    ["translate", "Hello unknown word", "-o", str(output_path)],
                    input="y\n",
                )

                assert result.exit_code == 0
                assert "Less than 50%" in result.output or "Warning" in result.output

    def test_translate_no_available_signs(self, mock_store):
        """Test error when no signs are available for translation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            with patch("src.commands.translate.get_store") as mock_get_store, \
                 patch("src.commands.translate.get_signs_dir") as mock_get_signs_dir, \
                 patch("src.commands.translate.translate_text") as mock_translate:
                mock_get_store.return_value = mock_store
                mock_get_signs_dir.return_value = Path(tmpdir)

                no_signs_result = MagicMock()
                no_signs_result.glosses = ["UNKNOWN", "WORDS"]
                no_signs_result.validation = MagicMock()
                no_signs_result.validation.coverage = 0.0
                no_signs_result.validation.missing = ["UNKNOWN", "WORDS"]
                mock_translate.return_value = no_signs_result

                mock_store.get_verified_sign.return_value = None

                # Confirm low coverage warning
                result = runner.invoke(
                    app,
                    ["translate", "Unknown words", "-o", str(output_path)],
                    input="y\n",
                )

                assert result.exit_code == 1
                assert "No available signs" in result.output

    def test_translate_shows_coverage_percentage(self, mock_store, mock_translation_result):
        """Test that coverage percentage is displayed."""
        with patch("src.commands.translate.get_store") as mock_get_store, \
             patch("src.commands.translate.get_signs_dir") as mock_get_signs_dir, \
             patch("src.commands.translate.translate_text") as mock_translate:
            mock_get_store.return_value = mock_store
            mock_get_signs_dir.return_value = Path("/tmp/signs")
            mock_translate.return_value = mock_translation_result

            result = runner.invoke(
                app,
                ["translate", "Hello, how are you?"],
            )

            assert result.exit_code == 0
            assert "Coverage:" in result.output
            assert "75%" in result.output

    def test_translate_shows_missing_signs(self, mock_store, mock_translation_result):
        """Test that missing signs are displayed."""
        with patch("src.commands.translate.get_store") as mock_get_store, \
             patch("src.commands.translate.get_signs_dir") as mock_get_signs_dir, \
             patch("src.commands.translate.translate_text") as mock_translate:
            mock_get_store.return_value = mock_store
            mock_get_signs_dir.return_value = Path("/tmp/signs")
            mock_translate.return_value = mock_translation_result

            result = runner.invoke(
                app,
                ["translate", "Hello, how are you?"],
            )

            assert result.exit_code == 0
            assert "Missing signs:" in result.output
            assert "HOW" in result.output

    def test_translate_decline_low_coverage(self, mock_store, mock_sign):
        """Test declining to continue with low coverage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.mp4"

            with patch("src.commands.translate.get_store") as mock_get_store, \
                 patch("src.commands.translate.get_signs_dir") as mock_get_signs_dir, \
                 patch("src.commands.translate.translate_text") as mock_translate, \
                 patch("src.commands.translate.compose_sequence") as mock_compose:
                mock_get_store.return_value = mock_store
                mock_get_signs_dir.return_value = Path(tmpdir)

                low_coverage_result = MagicMock()
                low_coverage_result.glosses = ["HELLO", "UNKNOWN"]
                low_coverage_result.validation = MagicMock()
                low_coverage_result.validation.coverage = 0.33
                low_coverage_result.validation.missing = ["UNKNOWN"]
                mock_translate.return_value = low_coverage_result

                mock_store.get_verified_sign.side_effect = lambda g: mock_sign if g == "HELLO" else None

                # Decline to continue
                result = runner.invoke(
                    app,
                    ["translate", "Hello unknown", "-o", str(output_path)],
                    input="n\n",
                )

                assert result.exit_code == 0
                mock_compose.assert_not_called()


class TestGlossesCommand:
    """Tests for the glosses command."""

    def test_glosses_shows_output(self, mock_store, mock_translation_result):
        """Test that glosses are displayed."""
        with patch("src.commands.translate.get_store") as mock_get_store, \
             patch("src.commands.translate.translate_text") as mock_translate:
            mock_get_store.return_value = mock_store
            mock_translate.return_value = mock_translation_result

            result = runner.invoke(
                app,
                ["glosses", "Hello, how are you?"],
            )

            assert result.exit_code == 0
            assert "HELLO" in result.output
            assert "HOW" in result.output
            assert "YOU" in result.output

    def test_glosses_with_validation(self, mock_store, mock_translation_result):
        """Test that missing signs are shown with validation."""
        with patch("src.commands.translate.get_store") as mock_get_store, \
             patch("src.commands.translate.translate_text") as mock_translate:
            mock_get_store.return_value = mock_store
            mock_translate.return_value = mock_translation_result

            result = runner.invoke(
                app,
                ["glosses", "Hello, how are you?"],
            )

            assert result.exit_code == 0
            assert "Missing:" in result.output
            assert "HOW" in result.output

    def test_glosses_without_validation(self, mock_store):
        """Test that --no-validate skips database check."""
        with patch("src.commands.translate.get_store") as mock_get_store, \
             patch("src.commands.translate.translate_text") as mock_translate:
            mock_get_store.return_value = mock_store

            result_no_validation = MagicMock()
            result_no_validation.glosses = ["HELLO", "HOW", "YOU"]
            result_no_validation.validation = None
            mock_translate.return_value = result_no_validation

            result = runner.invoke(
                app,
                ["glosses", "Hello, how are you?", "--no-validate"],
            )

            assert result.exit_code == 0
            assert "HELLO" in result.output
            assert "Missing:" not in result.output

    def test_glosses_shows_fingerspelled(self, mock_store):
        """Test that fingerspelled words are shown."""
        with patch("src.commands.translate.get_store") as mock_get_store, \
             patch("src.commands.translate.translate_text") as mock_translate:
            mock_get_store.return_value = mock_store

            result_with_fingerspell = MagicMock()
            result_with_fingerspell.glosses = ["HELLO", "J-O-H-N"]
            result_with_fingerspell.validation = MagicMock()
            result_with_fingerspell.validation.coverage = 0.5
            result_with_fingerspell.validation.missing = []
            result_with_fingerspell.validation.fingerspelled = ["J-O-H-N"]
            mock_translate.return_value = result_with_fingerspell

            result = runner.invoke(
                app,
                ["glosses", "Hello John"],
            )

            assert result.exit_code == 0
            assert "Fingerspelled:" in result.output
            assert "J-O-H-N" in result.output

    def test_glosses_output_format(self, mock_store):
        """Test that glosses are output in space-separated format."""
        with patch("src.commands.translate.get_store") as mock_get_store, \
             patch("src.commands.translate.translate_text") as mock_translate:
            mock_get_store.return_value = mock_store

            simple_result = MagicMock()
            simple_result.glosses = ["I", "HAPPY"]
            simple_result.validation = MagicMock()
            simple_result.validation.coverage = 1.0
            simple_result.validation.missing = []
            simple_result.validation.fingerspelled = []
            mock_translate.return_value = simple_result

            result = runner.invoke(
                app,
                ["glosses", "I am happy"],
            )

            assert result.exit_code == 0
            # Check that glosses appear together (space-separated)
            assert "I HAPPY" in result.output
