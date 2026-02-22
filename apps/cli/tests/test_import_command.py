"""Tests for the import command."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.main import app
from packages.database import Sign, SignStatus, VideoInfo


runner = CliRunner()


class TestImportCommand:
    """Tests for the import command."""

    def test_import_path_not_found(self, mock_store):
        """Test error when source path doesn't exist."""
        with patch("src.commands.import_cmd.get_store") as mock_get_store:
            mock_get_store.return_value = mock_store

            result = runner.invoke(
                app,
                ["import", "wlasl", "--source", "/nonexistent/path"],
            )

            assert result.exit_code == 1
            assert "Source path not found" in result.output

    def test_import_invalid_source(self, mock_store):
        """Test error for unknown source type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.side_effect = ValueError("Unknown source type: invalid")

                result = runner.invoke(
                    app,
                    ["import", "invalid", "--source", tmpdir],
                )

                assert result.exit_code == 1
                assert "Error" in result.output

    def test_import_single_gloss(self, mock_store, mock_importer, mock_imported_sign):
        """Test importing a single sign with --gloss option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer
                mock_importer.import_sign.return_value = mock_imported_sign

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir, "--gloss", "THANK-YOU"],
                )

                assert result.exit_code == 0
                assert "Imported THANK-YOU" in result.output
                mock_importer.import_sign.assert_called_once_with("THANK-YOU")

    def test_import_single_gloss_not_found(self, mock_store, mock_importer):
        """Test error when single gloss is not found in source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer
                mock_importer.import_sign.return_value = None

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir, "--gloss", "NONEXISTENT"],
                )

                assert result.exit_code == 0
                assert "not found" in result.output

    def test_import_dry_run(self, mock_store, mock_importer):
        """Test --dry-run shows what would be imported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer
                mock_importer.list_available.return_value = [
                    "HELLO", "WORLD", "THANK-YOU", "GOODBYE", "PLEASE"
                ]

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir, "--dry-run"],
                )

                assert result.exit_code == 0
                assert "5 signs available" in result.output
                assert "HELLO" in result.output
                mock_importer.import_all.assert_not_called()

    def test_import_dry_run_single_gloss(self, mock_store, mock_importer):
        """Test --dry-run with --gloss shows single sign."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir, "--gloss", "HELLO", "--dry-run"],
                )

                assert result.exit_code == 0
                assert "Would import HELLO" in result.output
                mock_importer.import_sign.assert_not_called()

    def test_import_with_limit(self, mock_store, mock_importer):
        """Test --limit option restricts number of imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer
                mock_importer.import_all.return_value = (10, [])

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir, "--limit", "10"],
                )

                assert result.exit_code == 0
                mock_importer.import_all.assert_called_once_with(limit=10)

    def test_import_full_success(self, mock_store, mock_importer):
        """Test full import with progress shows success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer
                mock_importer.import_all.return_value = (25, [])

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir],
                )

                assert result.exit_code == 0
                assert "Successfully imported 25 signs" in result.output
                mock_importer.import_all.assert_called_once_with(limit=None)

    def test_import_with_errors(self, mock_store, mock_importer):
        """Test import shows error summary when errors occur."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer
                errors = [
                    "Failed to import HELLO: video corrupt",
                    "Failed to import WORLD: missing metadata",
                    "Failed to import GOODBYE: invalid format",
                ]
                mock_importer.import_all.return_value = (22, errors)

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir],
                )

                assert result.exit_code == 0
                assert "Successfully imported 22 signs" in result.output
                assert "3 errors occurred" in result.output
                assert "Failed to import HELLO" in result.output

    def test_import_shows_next_steps(self, mock_store, mock_importer):
        """Test that import shows next steps after completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer
                mock_importer.import_all.return_value = (5, [])

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir],
                )

                assert result.exit_code == 0
                assert "verify --interactive" in result.output

    def test_import_dry_run_with_limit(self, mock_store, mock_importer):
        """Test --dry-run with --limit shows limited signs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer
                mock_importer.list_available.return_value = [
                    "HELLO", "WORLD", "THANK-YOU", "GOODBYE", "PLEASE",
                    "SORRY", "HELP", "YES", "NO", "MAYBE"
                ]

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir, "--dry-run", "--limit", "3"],
                )

                assert result.exit_code == 0
                assert "3 signs available" in result.output

    def test_import_how2sign_source(self, mock_store, mock_importer):
        """Test importing from how2sign source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer
                mock_importer.import_all.return_value = (15, [])

                result = runner.invoke(
                    app,
                    ["import", "how2sign", "--source", tmpdir],
                )

                assert result.exit_code == 0
                assert "Successfully imported 15 signs" in result.output
                mock_create.assert_called_once()
                call_args = mock_create.call_args[0]
                assert call_args[0] == "how2sign"

    def test_import_truncates_many_errors(self, mock_store, mock_importer):
        """Test that many errors are truncated in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer
                # Generate 15 errors
                errors = [f"Failed to import SIGN{i}: error" for i in range(15)]
                mock_importer.import_all.return_value = (10, errors)

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir],
                )

                assert result.exit_code == 0
                assert "15 errors occurred" in result.output
                assert "and 5 more" in result.output

    def test_import_normalizes_gloss(self, mock_store, mock_importer, mock_imported_sign):
        """Test that gloss is normalized for import message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer
                mock_importer.import_sign.return_value = mock_imported_sign

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir, "--gloss", "hello"],
                )

                assert result.exit_code == 0
                assert "Importing HELLO" in result.output

    def test_import_dry_run_no_list_available(self, mock_store, mock_importer):
        """Test dry run when importer doesn't have list_available method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                # Remove list_available method
                del mock_importer.list_available
                mock_create.return_value = mock_importer

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir, "--dry-run"],
                )

                assert result.exit_code == 0
                assert "would import" in result.output.lower()

    def test_import_many_available_signs_truncates(self, mock_store, mock_importer):
        """Test that many available signs are truncated in dry run output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.commands.import_cmd.get_store") as mock_get_store, \
                 patch("src.commands.import_cmd.create_importer") as mock_create:
                mock_get_store.return_value = mock_store
                mock_create.return_value = mock_importer
                # Generate 100 signs
                mock_importer.list_available.return_value = [f"SIGN{i}" for i in range(100)]

                result = runner.invoke(
                    app,
                    ["import", "wlasl", "--source", tmpdir, "--dry-run"],
                )

                assert result.exit_code == 0
                assert "100 signs available" in result.output
                assert "and 50 more" in result.output
