"""Tests for gloss validation against sign database."""

import pytest
from dataclasses import dataclass
from typing import Optional

from ..validator import (
    GlossValidator,
    ValidationMode,
    ValidationResult,
    FingerspellingHandler,
)


@dataclass
class MockSign:
    """Mock sign for testing."""
    gloss: str


class MockSignStore:
    """Mock sign store for testing validation."""

    def __init__(self, verified: set[str], pending: set[str] = None):
        self.verified = {g.upper() for g in verified}
        self.pending = {g.upper() for g in (pending or set())}

    def get_verified_sign(self, gloss: str) -> Optional[MockSign]:
        if gloss.upper() in self.verified:
            return MockSign(gloss=gloss.upper())
        return None

    def get_sign(self, gloss: str) -> Optional[MockSign]:
        gloss_upper = gloss.upper()
        if gloss_upper in self.verified or gloss_upper in self.pending:
            return MockSign(gloss=gloss_upper)
        return None

    def list_verified(self):
        return [MockSign(g) for g in self.verified]


class TestGlossValidator:
    """Test GlossValidator class."""

    def test_all_valid_glosses(self):
        store = MockSignStore(verified={"HELLO", "WORLD"})
        validator = GlossValidator(store)

        result = validator.validate(["HELLO", "WORLD"])

        assert result.is_complete
        assert result.coverage == 1.0
        assert result.valid_glosses == ["HELLO", "WORLD"]
        assert result.missing_glosses == []

    def test_some_missing_glosses(self):
        store = MockSignStore(verified={"HELLO"})
        validator = GlossValidator(store)

        result = validator.validate(["HELLO", "WORLD"])

        assert result.is_partial
        assert result.coverage == 0.5
        assert "HELLO" in result.valid_glosses
        assert "WORLD" in result.missing_glosses

    def test_all_missing_glosses(self):
        store = MockSignStore(verified=set())
        validator = GlossValidator(store)

        result = validator.validate(["HELLO", "WORLD"])

        assert not result.is_complete
        assert result.coverage == 0.0
        assert result.missing_glosses == ["HELLO", "WORLD"]

    def test_strict_mode_verified_only(self):
        store = MockSignStore(verified={"HELLO"}, pending={"WORLD"})
        validator = GlossValidator(store)

        result = validator.validate(
            ["HELLO", "WORLD"],
            mode=ValidationMode.STRICT,
        )

        assert "HELLO" in result.valid_glosses
        assert "WORLD" in result.missing_glosses

    def test_permissive_mode_includes_pending(self):
        store = MockSignStore(verified={"HELLO"}, pending={"WORLD"})
        validator = GlossValidator(store)

        result = validator.validate(
            ["HELLO", "WORLD"],
            mode=ValidationMode.PERMISSIVE,
        )

        assert "HELLO" in result.valid_glosses
        assert "WORLD" in result.valid_glosses
        assert result.is_complete

    def test_case_insensitive(self):
        store = MockSignStore(verified={"HELLO"})
        validator = GlossValidator(store)

        result = validator.validate(["hello", "Hello", "HELLO"])

        assert result.coverage == 1.0
        assert all(g == "HELLO" for g in result.valid_glosses)

    def test_no_store_all_valid(self):
        validator = GlossValidator(store=None)

        result = validator.validate(["ANYTHING", "GOES"])

        assert result.is_complete
        assert result.coverage == 1.0

    def test_empty_glosses(self):
        store = MockSignStore(verified={"HELLO"})
        validator = GlossValidator(store)

        result = validator.validate([])

        assert result.is_complete
        assert result.coverage == 1.0

    def test_substitution_used(self):
        # Store has HAVE but not HAS
        store = MockSignStore(verified={"HAVE", "I"})
        validator = GlossValidator(store)

        result = validator.validate(["I", "HAS"])

        # HAS should be substituted with HAVE
        assert "HAVE" in result.valid_glosses
        assert "HAS" not in result.valid_glosses
        assert result.fallback_glosses.get("HAS") == "HAVE"


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_is_complete_true(self):
        result = ValidationResult(
            valid_glosses=["A", "B"],
            missing_glosses=[],
            coverage=1.0,
        )
        assert result.is_complete
        assert not result.is_partial

    def test_is_partial_true(self):
        result = ValidationResult(
            valid_glosses=["A"],
            missing_glosses=["B"],
            coverage=0.5,
        )
        assert not result.is_complete
        assert result.is_partial

    def test_neither_complete_nor_partial(self):
        result = ValidationResult(
            valid_glosses=[],
            missing_glosses=["A", "B"],
            coverage=0.0,
        )
        assert not result.is_complete
        assert not result.is_partial


class TestFingerspellingHandler:
    """Test FingerspellingHandler class."""

    def test_to_fingerspelling(self):
        result = FingerspellingHandler.to_fingerspelling("JOHN")

        assert result == ["FS-J", "FS-O", "FS-H", "FS-N"]

    def test_to_fingerspelling_lowercase(self):
        result = FingerspellingHandler.to_fingerspelling("john")

        assert result == ["FS-J", "FS-O", "FS-H", "FS-N"]

    def test_to_fingerspelling_ignores_non_alpha(self):
        result = FingerspellingHandler.to_fingerspelling("A-1")

        assert result == ["FS-A"]

    def test_should_fingerspell_short_word(self):
        available = {"HELLO", "WORLD"}

        assert FingerspellingHandler.should_fingerspell("IT", available)
        assert FingerspellingHandler.should_fingerspell("I", available)

    def test_should_not_fingerspell_available(self):
        available = {"HELLO", "WORLD"}

        assert not FingerspellingHandler.should_fingerspell("HELLO", available)

    def test_should_fingerspell_proper_noun(self):
        available = {"HELLO", "WORLD"}

        # Proper nouns (capital first letter, not in db) should fingerspell
        assert FingerspellingHandler.should_fingerspell("John", available)


class TestGlossValidatorSuggestFingerspelling:
    """Test fingerspelling suggestions."""

    def test_suggest_fingerspelling(self):
        validator = GlossValidator()

        result = validator.suggest_fingerspelling("JOHN")

        assert result == ["FS-J", "FS-O", "FS-H", "FS-N"]


class TestGetAvailableGlosses:
    """Test getting available glosses from store."""

    def test_get_available_glosses(self):
        store = MockSignStore(verified={"HELLO", "WORLD", "CAT"})
        validator = GlossValidator(store)

        available = validator.get_available_glosses()

        assert available == {"HELLO", "WORLD", "CAT"}

    def test_get_available_glosses_no_store(self):
        validator = GlossValidator(store=None)

        available = validator.get_available_glosses()

        assert available == set()
