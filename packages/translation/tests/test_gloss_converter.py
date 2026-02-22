"""Tests for the main gloss converter pipeline."""

import pytest
from dataclasses import dataclass
from typing import Optional

from ..gloss_converter import (
    GlossConverter,
    GlossSequence,
    TranslationQuality,
    translate,
)
from ..validator import GlossValidator, ValidationMode


@dataclass
class MockSign:
    """Mock sign for testing."""
    gloss: str


class MockSignStore:
    """Mock sign store for testing."""

    def __init__(self, glosses: set[str]):
        self.glosses = {g.upper() for g in glosses}

    def get_verified_sign(self, gloss: str) -> Optional[MockSign]:
        if gloss.upper() in self.glosses:
            return MockSign(gloss=gloss.upper())
        return None

    def get_sign(self, gloss: str) -> Optional[MockSign]:
        return self.get_verified_sign(gloss)

    def list_verified(self):
        return [MockSign(g) for g in self.glosses]


class TestGlossConverter:
    """Test GlossConverter class."""

    def test_basic_translation(self):
        converter = GlossConverter()
        result = converter.translate("I am happy")

        assert isinstance(result, GlossSequence)
        assert "HAPPY" in result.glosses
        assert "AM" not in result.glosses

    def test_translation_with_validation(self):
        store = MockSignStore(glosses={"I", "HAPPY"})
        validator = GlossValidator(store)
        converter = GlossConverter(validator=validator)

        result = converter.translate("I am happy")

        assert result.validation is not None
        assert result.validation.coverage == 1.0

    def test_translation_with_missing_signs(self):
        store = MockSignStore(glosses={"I"})
        validator = GlossValidator(store)
        converter = GlossConverter(validator=validator, allow_fingerspelling=True)

        result = converter.translate("I am happy")

        # "HAPPY" should be fingerspelled
        assert "HAPPY" in result.fingerspelled or any("FS-" in g for g in result.glosses)

    def test_translation_without_fingerspelling(self):
        store = MockSignStore(glosses={"I"})
        validator = GlossValidator(store)
        converter = GlossConverter(validator=validator, allow_fingerspelling=False)

        result = converter.translate("I am happy")

        # Without fingerspelling, missing glosses are skipped
        assert not any("FS-" in g for g in result.glosses)

    def test_time_topic_comment_applied(self):
        converter = GlossConverter()
        result = converter.translate("I went to the store yesterday")

        # Time marker should be first
        assert result.glosses[0] == "YESTERDAY"

    def test_question_transformation(self):
        converter = GlossConverter()
        result = converter.translate("What is your name")

        # WH-word should be at end
        assert result.glosses[-1] == "WHAT"

    def test_negation_transformation(self):
        converter = GlossConverter()
        result = converter.translate("I don't understand")

        # Negation at end
        assert result.glosses[-1] == "NOT"

    def test_non_manual_markers_detected(self):
        converter = GlossConverter()
        result = converter.translate("What is your name?")

        assert len(result.non_manual_markers) > 0
        assert any(m.marker_type == "wh_question" for m in result.non_manual_markers)

    def test_empty_input(self):
        converter = GlossConverter()
        result = converter.translate("")

        assert result.glosses == []
        assert result.quality == TranslationQuality.INCOMPLETE

    def test_whitespace_only_input(self):
        converter = GlossConverter()
        result = converter.translate("   ")

        assert result.glosses == []


class TestTranslationQuality:
    """Test quality determination."""

    def test_high_quality(self):
        # All glosses available
        store = MockSignStore(glosses={"I", "HAPPY"})
        validator = GlossValidator(store)
        converter = GlossConverter(validator=validator)

        result = converter.translate("I happy")

        assert result.quality == TranslationQuality.HIGH
        assert result.confidence >= 0.9

    def test_medium_quality(self):
        # Most glosses available with some substitutions
        store = MockSignStore(glosses={"I", "HAPPY", "TODAY"})
        validator = GlossValidator(store)
        converter = GlossConverter(validator=validator)

        # Some signs might need substitution but most are there
        result = converter.translate("I am happy today")

        assert result.quality in [TranslationQuality.HIGH, TranslationQuality.MEDIUM]

    def test_low_quality_heavy_fingerspelling(self):
        # Most glosses need fingerspelling
        store = MockSignStore(glosses={"I"})
        validator = GlossValidator(store)
        converter = GlossConverter(validator=validator)

        result = converter.translate("I visited California yesterday")

        # Heavy fingerspelling should result in low quality
        assert result.quality in [TranslationQuality.LOW, TranslationQuality.MEDIUM]


class TestGlossSequence:
    """Test GlossSequence dataclass methods."""

    def test_to_string(self):
        seq = GlossSequence(glosses=["HELLO", "WORLD"])

        assert seq.to_string() == "HELLO WORLD"

    def test_to_string_custom_separator(self):
        seq = GlossSequence(glosses=["HELLO", "WORLD"])

        assert seq.to_string("-") == "HELLO-WORLD"

    def test_to_dict(self):
        seq = GlossSequence(
            glosses=["HELLO", "WORLD"],
            original_text="Hello world",
            confidence=0.9,
            quality=TranslationQuality.HIGH,
        )

        d = seq.to_dict()

        assert d["glosses"] == ["HELLO", "WORLD"]
        assert d["gloss_string"] == "HELLO WORLD"
        assert d["original_text"] == "Hello world"
        assert d["confidence"] == 0.9
        assert d["quality"] == "high"


class TestBatchTranslation:
    """Test batch translation."""

    def test_translate_batch(self):
        converter = GlossConverter()
        sentences = [
            "I am happy",
            "You are sad",
            "What is your name",
        ]

        results = converter.translate_batch(sentences)

        assert len(results) == 3
        assert all(isinstance(r, GlossSequence) for r in results)

    def test_translate_batch_empty(self):
        converter = GlossConverter()

        results = converter.translate_batch([])

        assert results == []


class TestConvenienceFunction:
    """Test translate convenience function."""

    def test_translate_function(self):
        result = translate("I am happy")

        assert isinstance(result, GlossSequence)
        assert "HAPPY" in result.glosses

    def test_translate_function_with_store(self):
        store = MockSignStore(glosses={"I", "HAPPY"})

        result = translate("I am happy", store=store)

        assert result.validation is not None
        assert result.validation.coverage == 1.0


class TestRealWorldExamples:
    """Test real-world translation examples."""

    def test_greeting(self):
        result = translate("Hello, how are you?")

        assert "HELLO" in result.glosses
        assert "YOU" in result.glosses

    def test_introduction(self):
        result = translate("My name is John")

        # Should have name-related glosses, John might be fingerspelled
        assert "MY" in result.glosses or "NAME" in result.glosses

    def test_simple_statement(self):
        result = translate("The cat is sleeping")

        assert "CAT" in result.glosses
        assert "SLEEP" in result.glosses
        # No articles or be-verbs
        assert "THE" not in result.glosses
        assert "IS" not in result.glosses

    def test_past_tense(self):
        result = translate("Yesterday I walked to school")

        assert result.glosses[0] == "YESTERDAY"
        assert "WALK" in result.glosses  # Simplified from "walked"

    def test_negation_sentence(self):
        result = translate("I cannot help you")

        # Negation should be present
        assert "NOT" in result.glosses
        assert result.glosses[-1] == "NOT"

    def test_question_where(self):
        result = translate("Where is the bathroom")

        assert result.glosses[-1] == "WHERE"
        assert "BATHROOM" in result.glosses
