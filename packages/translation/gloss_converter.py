"""Main translation pipeline: English to ASL gloss conversion.

Combines grammar rules, validation, and confidence scoring to produce
ASL gloss sequences from English text.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .grammar_rules import (
    apply_all_rules,
    detect_non_manual_markers,
    NonManualMarker,
)
from .validator import (
    GlossValidator,
    ValidationMode,
    ValidationResult,
    FingerspellingHandler,
)


class TranslationQuality(Enum):
    """Quality level of the translation."""
    HIGH = "high"        # All glosses available, high confidence
    MEDIUM = "medium"    # Most glosses available, some substitutions
    LOW = "low"          # Many missing glosses, heavy fingerspelling
    INCOMPLETE = "incomplete"  # Cannot produce usable translation


@dataclass
class GlossSequence:
    """Result of translating English to ASL glosses."""
    glosses: list[str] = field(default_factory=list)
    original_text: str = ""
    confidence: float = 0.0
    quality: TranslationQuality = TranslationQuality.INCOMPLETE
    non_manual_markers: list[NonManualMarker] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    fingerspelled: list[str] = field(default_factory=list)  # Words that were fingerspelled

    def to_string(self, separator: str = " ") -> str:
        """Convert gloss sequence to string.

        Args:
            separator: String to join glosses with

        Returns:
            Gloss string (e.g., "YESTERDAY STORE I GO")
        """
        return separator.join(self.glosses)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "glosses": self.glosses,
            "gloss_string": self.to_string(),
            "original_text": self.original_text,
            "confidence": self.confidence,
            "quality": self.quality.value,
            "coverage": self.validation.coverage if self.validation else 1.0,
            "missing_glosses": self.validation.missing_glosses if self.validation else [],
            "fingerspelled": self.fingerspelled,
            "non_manual_markers": [
                {
                    "type": nm.marker_type,
                    "start": nm.start_gloss_index,
                    "end": nm.end_gloss_index,
                    "description": nm.description,
                }
                for nm in self.non_manual_markers
            ],
        }


class GlossConverter:
    """Converts English text to ASL gloss sequences."""

    def __init__(
        self,
        validator: Optional[GlossValidator] = None,
        validation_mode: ValidationMode = ValidationMode.STRICT,
        allow_fingerspelling: bool = True,
    ):
        """Initialize the converter.

        Args:
            validator: GlossValidator instance for checking sign availability
            validation_mode: How strictly to validate glosses
            allow_fingerspelling: Whether to use fingerspelling for missing signs
        """
        self.validator = validator or GlossValidator()
        self.validation_mode = validation_mode
        self.allow_fingerspelling = allow_fingerspelling

    def translate(self, english: str) -> GlossSequence:
        """Translate English text to ASL gloss sequence.

        Args:
            english: English sentence to translate

        Returns:
            GlossSequence with glosses, confidence, and metadata
        """
        if not english or not english.strip():
            return GlossSequence(
                original_text=english,
                quality=TranslationQuality.INCOMPLETE,
            )

        # Apply grammar rules to get initial glosses
        raw_glosses = apply_all_rules(english)

        # Detect non-manual markers
        non_manual = detect_non_manual_markers(english, raw_glosses)

        # Validate glosses against database
        validation = self.validator.validate(raw_glosses, self.validation_mode)

        # Build final gloss sequence
        final_glosses = []
        fingerspelled = []

        for gloss in raw_glosses:
            gloss_upper = gloss.upper()

            if gloss_upper in [g.upper() for g in validation.valid_glosses]:
                final_glosses.append(gloss_upper)
            elif gloss_upper in validation.fallback_glosses:
                # Use substitution
                sub = validation.fallback_glosses[gloss_upper]
                final_glosses.extend(sub.split())
            elif self.allow_fingerspelling:
                # Fingerspell missing gloss
                fs_glosses = FingerspellingHandler.to_fingerspelling(gloss_upper)
                final_glosses.extend(fs_glosses)
                fingerspelled.append(gloss_upper)
            else:
                # Skip missing gloss
                pass

        # Calculate confidence
        confidence = self._calculate_confidence(raw_glosses, validation, fingerspelled)

        # Determine quality level
        quality = self._determine_quality(validation, fingerspelled, raw_glosses)

        return GlossSequence(
            glosses=final_glosses,
            original_text=english,
            confidence=confidence,
            quality=quality,
            non_manual_markers=non_manual,
            validation=validation,
            fingerspelled=fingerspelled,
        )

    def translate_batch(self, sentences: list[str]) -> list[GlossSequence]:
        """Translate multiple sentences.

        Args:
            sentences: List of English sentences

        Returns:
            List of GlossSequence results
        """
        return [self.translate(s) for s in sentences]

    def _calculate_confidence(
        self,
        raw_glosses: list[str],
        validation: ValidationResult,
        fingerspelled: list[str],
    ) -> float:
        """Calculate confidence score for the translation.

        Factors:
        - Coverage: How many glosses are available
        - Substitutions: Penalty for using fallbacks
        - Fingerspelling: Penalty for excessive fingerspelling

        Args:
            raw_glosses: Original gloss sequence before validation
            validation: Validation result
            fingerspelled: List of fingerspelled words

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not raw_glosses:
            return 0.0

        # Base confidence from coverage
        base_confidence = validation.coverage

        # Penalty for substitutions (each substitution reduces confidence slightly)
        sub_penalty = len(validation.fallback_glosses) * 0.05
        base_confidence -= sub_penalty

        # Penalty for fingerspelling (each fingerspelled word reduces confidence)
        fs_penalty = len(fingerspelled) * 0.1
        base_confidence -= fs_penalty

        # Clamp to valid range
        return max(0.0, min(1.0, base_confidence))

    def _determine_quality(
        self,
        validation: ValidationResult,
        fingerspelled: list[str],
        raw_glosses: list[str],
    ) -> TranslationQuality:
        """Determine translation quality level.

        Args:
            validation: Validation result
            fingerspelled: Fingerspelled words
            raw_glosses: Original glosses

        Returns:
            Quality level
        """
        if not raw_glosses:
            return TranslationQuality.INCOMPLETE

        coverage = validation.coverage
        fs_ratio = len(fingerspelled) / len(raw_glosses) if raw_glosses else 0

        if coverage >= 0.9 and fs_ratio <= 0.1:
            return TranslationQuality.HIGH
        elif coverage >= 0.7 and fs_ratio <= 0.3:
            return TranslationQuality.MEDIUM
        elif coverage >= 0.5:
            return TranslationQuality.LOW
        else:
            return TranslationQuality.INCOMPLETE


def translate(english: str, store=None) -> GlossSequence:
    """Convenience function for quick translation.

    Args:
        english: English text to translate
        store: Optional sign store for validation

    Returns:
        GlossSequence result
    """
    validator = GlossValidator(store)
    converter = GlossConverter(validator)
    return converter.translate(english)
