"""Validate glosses against the sign database.

Ensures that generated gloss sequences can be rendered with available signs.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol


class SignLookup(Protocol):
    """Protocol for sign database lookup."""

    def get_sign(self, gloss: str) -> Optional[object]:
        """Get a sign by gloss from any status."""
        ...

    def get_verified_sign(self, gloss: str) -> Optional[object]:
        """Get a verified sign only."""
        ...


class ValidationMode(Enum):
    """How strictly to validate glosses."""
    STRICT = "strict"          # Only verified signs
    PERMISSIVE = "permissive"  # Any sign (including pending/imported)
    REPORT_ONLY = "report_only"  # Don't filter, just report missing


@dataclass
class ValidationResult:
    """Result of validating a gloss sequence."""
    valid_glosses: list[str] = field(default_factory=list)
    missing_glosses: list[str] = field(default_factory=list)
    fallback_glosses: dict[str, str] = field(default_factory=dict)  # original -> fallback
    coverage: float = 0.0  # Percentage of glosses that are available

    @property
    def is_complete(self) -> bool:
        """True if all glosses are available."""
        return len(self.missing_glosses) == 0

    @property
    def is_partial(self) -> bool:
        """True if some but not all glosses are available."""
        return len(self.valid_glosses) > 0 and len(self.missing_glosses) > 0


class GlossValidator:
    """Validates gloss sequences against the sign database."""

    # Common fingerspelling fallbacks
    FINGERSPELL_THRESHOLD = 4  # Words shorter than this might fingerspell

    # Common sign substitutions (gloss -> alternative glosses to try)
    SUBSTITUTIONS = {
        "AM": ["ME"],
        "I": ["ME"],
        "MY": ["ME POSS"],
        "YOUR": ["YOU POSS"],
        "HIS": ["HE POSS"],
        "HER": ["SHE POSS"],
        "THEIR": ["THEY POSS"],
        "OUR": ["WE POSS"],
        "GOING": ["GO"],
        "WENT": ["GO"],
        "GOES": ["GO"],
        "DOING": ["DO"],
        "DID": ["DO"],
        "DOES": ["DO"],
        "SAID": ["SAY"],
        "SAYS": ["SAY"],
        "SAYING": ["SAY"],
        "HAVE": ["HAVE"],
        "HAS": ["HAVE"],
        "HAD": ["HAVE"],
        "HAVING": ["HAVE"],
        "WOULD": ["WILL"],
        "COULD": ["CAN"],
        "SHOULD": ["MUST"],
    }

    def __init__(self, store: Optional[SignLookup] = None):
        """Initialize validator.

        Args:
            store: Sign database store. If None, validation is disabled.
        """
        self.store = store

    def validate(
        self,
        glosses: list[str],
        mode: ValidationMode = ValidationMode.STRICT,
    ) -> ValidationResult:
        """Validate a sequence of glosses.

        Args:
            glosses: List of glosses to validate
            mode: Validation strictness

        Returns:
            ValidationResult with valid/missing glosses and coverage
        """
        if self.store is None:
            # No database connection - everything is valid
            return ValidationResult(
                valid_glosses=glosses.copy(),
                coverage=1.0,
            )

        result = ValidationResult()

        for gloss in glosses:
            gloss_upper = gloss.upper()

            # Check if gloss exists
            sign = self._lookup_sign(gloss_upper, mode)

            if sign is not None:
                result.valid_glosses.append(gloss_upper)
            else:
                # Try substitutions
                found_sub = False
                if gloss_upper in self.SUBSTITUTIONS:
                    for sub in self.SUBSTITUTIONS[gloss_upper]:
                        # Check if substitution is a single gloss or compound
                        sub_parts = sub.split()
                        all_valid = all(
                            self._lookup_sign(part, mode) is not None
                            for part in sub_parts
                        )
                        if all_valid:
                            result.valid_glosses.extend(sub_parts)
                            result.fallback_glosses[gloss_upper] = sub
                            found_sub = True
                            break

                if not found_sub:
                    result.missing_glosses.append(gloss_upper)

        # Calculate coverage
        total = len(result.valid_glosses) + len(result.missing_glosses)
        if total > 0:
            result.coverage = len(result.valid_glosses) / total
        else:
            result.coverage = 1.0

        return result

    def _lookup_sign(self, gloss: str, mode: ValidationMode) -> Optional[object]:
        """Look up a sign based on validation mode."""
        if self.store is None:
            return None

        if mode == ValidationMode.STRICT:
            return self.store.get_verified_sign(gloss)
        else:
            return self.store.get_sign(gloss)

    def suggest_fingerspelling(self, gloss: str) -> list[str]:
        """Suggest fingerspelling for a missing gloss.

        For short words or proper nouns, fingerspelling is common in ASL.

        Args:
            gloss: The missing gloss

        Returns:
            List of letter glosses for fingerspelling (e.g., ["J", "O", "H", "N"])
        """
        # Convert to individual letters
        letters = list(gloss.upper())
        return [f"FS-{letter}" for letter in letters if letter.isalpha()]

    def get_available_glosses(self) -> set[str]:
        """Get all available glosses in the database.

        Returns:
            Set of available gloss strings
        """
        if self.store is None:
            return set()

        # This requires a list_signs method on the store
        available = set()
        if hasattr(self.store, "list_verified"):
            for sign in self.store.list_verified():
                available.add(sign.gloss.upper())
        return available


class FingerspellingHandler:
    """Handles fingerspelling for proper nouns and unknown words."""

    # Letters that have ASL signs
    ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    @classmethod
    def to_fingerspelling(cls, word: str) -> list[str]:
        """Convert a word to fingerspelling glosses.

        Args:
            word: Word to fingerspell

        Returns:
            List of letter glosses
        """
        result = []
        for char in word.upper():
            if char in cls.ALPHABET:
                result.append(f"FS-{char}")
        return result

    @classmethod
    def should_fingerspell(cls, word: str, available_glosses: set[str]) -> bool:
        """Determine if a word should be fingerspelled.

        Fingerspelling is used for:
        - Proper nouns (names, places)
        - Technical terms
        - Words without established signs
        - Short words (often fingerspelled rather than signed)

        Args:
            word: Word to check
            available_glosses: Set of available sign glosses

        Returns:
            True if word should be fingerspelled
        """
        word_upper = word.upper()

        # If there's a sign for it, don't fingerspell
        if word_upper in available_glosses:
            return False

        # Very short words (1-2 letters) often fingerspelled
        if len(word) <= 2:
            return True

        # Check if it looks like a proper noun (starts with capital in original)
        # This is a heuristic - in real usage, NER would be better
        if word[0].isupper() and word_upper not in available_glosses:
            return True

        return False
