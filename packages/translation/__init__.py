# Translation package - English to ASL gloss conversion
"""
Convert English text to ASL gloss sequences using research-backed grammar rules.

Example usage:
    from packages.translation import translate, GlossConverter

    # Quick translation
    result = translate("I am happy")
    print(result.to_string())  # "I HAPPY"

    # With database validation
    from packages.database import SignStore
    store = SignStore("data/signs")
    result = translate("I am happy", store=store)
    print(f"Coverage: {result.validation.coverage}")
"""

from .gloss_converter import (
    GlossConverter,
    GlossSequence,
    TranslationQuality,
    translate,
)
from .grammar_rules import (
    apply_all_rules,
    detect_non_manual_markers,
    NonManualMarker,
    Token,
    TokenType,
    rule_remove_auxiliaries,
    rule_simplify_verbs,
    rule_time_topic_comment,
    rule_question_wh_at_end,
    rule_negation_at_end,
    rule_remove_articles,
    rule_minimal_be_verbs,
)
from .validator import (
    GlossValidator,
    ValidationMode,
    ValidationResult,
    FingerspellingHandler,
)

__all__ = [
    # Main API
    "translate",
    "GlossConverter",
    "GlossSequence",
    "TranslationQuality",
    # Grammar
    "apply_all_rules",
    "detect_non_manual_markers",
    "NonManualMarker",
    "Token",
    "TokenType",
    # Validation
    "GlossValidator",
    "ValidationMode",
    "ValidationResult",
    "FingerspellingHandler",
]
