"""Tests for ASL grammar rules.

Each test case validates that grammar rules produce expected ASL gloss output.
Test cases are based on examples from linguistic literature.
"""

import pytest

from ..grammar_rules import (
    tokenize,
    Token,
    TokenType,
    rule_time_topic_comment,
    rule_remove_articles,
    rule_minimal_be_verbs,
    rule_question_wh_at_end,
    rule_negation_at_end,
    rule_simplify_verbs,
    rule_remove_auxiliaries,
    rule_convert_to_gloss,
    apply_all_rules,
    detect_non_manual_markers,
)


class TestTokenize:
    """Test tokenization and classification."""

    def test_simple_sentence(self):
        tokens = tokenize("I am happy")
        assert len(tokens) == 3
        assert tokens[0].text == "i"
        assert tokens[1].text == "am"
        assert tokens[1].token_type == TokenType.BE_VERB
        assert tokens[2].text == "happy"

    def test_time_marker_detection(self):
        tokens = tokenize("yesterday I went")
        assert tokens[0].token_type == TokenType.TIME
        assert tokens[0].text == "yesterday"

    def test_question_word_detection(self):
        tokens = tokenize("what is your name")
        assert tokens[0].token_type == TokenType.QUESTION_WORD
        assert tokens[0].text == "what"

    def test_article_detection(self):
        tokens = tokenize("the cat sat on a mat")
        article_tokens = [t for t in tokens if t.token_type == TokenType.ARTICLE]
        assert len(article_tokens) == 2

    def test_contraction_expansion(self):
        tokens = tokenize("I don't understand")
        # "don't" should expand to "do not"
        texts = [t.text for t in tokens]
        assert "not" in texts

    def test_negation_detection(self):
        tokens = tokenize("I do not like it")
        negation_tokens = [t for t in tokens if t.token_type == TokenType.NEGATION]
        assert len(negation_tokens) == 1


class TestTimeTopicComment:
    """Test Time-Topic-Comment rule.

    Source: Valli & Lucas (2000), Ch. 5
    """

    def test_time_moves_first(self):
        tokens = tokenize("I went to store yesterday")
        result = rule_time_topic_comment(tokens)

        # Time marker should be first
        assert result[0].text == "yesterday"

    def test_multiple_time_markers(self):
        tokens = tokenize("I always eat breakfast morning")
        result = rule_time_topic_comment(tokens)

        # Both time markers should be at front
        time_tokens = [t for t in result[:2] if t.token_type == TokenType.TIME]
        assert len(time_tokens) == 2

    def test_no_time_marker(self):
        tokens = tokenize("I like cats")
        result = rule_time_topic_comment(tokens)

        # Should remain unchanged
        assert [t.text for t in result] == [t.text for t in tokens]


class TestRemoveArticles:
    """Test article removal rule.

    Source: Valli & Lucas (2000), Ch. 4
    """

    def test_removes_the(self):
        tokens = tokenize("the cat")
        result = rule_remove_articles(tokens)

        texts = [t.text for t in result]
        assert "the" not in texts
        assert "cat" in texts

    def test_removes_a_an(self):
        tokens = tokenize("a dog and an apple")
        result = rule_remove_articles(tokens)

        texts = [t.text for t in result]
        assert "a" not in texts
        assert "an" not in texts


class TestMinimalBeVerbs:
    """Test be-verb removal rule.

    Source: Sandler & Lillo-Martin (2006), Ch. 8
    """

    def test_removes_is(self):
        tokens = tokenize("she is happy")
        result = rule_minimal_be_verbs(tokens)

        texts = [t.text for t in result]
        assert "is" not in texts
        assert "she" in texts
        assert "happy" in texts

    def test_removes_are(self):
        tokens = tokenize("they are tired")
        result = rule_minimal_be_verbs(tokens)

        texts = [t.text for t in result]
        assert "are" not in texts


class TestQuestionWhAtEnd:
    """Test WH-question rule.

    Source: Neidle et al. (2000), Ch. 3
    """

    def test_what_moves_to_end(self):
        tokens = tokenize("what is your name")
        tokens = rule_remove_articles(tokens)
        tokens = rule_minimal_be_verbs(tokens)
        result = rule_question_wh_at_end(tokens)

        # "what" should be last
        assert result[-1].text == "what"

    def test_where_moves_to_end(self):
        tokens = tokenize("where is the bathroom")
        tokens = rule_remove_articles(tokens)
        tokens = rule_minimal_be_verbs(tokens)
        result = rule_question_wh_at_end(tokens)

        assert result[-1].text == "where"

    def test_why_moves_to_end(self):
        tokens = tokenize("why are you sad")
        tokens = rule_minimal_be_verbs(tokens)
        result = rule_question_wh_at_end(tokens)

        assert result[-1].text == "why"


class TestNegationAtEnd:
    """Test negation rule.

    Source: Neidle et al. (2000), Ch. 4
    """

    def test_not_moves_to_end(self):
        tokens = tokenize("I do not understand")
        result = rule_negation_at_end(tokens)

        # Negation should be at end
        assert result[-1].token_type == TokenType.NEGATION

    def test_dont_becomes_not_at_end(self):
        tokens = tokenize("I don't like it")
        result = rule_negation_at_end(tokens)

        assert result[-1].token_type == TokenType.NEGATION
        assert result[-1].text == "not"


class TestSimplifyVerbs:
    """Test verb simplification."""

    def test_removes_ing(self):
        tokens = tokenize("I am running")
        tokens = [t for t in tokens if t.token_type == TokenType.WORD]
        result = rule_simplify_verbs(tokens)

        texts = [t.text for t in result]
        assert "run" in texts

    def test_removes_ed(self):
        tokens = tokenize("I walked home")
        tokens = [t for t in tokens if t.token_type == TokenType.WORD]
        result = rule_simplify_verbs(tokens)

        texts = [t.text for t in result]
        assert "walk" in texts

    def test_handles_doubled_consonant(self):
        tokens = tokenize("running")
        result = rule_simplify_verbs(tokens)

        assert result[0].text == "run"

    def test_irregular_verb_went(self):
        tokens = tokenize("I went home")
        tokens = [t for t in tokens if t.token_type == TokenType.WORD]
        result = rule_simplify_verbs(tokens)

        texts = [t.text for t in result]
        assert "go" in texts
        assert "went" not in texts

    def test_irregular_verb_saw(self):
        tokens = tokenize("I saw the movie")
        tokens = [t for t in tokens if t.token_type == TokenType.WORD]
        result = rule_simplify_verbs(tokens)

        texts = [t.text for t in result]
        assert "see" in texts

    def test_irregular_verb_ate(self):
        tokens = tokenize("She ate breakfast")
        tokens = [t for t in tokens if t.token_type == TokenType.WORD]
        result = rule_simplify_verbs(tokens)

        texts = [t.text for t in result]
        assert "eat" in texts

    def test_irregular_verb_told(self):
        tokens = tokenize("He told me")
        tokens = [t for t in tokens if t.token_type == TokenType.WORD]
        result = rule_simplify_verbs(tokens)

        texts = [t.text for t in result]
        assert "tell" in texts


class TestRemoveAuxiliaries:
    """Test auxiliary verb removal.

    Source: Valli & Lucas (2000), Ch. 6
    """

    def test_removes_do(self):
        tokens = tokenize("I do like pizza")
        result = rule_remove_auxiliaries(tokens)

        texts = [t.text for t in result]
        assert "do" not in texts
        assert "like" in texts

    def test_removes_did(self):
        tokens = tokenize("I did go yesterday")
        result = rule_remove_auxiliaries(tokens)

        texts = [t.text for t in result]
        assert "did" not in texts

    def test_removes_does(self):
        tokens = tokenize("She does like cats")
        result = rule_remove_auxiliaries(tokens)

        texts = [t.text for t in result]
        assert "does" not in texts

    def test_removes_will(self):
        tokens = tokenize("I will go")
        result = rule_remove_auxiliaries(tokens)

        texts = [t.text for t in result]
        assert "will" not in texts

    def test_removes_would(self):
        tokens = tokenize("I would like that")
        result = rule_remove_auxiliaries(tokens)

        texts = [t.text for t in result]
        assert "would" not in texts

    def test_removes_can(self):
        tokens = tokenize("I can help")
        result = rule_remove_auxiliaries(tokens)

        texts = [t.text for t in result]
        assert "can" not in texts


class TestApplyAllRules:
    """Test full translation pipeline."""

    def test_basic_sentence(self):
        # "I am happy" -> "I HAPPY" (remove be-verb)
        glosses = apply_all_rules("I am happy")
        assert "HAPPY" in glosses
        assert "AM" not in glosses

    def test_time_topic_comment_full(self):
        # "I went to the store yesterday" -> "YESTERDAY STORE I GO"
        glosses = apply_all_rules("I went to the store yesterday")

        assert glosses[0] == "YESTERDAY"
        assert "STORE" in glosses
        assert "THE" not in glosses

    def test_question_full(self):
        # "What is your name?" -> "YOUR NAME WHAT"
        glosses = apply_all_rules("What is your name")

        assert glosses[-1] == "WHAT"
        assert "IS" not in glosses

    def test_negation_full(self):
        # "I don't understand" -> "I UNDERSTAND NOT"
        glosses = apply_all_rules("I don't understand")

        assert glosses[-1] == "NOT"
        assert "UNDERSTAND" in glosses

    def test_complex_sentence(self):
        # "Yesterday I went to the big store"
        glosses = apply_all_rules("Yesterday I went to the big store")

        # Time first
        assert glosses[0] == "YESTERDAY"
        # No articles
        assert "THE" not in glosses


class TestNonManualMarkers:
    """Test detection of non-manual markers."""

    def test_wh_question_marker(self):
        glosses = apply_all_rules("What is your name")
        markers = detect_non_manual_markers("What is your name?", glosses)

        wh_markers = [m for m in markers if m.marker_type == "wh_question"]
        assert len(wh_markers) == 1

    def test_yes_no_question_marker(self):
        glosses = apply_all_rules("Do you like coffee")
        markers = detect_non_manual_markers("Do you like coffee?", glosses)

        yn_markers = [m for m in markers if m.marker_type == "yes_no_question"]
        assert len(yn_markers) == 1

    def test_negation_marker(self):
        glosses = apply_all_rules("I don't understand")
        markers = detect_non_manual_markers("I don't understand", glosses)

        neg_markers = [m for m in markers if m.marker_type == "negation"]
        assert len(neg_markers) == 1


class TestEdgeCases:
    """Test edge cases and special inputs."""

    def test_empty_string(self):
        glosses = apply_all_rules("")
        assert glosses == []

    def test_single_word(self):
        glosses = apply_all_rules("Hello")
        assert glosses == ["HELLO"]

    def test_punctuation_handling(self):
        glosses = apply_all_rules("Hello, world!")
        assert "HELLO" in glosses
        assert "WORLD" in glosses

    def test_numbers(self):
        glosses = apply_all_rules("I have 3 cats")
        assert "3" in glosses

    def test_multiple_sentences(self):
        # Each sentence should work independently
        g1 = apply_all_rules("I am happy")
        g2 = apply_all_rules("You are sad")

        assert "HAPPY" in g1
        assert "SAD" in g2
