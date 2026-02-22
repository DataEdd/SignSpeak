"""ASL grammar transformation rules.

Each rule is backed by linguistic research and includes citations.
Rules transform English parse structures into ASL gloss order.

References:
- Valli, C. & Lucas, C. (2000). Linguistics of American Sign Language. 3rd ed.
- Sandler, W. & Lillo-Martin, D. (2006). Sign Language and Linguistic Universals.
- Neidle, C. et al. (2000). The Syntax of American Sign Language.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re


class TokenType(Enum):
    """Token types for parsed English."""
    WORD = "word"
    TIME = "time"
    QUESTION_WORD = "question_word"
    NEGATION = "negation"
    ARTICLE = "article"
    BE_VERB = "be_verb"
    PRONOUN = "pronoun"
    NUMBER = "number"


@dataclass
class Token:
    """A token in the parse."""
    text: str
    token_type: TokenType
    original: str = ""

    def __post_init__(self):
        if not self.original:
            self.original = self.text


# Word lists for classification
TIME_MARKERS = {
    "yesterday", "today", "tomorrow", "now", "later", "soon",
    "before", "after", "always", "never", "sometimes", "often",
    "morning", "afternoon", "evening", "night", "week", "month", "year",
    "last", "next", "ago", "recently", "already", "just", "finally",
}

QUESTION_WORDS = {
    "what", "who", "where", "when", "why", "how", "which", "whose",
}

ARTICLES = {"a", "an", "the"}

BE_VERBS = {"is", "are", "am", "was", "were", "be", "been", "being"}

NEGATION_WORDS = {
    "not", "no", "never", "nothing", "nobody", "nowhere",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't",
    "shouldn't", "can't", "cannot", "isn't", "aren't", "wasn't", "weren't",
}

# Auxiliary verbs to remove (especially in negation contexts)
AUXILIARY_VERBS = {"do", "does", "did", "will", "would", "could", "should", "can", "may", "might"}

# Irregular verb mappings (past/participle -> base form)
IRREGULAR_VERBS = {
    # be (handled separately)
    # have
    "had": "have",
    "has": "have",
    "having": "have",
    # do
    "did": "do",
    "does": "do",
    "doing": "do",
    "done": "do",
    # go
    "went": "go",
    "gone": "go",
    "going": "go",
    "goes": "go",
    # come
    "came": "come",
    "coming": "come",
    "comes": "come",
    # see
    "saw": "see",
    "seen": "see",
    "seeing": "see",
    "sees": "see",
    # know
    "knew": "know",
    "known": "know",
    "knowing": "know",
    "knows": "know",
    # think
    "thought": "think",
    "thinking": "think",
    "thinks": "think",
    # take
    "took": "take",
    "taken": "take",
    "taking": "take",
    "takes": "take",
    # make
    "made": "make",
    "making": "make",
    "makes": "make",
    # get
    "got": "get",
    "gotten": "get",
    "getting": "get",
    "gets": "get",
    # give
    "gave": "give",
    "given": "give",
    "giving": "give",
    "gives": "give",
    # find
    "found": "find",
    "finding": "find",
    "finds": "find",
    # say
    "said": "say",
    "saying": "say",
    "says": "say",
    # tell
    "told": "tell",
    "telling": "tell",
    "tells": "tell",
    # feel
    "felt": "feel",
    "feeling": "feel",
    "feels": "feel",
    # become
    "became": "become",
    "becoming": "become",
    "becomes": "become",
    # leave
    "left": "leave",
    "leaving": "leave",
    "leaves": "leave",
    # put
    "putting": "put",
    "puts": "put",
    # keep
    "kept": "keep",
    "keeping": "keep",
    "keeps": "keep",
    # let
    "letting": "let",
    "lets": "let",
    # begin
    "began": "begin",
    "begun": "begin",
    "beginning": "begin",
    "begins": "begin",
    # seem
    "seemed": "seem",
    "seeming": "seem",
    "seems": "seem",
    # help
    "helped": "help",
    "helping": "help",
    "helps": "help",
    # show
    "showed": "show",
    "shown": "show",
    "showing": "show",
    "shows": "show",
    # hear
    "heard": "hear",
    "hearing": "hear",
    "hears": "hear",
    # play
    "played": "play",
    "playing": "play",
    "plays": "play",
    # run
    "ran": "run",
    "running": "run",
    "runs": "run",
    # move
    "moved": "move",
    "moving": "move",
    "moves": "move",
    # live
    "lived": "live",
    "living": "live",
    "lives": "live",
    # believe
    "believed": "believe",
    "believing": "believe",
    "believes": "believe",
    # bring
    "brought": "bring",
    "bringing": "bring",
    "brings": "bring",
    # write
    "wrote": "write",
    "written": "write",
    "writing": "write",
    "writes": "write",
    # sit
    "sat": "sit",
    "sitting": "sit",
    "sits": "sit",
    # stand
    "stood": "stand",
    "standing": "stand",
    "stands": "stand",
    # lose
    "lost": "lose",
    "losing": "lose",
    "loses": "lose",
    # pay
    "paid": "pay",
    "paying": "pay",
    "pays": "pay",
    # meet
    "met": "meet",
    "meeting": "meet",
    "meets": "meet",
    # include
    "included": "include",
    "including": "include",
    "includes": "include",
    # continue
    "continued": "continue",
    "continuing": "continue",
    "continues": "continue",
    # set
    "setting": "set",
    "sets": "set",
    # learn
    "learned": "learn",
    "learnt": "learn",
    "learning": "learn",
    "learns": "learn",
    # lead
    "led": "lead",
    "leading": "lead",
    "leads": "lead",
    # understand
    "understood": "understand",
    "understanding": "understand",
    "understands": "understand",
    # watch
    "watched": "watch",
    "watching": "watch",
    "watches": "watch",
    # follow
    "followed": "follow",
    "following": "follow",
    "follows": "follow",
    # stop
    "stopped": "stop",
    "stopping": "stop",
    "stops": "stop",
    # create
    "created": "create",
    "creating": "create",
    "creates": "create",
    # speak
    "spoke": "speak",
    "spoken": "speak",
    "speaking": "speak",
    "speaks": "speak",
    # read
    "reading": "read",
    "reads": "read",
    # spend
    "spent": "spend",
    "spending": "spend",
    "spends": "spend",
    # grow
    "grew": "grow",
    "grown": "grow",
    "growing": "grow",
    "grows": "grow",
    # open
    "opened": "open",
    "opening": "open",
    "opens": "open",
    # walk
    "walked": "walk",
    "walking": "walk",
    "walks": "walk",
    # win
    "won": "win",
    "winning": "win",
    "wins": "win",
    # teach
    "taught": "teach",
    "teaching": "teach",
    "teaches": "teach",
    # offer
    "offered": "offer",
    "offering": "offer",
    "offers": "offer",
    # remember
    "remembered": "remember",
    "remembering": "remember",
    "remembers": "remember",
    # love
    "loved": "love",
    "loving": "love",
    "loves": "love",
    # consider
    "considered": "consider",
    "considering": "consider",
    "considers": "consider",
    # appear
    "appeared": "appear",
    "appearing": "appear",
    "appears": "appear",
    # buy
    "bought": "buy",
    "buying": "buy",
    "buys": "buy",
    # wait
    "waited": "wait",
    "waiting": "wait",
    "waits": "wait",
    # serve
    "served": "serve",
    "serving": "serve",
    "serves": "serve",
    # die
    "died": "die",
    "dying": "die",
    "dies": "die",
    # send
    "sent": "send",
    "sending": "send",
    "sends": "send",
    # build
    "built": "build",
    "building": "build",
    "builds": "build",
    # stay
    "stayed": "stay",
    "staying": "stay",
    "stays": "stay",
    # fall
    "fell": "fall",
    "fallen": "fall",
    "falling": "fall",
    "falls": "fall",
    # cut
    "cutting": "cut",
    "cuts": "cut",
    # reach
    "reached": "reach",
    "reaching": "reach",
    "reaches": "reach",
    # kill
    "killed": "kill",
    "killing": "kill",
    "kills": "kill",
    # raise
    "raised": "raise",
    "raising": "raise",
    "raises": "raise",
    # pass
    "passed": "pass",
    "passing": "pass",
    "passes": "pass",
    # sell
    "sold": "sell",
    "selling": "sell",
    "sells": "sell",
    # decide
    "decided": "decide",
    "deciding": "decide",
    "decides": "decide",
    # return
    "returned": "return",
    "returning": "return",
    "returns": "return",
    # explain
    "explained": "explain",
    "explaining": "explain",
    "explains": "explain",
    # hope
    "hoped": "hope",
    "hoping": "hope",
    "hopes": "hope",
    # develop
    "developed": "develop",
    "developing": "develop",
    "develops": "develop",
    # carry
    "carried": "carry",
    "carrying": "carry",
    "carries": "carry",
    # break
    "broke": "break",
    "broken": "break",
    "breaking": "break",
    "breaks": "break",
    # receive
    "received": "receive",
    "receiving": "receive",
    "receives": "receive",
    # agree
    "agreed": "agree",
    "agreeing": "agree",
    "agrees": "agree",
    # support
    "supported": "support",
    "supporting": "support",
    "supports": "support",
    # hit
    "hitting": "hit",
    "hits": "hit",
    # produce
    "produced": "produce",
    "producing": "produce",
    "produces": "produce",
    # eat
    "ate": "eat",
    "eaten": "eat",
    "eating": "eat",
    "eats": "eat",
    # cover
    "covered": "cover",
    "covering": "cover",
    "covers": "cover",
    # catch
    "caught": "catch",
    "catching": "catch",
    "catches": "catch",
    # draw
    "drew": "draw",
    "drawn": "draw",
    "drawing": "draw",
    "draws": "draw",
    # choose
    "chose": "choose",
    "chosen": "choose",
    "choosing": "choose",
    "chooses": "choose",
    # like
    "liked": "like",
    "liking": "like",
    "likes": "like",
    # want
    "wanted": "want",
    "wanting": "want",
    "wants": "want",
    # need
    "needed": "need",
    "needing": "need",
    "needs": "need",
    # try
    "tried": "try",
    "trying": "try",
    "tries": "try",
    # use
    "used": "use",
    "using": "use",
    "uses": "use",
    # work
    "worked": "work",
    "working": "work",
    "works": "work",
    # call
    "called": "call",
    "calling": "call",
    "calls": "call",
    # ask
    "asked": "ask",
    "asking": "ask",
    "asks": "ask",
    # sleep
    "slept": "sleep",
    "sleeping": "sleep",
    "sleeps": "sleep",
}

# Contractions to expand
CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "can't": "can not",
    "cannot": "can not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "let's": "let us",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "there's": "there is",
    "here's": "here is",
}


def classify_token(word: str) -> TokenType:
    """Classify a word into its token type."""
    word_lower = word.lower()

    if word_lower in TIME_MARKERS:
        return TokenType.TIME
    if word_lower in QUESTION_WORDS:
        return TokenType.QUESTION_WORD
    if word_lower in NEGATION_WORDS:
        return TokenType.NEGATION
    if word_lower in ARTICLES:
        return TokenType.ARTICLE
    if word_lower in BE_VERBS:
        return TokenType.BE_VERB
    if word_lower.isdigit():
        return TokenType.NUMBER

    return TokenType.WORD


def tokenize(text: str) -> list[Token]:
    """Tokenize and classify English text.

    Args:
        text: English sentence

    Returns:
        List of classified tokens
    """
    # Expand contractions first
    text_lower = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text_lower = text_lower.replace(contraction, expansion)

    # Split on whitespace and punctuation, keeping words
    words = re.findall(r"[a-zA-Z0-9']+", text_lower)

    tokens = []
    for word in words:
        token_type = classify_token(word)
        tokens.append(Token(
            text=word,
            token_type=token_type,
            original=word,
        ))

    return tokens


def rule_time_topic_comment(tokens: list[Token]) -> list[Token]:
    """Apply Time-Topic-Comment (TTC) structure.

    ASL uses Time-Topic-Comment ordering, where time markers come first,
    followed by the topic (what you're talking about), then the comment.

    Example: "I went to the store yesterday" -> YESTERDAY STORE I GO

    Source: Valli & Lucas (2000), Ch. 5 - "ASL exhibits Topic-Comment
    structure with time adverbials fronted to clause-initial position."

    Args:
        tokens: List of tokens

    Returns:
        Reordered tokens with time markers first
    """
    time_tokens = []
    other_tokens = []

    for token in tokens:
        if token.token_type == TokenType.TIME:
            time_tokens.append(token)
        else:
            other_tokens.append(token)

    # Time markers go first
    return time_tokens + other_tokens


def rule_remove_articles(tokens: list[Token]) -> list[Token]:
    """Remove articles (a, an, the).

    ASL does not use articles. Definiteness/indefiniteness is conveyed
    through pointing, spatial reference, or context.

    Example: "the cat" -> CAT

    Source: Valli & Lucas (2000), Ch. 4 - "ASL lacks function words
    equivalent to English articles."

    Args:
        tokens: List of tokens

    Returns:
        Tokens with articles removed
    """
    return [t for t in tokens if t.token_type != TokenType.ARTICLE]


def rule_minimal_be_verbs(tokens: list[Token]) -> list[Token]:
    """Remove most forms of 'to be'.

    ASL typically omits copular 'be' verbs. States and attributes are
    expressed by directly combining subject with predicate.

    Example: "She is happy" -> SHE HAPPY

    Source: Sandler & Lillo-Martin (2006), Ch. 8 - "The copula 'be' is
    generally absent in ASL predicate structures."

    Args:
        tokens: List of tokens

    Returns:
        Tokens with be-verbs removed
    """
    return [t for t in tokens if t.token_type != TokenType.BE_VERB]


def rule_question_wh_at_end(tokens: list[Token]) -> list[Token]:
    """Move WH-question words to the end.

    In ASL WH-questions, the question word often appears at the end
    of the clause, accompanied by a characteristic facial expression
    (lowered brows, head forward).

    Example: "What is your name?" -> YOUR NAME WHAT

    Source: Neidle et al. (2000), Ch. 3 - "WH-question words may appear
    clause-finally in ASL, with obligatory non-manual marking."

    Args:
        tokens: List of tokens

    Returns:
        Tokens with WH-words moved to end
    """
    question_tokens = []
    other_tokens = []

    for token in tokens:
        if token.token_type == TokenType.QUESTION_WORD:
            question_tokens.append(token)
        else:
            other_tokens.append(token)

    # WH-words go at the end
    return other_tokens + question_tokens


def rule_negation_at_end(tokens: list[Token]) -> list[Token]:
    """Move negation to the end of the clause.

    ASL negation often appears at clause-final position, accompanied
    by a side-to-side headshake (non-manual marker).

    Example: "I don't understand" -> I UNDERSTAND NOT

    Source: Neidle et al. (2000), Ch. 4 - "Negation in ASL can be
    expressed through manual signs (NOT, NONE) typically in clause-final
    position, always with accompanying negative headshake."

    Args:
        tokens: List of tokens

    Returns:
        Tokens with negation moved to end
    """
    negation_tokens = []
    other_tokens = []

    for token in tokens:
        if token.token_type == TokenType.NEGATION:
            negation_tokens.append(token)
        else:
            other_tokens.append(token)

    # Keep only one negation marker at the end
    if negation_tokens:
        return other_tokens + [Token(text="not", token_type=TokenType.NEGATION)]
    return other_tokens


def rule_convert_to_gloss(tokens: list[Token]) -> list[str]:
    """Convert tokens to uppercase gloss format.

    Glosses are written in uppercase by convention to distinguish
    them from English words.

    Args:
        tokens: List of tokens

    Returns:
        List of gloss strings
    """
    glosses = []
    for token in tokens:
        gloss = token.text.upper()
        # Clean up common transformations
        if gloss == "NOT":
            gloss = "NOT"  # Keep as NOT for negation
        glosses.append(gloss)
    return glosses


def rule_simplify_verbs(tokens: list[Token]) -> list[Token]:
    """Simplify verb forms to base form.

    ASL verbs do not inflect for tense (past, present, future).
    Tense is indicated by time markers or understood from context.

    Example: "walked" -> WALK, "running" -> RUN, "went" -> GO

    Source: Valli & Lucas (2000), Ch. 6 - "ASL verbs are not marked
    for tense; temporal reference is established through time adverbials."

    Args:
        tokens: List of tokens

    Returns:
        Tokens with simplified verb forms
    """
    result = []
    for token in tokens:
        if token.token_type == TokenType.WORD:
            text = token.text.lower()

            # First check irregular verb mappings
            if text in IRREGULAR_VERBS:
                text = IRREGULAR_VERBS[text]
            else:
                # Fall back to suffix-based stemming for regular verbs
                if text.endswith("ing"):
                    text = text[:-3]
                    # Handle doubling: running -> run
                    if len(text) > 2 and text[-1] == text[-2]:
                        text = text[:-1]
                    # Handle e-drop: taking -> tak -> take
                    if len(text) > 1 and text[-1] in "bcdfghjklmnpqrstvwxz":
                        # Common pattern: consonant after vowel often drops e
                        pass  # Keep as is, some words need 'e' added but this gets complex
                elif text.endswith("ed"):
                    text = text[:-2]
                    if text.endswith("i"):  # tried -> try
                        text = text[:-1] + "y"
                elif text.endswith("s") and not text.endswith("ss") and len(text) > 2:
                    # Remove plural/third person s, but not words ending in ss
                    text = text[:-1]
                    if text.endswith("ie"):  # tries -> try
                        text = text[:-2] + "y"

            result.append(Token(text=text, token_type=token.token_type, original=token.original))
        else:
            result.append(token)

    return result


def rule_remove_auxiliaries(tokens: list[Token]) -> list[Token]:
    """Remove auxiliary verbs (do, does, did, will, etc.).

    In ASL, auxiliary verbs are generally not used. Tense and aspect
    are conveyed through time markers and context, not auxiliaries.

    Example: "I do not like" -> I LIKE NOT (do is removed)

    Source: Valli & Lucas (2000), Ch. 6 - "English auxiliaries do not
    have direct ASL equivalents; aspect is marked through other means."

    Args:
        tokens: List of tokens

    Returns:
        Tokens with auxiliary verbs removed
    """
    return [t for t in tokens if t.text.lower() not in AUXILIARY_VERBS]


def apply_all_rules(text: str) -> list[str]:
    """Apply all grammar rules to English text.

    Applies rules in the correct order:
    1. Tokenize and classify
    2. Remove articles
    3. Remove be-verbs
    4. Remove auxiliary verbs
    5. Simplify verb forms
    6. Apply Time-Topic-Comment ordering
    7. Move WH-questions to end
    8. Move negation to end
    9. Convert to glosses

    Args:
        text: English sentence

    Returns:
        List of ASL glosses
    """
    # Tokenize
    tokens = tokenize(text)

    # Apply transformations
    tokens = rule_remove_articles(tokens)
    tokens = rule_minimal_be_verbs(tokens)
    tokens = rule_remove_auxiliaries(tokens)
    tokens = rule_simplify_verbs(tokens)
    tokens = rule_time_topic_comment(tokens)
    tokens = rule_question_wh_at_end(tokens)
    tokens = rule_negation_at_end(tokens)

    # Convert to glosses
    glosses = rule_convert_to_gloss(tokens)

    return glosses


# Non-manual markers (for future video layer integration)
@dataclass
class NonManualMarker:
    """Non-manual marker for grammatical meaning.

    ASL uses facial expressions and head movements as grammatical markers.
    These are as important as the manual signs themselves.

    Source: Neidle et al. (2000), Ch. 1 - "Non-manual markers in ASL
    serve critical grammatical functions including marking questions,
    negation, topics, and relative clauses."
    """
    marker_type: str  # e.g., "wh_question", "yes_no_question", "negation", "topic"
    start_gloss_index: int
    end_gloss_index: int  # -1 for end of sentence
    description: str = ""


def detect_non_manual_markers(text: str, glosses: list[str]) -> list[NonManualMarker]:
    """Detect required non-manual markers for the sentence.

    Args:
        text: Original English text
        glosses: Generated gloss sequence

    Returns:
        List of non-manual markers to apply
    """
    markers = []
    text_lower = text.lower()

    # WH-questions: lowered brows, head forward
    if any(qw in text_lower for qw in QUESTION_WORDS):
        markers.append(NonManualMarker(
            marker_type="wh_question",
            start_gloss_index=0,
            end_gloss_index=-1,
            description="Lowered eyebrows, slight head tilt forward"
        ))
    # Yes/No questions: raised brows
    elif text.strip().endswith("?"):
        markers.append(NonManualMarker(
            marker_type="yes_no_question",
            start_gloss_index=0,
            end_gloss_index=-1,
            description="Raised eyebrows, head tilted forward"
        ))

    # Negation: headshake
    if any(neg in text_lower for neg in NEGATION_WORDS):
        markers.append(NonManualMarker(
            marker_type="negation",
            start_gloss_index=0,
            end_gloss_index=-1,
            description="Side-to-side headshake"
        ))

    return markers
