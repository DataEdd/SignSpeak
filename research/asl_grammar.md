# ASL Grammar Rules Documentation

This document contains linguistically-sourced grammar rules for the SignBridge translation system. Each rule must have citations, examples, and test cases.

---

## 1. Time-First Rule (Temporal Fronting)

### Description
In ASL, temporal information (time signs) typically appears at the beginning of the sentence to establish the temporal frame before the main content. This differs from English where time expressions can appear in various positions.

### Linguistic Basis

**Primary Sources:**

1. **Valli & Lucas (2000)**, *Linguistics of American Sign Language: An Introduction*, pp. 142-145:
   > "Time adverbs typically occur at the beginning of sentences to establish temporal frame. This temporal establishment sets up the context for everything that follows."

2. **Baker-Shenk & Cokely (1980)**, *American Sign Language: A Teacher's Resource Text on Grammar and Culture*, p. 163:
   > "Time signs generally precede the rest of the sentence. The signer establishes WHEN something happened before describing WHAT happened."

3. **Neidle et al. (2000)**, *The Syntax of American Sign Language*, pp. 45-47:
   > "Temporal adverbials occupy a sentence-initial position, preceding the subject and predicate. This is obligatory for many time expressions."

### Rule Specification

```
IF sentence contains time expression THEN
    MOVE time expression to sentence start
    REMOVE redundant tense markers
```

**Time Signs Affected:**
- YESTERDAY, TODAY, TOMORROW
- LAST-WEEK, THIS-WEEK, NEXT-WEEK
- LAST-MONTH, THIS-MONTH, NEXT-MONTH
- LAST-YEAR, THIS-YEAR, NEXT-YEAR
- MORNING, AFTERNOON, EVENING, NIGHT
- NOW, LATER, RECENTLY, SOON
- Specific times: MONDAY, TUESDAY, etc.
- Duration markers: LONG-TIME-AGO, FUTURE

### Examples

| English | ASL Gloss | Notes |
|---------|-----------|-------|
| I went to the store yesterday. | YESTERDAY I GO STORE | Time fronted, past tense dropped |
| Tomorrow I will eat pizza. | TOMORROW I EAT PIZZA | Time fronted, future tense dropped |
| She studied last night. | LAST-NIGHT SHE STUDY | Time fronted |
| We will meet next week. | NEXT-WEEK WE MEET | Time fronted |
| I'm eating now. | NOW I EAT | Time fronted |
| He called me this morning. | THIS-MORNING HE CALL ME | Time fronted |

### Edge Cases

1. **Multiple time expressions**: Most specific comes first
   - "I'll see you tomorrow at 3pm" → TOMORROW 3 I SEE YOU

2. **Implied time from context**: If time was already established, may be omitted
   - "Yesterday I went to school. I saw my friend." → YESTERDAY I GO SCHOOL. I SEE MY FRIEND.

3. **Questions about time**: WH-word may come at end
   - "When did you go?" → YOU GO WHEN?

4. **Conditional time**: "If" clauses have their own structure
   - "If it rains tomorrow..." → TOMORROW SUPPOSE RAIN...

### Implementation Notes

For `packages/translation/grammar_rules.py`:
- Detect time expressions using NLP NER or keyword matching
- Store time position for later reconstruction if needed
- Handle compound time expressions (LAST WEEK MONDAY)

### Test Cases

See `test_corpus.md` for validated examples.

---

## 2. Topic-Comment Structure

### Description
ASL uses Topic-Comment structure rather than English's Subject-Verb-Object (SVO). The topic is established first (often with raised eyebrows), then the comment is made about it.

### Linguistic Basis

**Primary Sources:**

1. **Valli & Lucas (2000)**, *Linguistics of American Sign Language*, pp. 135-141:
   > "Topicalization is a fundamental feature of ASL syntax. The topic is marked with raised eyebrows and a slight head tilt, followed by a pause, then the comment."

2. **Liddell (2003)**, *Grammar, Gesture, and Meaning in ASL*, pp. 52-58:
   > "Topic-comment structures allow the signer to establish a referent clearly before making statements about it."

### Non-Manual Markers

- **Topic**: Raised eyebrows, head tilt back, slight pause
- Notation: `t` or `topic` over the glosses

```
      ___t___
BOOK, I READ FINISH
```

### Examples

| English | ASL Gloss | Notes |
|---------|-----------|-------|
| I read the book. | BOOK, I READ | Book topicalized |
| The car is red. | CAR, RED | Topic-comment, no "is" |
| I like coffee. | COFFEE, I LIKE | Object topicalized |
| My sister is tall. | MY SISTER, TALL | Subject as topic |

### Edge Cases

1. Not all sentences require topicalization
2. Context determines when topicalization is appropriate
3. New information typically comes in the comment

---

## 3. Dropped Elements

### Description
ASL regularly drops elements that would be required in English, including articles, copula verbs, and some pronouns when recoverable from context.

### Linguistic Basis

1. **Valli & Lucas (2000)**, pp. 126-130:
   > "ASL has no equivalent to English articles 'a', 'an', 'the'. Definiteness is conveyed through other means including pointing, spatial reference, and context."

2. **Sandler & Lillo-Martin (2006)**, *Sign Language and Linguistic Universals*, pp. 78-82:
   > "The copula 'be' has no direct equivalent in ASL. Predicate adjectives and predicate nominals are expressed directly without a linking verb."

### Elements to Drop

| English Element | ASL Treatment | Example |
|-----------------|---------------|---------|
| a, an, the | DROP | "the book" → BOOK |
| am, is, are, was, were | DROP | "I am happy" → I HAPPY |
| -ing (progressive) | DROP | "I am eating" → I EAT |
| -ed (past) | DROP (use time sign) | "I walked" → PAST I WALK or YESTERDAY I WALK |
| will (future) | DROP (use time sign) | "I will go" → TOMORROW I GO |

### Examples

| English | ASL Gloss |
|---------|-----------|
| The cat is on the table. | CAT ON TABLE |
| She is a teacher. | SHE TEACHER |
| I am going to the store. | I GO STORE |
| He was reading a book. | PAST HE READ BOOK |

---

## 4. WH-Question Placement

### Description
In ASL, WH-question words (WHO, WHAT, WHERE, WHEN, WHY, HOW) often appear at the end of the sentence, sometimes duplicated at both beginning and end.

### Linguistic Basis

1. **Baker-Shenk & Cokely (1980)**, pp. 178-183:
   > "WH-signs typically occur in sentence-final position in ASL, often accompanied by furrowed brows and a forward head tilt."

2. **Neidle et al. (2000)**, pp. 89-95:
   > "WH-phrases in ASL can appear sentence-finally, sentence-initially, or in both positions simultaneously (doubling)."

### Non-Manual Markers

- **WH-questions**: Furrowed eyebrows, head tilted forward
- Notation: `wh` over the glosses

```
      ___wh___
YOU NAME WHAT
```

### Examples

| English | ASL Gloss | Alternative |
|---------|-----------|-------------|
| What is your name? | YOU NAME WHAT | WHAT YOU NAME WHAT |
| Where do you live? | YOU LIVE WHERE | WHERE YOU LIVE WHERE |
| Who is that? | THAT WHO | WHO THAT WHO |
| When are you leaving? | YOU LEAVE WHEN | |
| Why did you go? | YOU GO WHY | |
| How are you? | YOU HOW | HOW YOU |

---

## 5. Yes/No Questions

### Description
Yes/No questions in ASL are distinguished from statements primarily through non-manual markers, not word order changes.

### Linguistic Basis

1. **Valli & Lucas (2000)**, pp. 173-175:
   > "Yes-no questions are marked by raised eyebrows, widened eyes, and a forward head tilt. The manual signs may be identical to a statement."

### Non-Manual Markers

- **Yes/No questions**: Raised eyebrows, widened eyes, head tilted forward, last sign held slightly longer
- Notation: `q` or `y/n` over the glosses

```
     ___q___
YOU STUDENT?
```

### Examples

| English | ASL Gloss | Non-Manual |
|---------|-----------|------------|
| Are you a student? | YOU STUDENT | Raised eyebrows |
| Did you eat? | YOU EAT FINISH | Raised eyebrows |
| Is she coming? | SHE COME | Raised eyebrows |

---

## Implementation Priority

For the translation package, implement rules in this order:

1. **Time-First** - High impact, clear rules
2. **Dropped Elements** - Essential for natural output
3. **WH-Question Placement** - Important for questions
4. **Yes/No Questions** - Non-manual markers (metadata)
5. **Topic-Comment** - Complex, context-dependent

---

## References

See `bibliography.md` for complete source information.
