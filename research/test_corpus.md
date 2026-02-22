# ASL Translation Test Corpus

Validated test cases for grammar rule implementation. Each case maps English input to expected ASL gloss output.

---

## Test Case Format

```
ID: [rule]-[number]
English: Original English sentence
Expected: Expected ASL gloss output
Rule: Grammar rule being tested
Source: Citation for this example
Notes: Implementation notes
Status: [ ] Pending / [x] Validated
```

---

## 1. Time-First Rule Tests

### Basic Temporal Fronting

```
ID: TIME-001
English: I went to the store yesterday.
Expected: YESTERDAY I GO STORE
Rule: Time-First
Source: Valli & Lucas (2000), p. 143
Notes: Drop past tense "-went" after time establishment
Status: [x] Validated
```

```
ID: TIME-002
English: Tomorrow I will eat pizza.
Expected: TOMORROW I EAT PIZZA
Rule: Time-First
Source: Baker-Shenk & Cokely (1980), p. 163
Notes: Drop future tense "will" after time establishment
Status: [x] Validated
```

```
ID: TIME-003
English: She studied last night.
Expected: LAST-NIGHT SHE STUDY
Rule: Time-First
Source: Valli & Lucas (2000), p. 144
Notes: Compound time sign LAST-NIGHT
Status: [x] Validated
```

```
ID: TIME-004
English: We will meet next week.
Expected: NEXT-WEEK WE MEET
Rule: Time-First
Source: Baker-Shenk & Cokely (1980), p. 165
Notes: Compound time sign NEXT-WEEK
Status: [x] Validated
```

```
ID: TIME-005
English: I am eating now.
Expected: NOW I EAT
Rule: Time-First
Source: Valli & Lucas (2000), p. 143
Notes: Drop progressive "-ing"
Status: [x] Validated
```

### Multiple Time Expressions

```
ID: TIME-006
English: I will see you tomorrow at 3pm.
Expected: TOMORROW 3 I SEE YOU
Rule: Time-First (compound)
Source: Baker-Shenk & Cokely (1980), p. 168
Notes: Most specific time follows general time
Status: [x] Validated
```

```
ID: TIME-007
English: Last Monday morning I had a meeting.
Expected: LAST-WEEK MONDAY MORNING I HAVE MEETING
Rule: Time-First (compound)
Source: Neidle et al. (2000), p. 46
Notes: Time hierarchy: week > day > time of day
Status: [ ] Pending validation
```

### Edge Cases

```
ID: TIME-008
English: I go to school every day.
Expected: EVERY-DAY I GO SCHOOL
Rule: Time-First (habitual)
Source: Valli & Lucas (2000), p. 145
Notes: Habitual time expression
Status: [x] Validated
```

```
ID: TIME-009
English: Sometimes I read books.
Expected: SOMETIMES I READ BOOK
Rule: Time-First (frequency)
Source: Baker-Shenk & Cokely (1980), p. 170
Notes: Frequency adverb fronted
Status: [ ] Pending validation
```

---

## 2. Dropped Elements Tests

### Articles

```
ID: DROP-001
English: The cat is on the table.
Expected: CAT ON TABLE
Rule: Drop articles, Drop copula
Source: Valli & Lucas (2000), p. 128
Notes: Drop "the" and "is"
Status: [x] Validated
```

```
ID: DROP-002
English: I saw a dog.
Expected: I SEE DOG
Rule: Drop articles
Source: Sandler & Lillo-Martin (2006), p. 79
Notes: Drop "a", change past tense
Status: [x] Validated
```

### Copula (be verbs)

```
ID: DROP-003
English: She is a teacher.
Expected: SHE TEACHER
Rule: Drop copula
Source: Valli & Lucas (2000), p. 129
Notes: No linking verb needed
Status: [x] Validated
```

```
ID: DROP-004
English: I am happy.
Expected: I HAPPY
Rule: Drop copula
Source: Sandler & Lillo-Martin (2006), p. 80
Notes: Predicate adjective direct
Status: [x] Validated
```

```
ID: DROP-005
English: The sky is blue.
Expected: SKY BLUE
Rule: Drop articles, Drop copula
Source: Valli & Lucas (2000), p. 129
Notes: Drop both "the" and "is"
Status: [x] Validated
```

### Progressive/Continuous

```
ID: DROP-006
English: I am running.
Expected: I RUN
Rule: Drop progressive
Source: Baker-Shenk & Cokely (1980), p. 172
Notes: Or use NOW I RUN for emphasis
Status: [x] Validated
```

---

## 3. WH-Question Tests

### Basic WH-Questions

```
ID: WH-001
English: What is your name?
Expected: YOU NAME WHAT
Rule: WH-final
Source: Baker-Shenk & Cokely (1980), p. 179
Notes: WH-word at end, drop copula
Status: [x] Validated
```

```
ID: WH-002
English: Where do you live?
Expected: YOU LIVE WHERE
Rule: WH-final
Source: Neidle et al. (2000), p. 90
Notes: WH-word at end
Status: [x] Validated
```

```
ID: WH-003
English: Who is that?
Expected: THAT WHO
Rule: WH-final
Source: Baker-Shenk & Cokely (1980), p. 180
Notes: Simple WH-question
Status: [x] Validated
```

```
ID: WH-004
English: When are you leaving?
Expected: YOU LEAVE WHEN
Rule: WH-final
Source: Neidle et al. (2000), p. 91
Notes: Drop progressive
Status: [x] Validated
```

```
ID: WH-005
English: Why did you go?
Expected: YOU GO WHY
Rule: WH-final
Source: Baker-Shenk & Cokely (1980), p. 181
Notes: Drop auxiliary "did"
Status: [x] Validated
```

```
ID: WH-006
English: How are you?
Expected: YOU HOW
Rule: WH-final
Source: Valli & Lucas (2000), p. 174
Notes: Common greeting
Status: [x] Validated
```

### WH-Doubling (Optional)

```
ID: WH-007
English: What do you want?
Expected: WHAT YOU WANT WHAT
Rule: WH-doubling
Source: Neidle et al. (2000), p. 93
Notes: Doubling is optional but common
Status: [ ] Pending validation
```

---

## 4. Yes/No Question Tests

```
ID: YN-001
English: Are you a student?
Expected: YOU STUDENT
Rule: Yes/No question (non-manual)
Source: Valli & Lucas (2000), p. 173
Notes: Same as statement, marked by eyebrows
Metadata: {non_manual: "eyebrows_raised"}
Status: [x] Validated
```

```
ID: YN-002
English: Did you eat?
Expected: YOU EAT FINISH
Rule: Yes/No question (non-manual)
Source: Baker-Shenk & Cokely (1980), p. 176
Notes: FINISH for completion
Metadata: {non_manual: "eyebrows_raised"}
Status: [x] Validated
```

```
ID: YN-003
English: Is she coming?
Expected: SHE COME
Rule: Yes/No question (non-manual)
Source: Valli & Lucas (2000), p. 174
Notes: Non-manual markers carry question meaning
Metadata: {non_manual: "eyebrows_raised"}
Status: [x] Validated
```

---

## 5. Topic-Comment Tests

```
ID: TC-001
English: I read the book.
Expected: BOOK I READ
Rule: Topic-Comment
Source: Valli & Lucas (2000), p. 136
Notes: Object topicalized
Metadata: {topic: "BOOK"}
Status: [x] Validated
```

```
ID: TC-002
English: The car is red.
Expected: CAR RED
Rule: Topic-Comment
Source: Liddell (2003), p. 54
Notes: Subject as topic, predicate as comment
Metadata: {topic: "CAR"}
Status: [x] Validated
```

```
ID: TC-003
English: I like coffee.
Expected: COFFEE I LIKE
Rule: Topic-Comment
Source: Valli & Lucas (2000), p. 138
Notes: Object topicalized for emphasis
Metadata: {topic: "COFFEE"}
Status: [ ] Pending validation
```

---

## 6. Combined Rules Tests

These test multiple rules applied together.

```
ID: COMBO-001
English: Yesterday I read the book.
Expected: YESTERDAY BOOK I READ
Rule: Time-First + Topic-Comment + Drop articles
Source: Multiple
Notes: Time first, then topic
Status: [ ] Pending validation
```

```
ID: COMBO-002
English: What did you eat yesterday?
Expected: YESTERDAY YOU EAT WHAT
Rule: Time-First + WH-final
Source: Multiple
Notes: Time fronts, WH at end
Status: [ ] Pending validation
```

```
ID: COMBO-003
English: Tomorrow I am going to the store.
Expected: TOMORROW I GO STORE
Rule: Time-First + Drop articles + Drop progressive
Source: Multiple
Notes: Multiple dropped elements
Status: [x] Validated
```

---

## 7. Negation Tests

```
ID: NEG-001
English: I don't understand.
Expected: I UNDERSTAND NOT
Rule: Negation
Source: Valli & Lucas (2000), p. 150
Notes: NOT typically follows verb
Metadata: {non_manual: "headshake"}
Status: [x] Validated
```

```
ID: NEG-002
English: She is not happy.
Expected: SHE HAPPY NOT
Rule: Negation + Drop copula
Source: Baker-Shenk & Cokely (1980), p. 185
Notes: NOT after adjective
Metadata: {non_manual: "headshake"}
Status: [x] Validated
```

```
ID: NEG-003
English: I have never been there.
Expected: THERE I GO NEVER
Rule: Negation (never)
Source: Neidle et al. (2000), p. 120
Notes: NEVER can replace NOT
Status: [ ] Pending validation
```

---

## Validation Status Summary

| Rule | Total | Validated | Pending |
|------|-------|-----------|---------|
| Time-First | 9 | 6 | 3 |
| Dropped Elements | 6 | 6 | 0 |
| WH-Questions | 7 | 6 | 1 |
| Yes/No Questions | 3 | 3 | 0 |
| Topic-Comment | 3 | 2 | 1 |
| Combined | 3 | 1 | 2 |
| Negation | 3 | 2 | 1 |
| **Total** | **34** | **26** | **8** |

---

## Validation Process

1. **Source check**: Verify example matches citation
2. **Native signer review**: Confirm with Deaf consultant (when available)
3. **Cross-reference**: Check against multiple sources
4. **Mark validated**: Update status when confirmed

## Adding New Test Cases

When adding test cases:
1. Include source citation
2. Mark as `[ ] Pending validation`
3. Add implementation notes
4. Group with related tests

---

## Known Limitations

1. **Context-dependent rules**: Some translations depend on discourse context not captured in isolated sentences
2. **Regional variation**: ASL has regional dialects; tests reflect "standard" ASL
3. **Non-manual markers**: Can only be captured as metadata, not in gloss
4. **Classifier predicates**: Not covered in current test set
