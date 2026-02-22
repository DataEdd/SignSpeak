/**
 * Client-side ASL grammar engine for demo mode.
 * Mirrors the real backend's translation pipeline so the GitHub Pages
 * demo works without a running server.
 */

const ARTICLES = new Set(['a', 'an', 'the'])
const BE_VERBS = new Set(['is', 'are', 'was', 'were', 'am', 'be', 'been', 'being'])
const AUXILIARIES = new Set([
  'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can',
  'shall', 'may', 'might', 'must', 'have', 'has', 'had'
])
const TIME_WORDS = new Set([
  'yesterday', 'tomorrow', 'today', 'now', 'later', 'soon', 'recently',
  'already', 'before', 'after', 'always', 'never', 'sometimes',
  'morning', 'evening', 'tonight', 'weekly', 'daily', 'monthly'
])
const WH_WORDS = new Set(['what', 'where', 'when', 'who', 'why', 'how', 'which'])
const NEGATIONS = new Set(["not", "n't", "don't", "doesn't", "didn't", "won't",
  "wouldn't", "couldn't", "shouldn't", "can't", "isn't", "aren't",
  "wasn't", "weren't", "haven't", "hasn't", "hadn't"])

/** Strip -ing / -ed suffixes to approximate the base verb form. */
function simplifyVerb(word) {
  if (word.endsWith('ing') && word.length > 4) {
    const stem = word.slice(0, -3)
    if (stem.endsWith(stem[stem.length - 1]) && stem.length > 2) {
      return stem.slice(0, -1) // running -> run
    }
    return stem // signing -> sign
  }
  if (word.endsWith('ed') && word.length > 4) {
    return word.slice(0, -2) // helped -> help
  }
  return word
}

/**
 * Translate English text to ASL gloss order.
 *
 * Rules applied (simplified ASL grammar):
 *  1. Tokenize and lowercase
 *  2. Remove articles, be-verbs, auxiliaries
 *  3. Simplify verb forms (-ing, -ed → base)
 *  4. Move time words to front
 *  5. Move WH-words to end (question reordering)
 *  6. Move negation to end
 *  7. Uppercase all glosses
 *
 * @param {string} text - English sentence
 * @returns {{ glosses: string[], confidence: number, isDemo: true }}
 */
export function mockTranslate(text) {
  if (!text || !text.trim()) {
    return { glosses: [], confidence: 0, isDemo: true }
  }

  const tokens = text
    .toLowerCase()
    .replace(/[^\w\s'-]/g, '')
    .split(/\s+/)
    .filter(Boolean)

  const timeTokens = []
  const whTokens = []
  const negTokens = []
  const mainTokens = []

  for (const token of tokens) {
    if (TIME_WORDS.has(token)) {
      timeTokens.push(token)
    } else if (WH_WORDS.has(token)) {
      whTokens.push(token)
    } else if (NEGATIONS.has(token)) {
      negTokens.push('NOT')
    } else if (ARTICLES.has(token) || BE_VERBS.has(token) || AUXILIARIES.has(token)) {
      // drop these
    } else {
      mainTokens.push(simplifyVerb(token))
    }
  }

  // ASL word order: TIME + TOPIC/MAIN + WH-QUESTION + NEGATION
  const glosses = [
    ...timeTokens,
    ...mainTokens,
    ...whTokens,
    ...negTokens,
  ].map(g => g.toUpperCase())

  const confidence = Math.max(0.65, 0.95 - tokens.length * 0.01)

  return { glosses, confidence, isDemo: true }
}
