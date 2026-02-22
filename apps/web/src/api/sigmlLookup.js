/**
 * SiGML lookup and combiner functions.
 * Provides gloss-to-SiGML resolution and multi-sign combination.
 */
import { SIGML_SIGNS, WORD_TO_GLOSS } from '../data/sigmlDictionary'

/**
 * Get the SiGML XML string for a known gloss.
 * @param {string} gloss - Uppercase gloss (e.g. "HELLO")
 * @returns {string|null} SiGML XML or null if not found
 */
export function getSigmlForGloss(gloss) {
  return SIGML_SIGNS[gloss] || null
}

/**
 * Look up the gloss for an English word.
 * @param {string} word - Lowercase English word
 * @returns {string|null} Uppercase gloss or null
 */
export function wordToGloss(word) {
  return WORD_TO_GLOSS[word.toLowerCase()] || null
}

/**
 * Extract the inner content of a <sigml> document (the <hns_sign> elements).
 * @param {string} sigmlText - Full SiGML XML document
 * @returns {string|null} Inner content or null
 */
function extractSignContent(sigmlText) {
  if (!sigmlText) return null

  // Match content inside <sigml> tags
  const match = sigmlText.match(/<sigml[^>]*>([\s\S]*?)<\/sigml>/)
  if (match) return match[1].trim()

  // Fallback: look for hns_sign or sign tags
  const hnsMatch = sigmlText.match(/(<hns_sign[\s\S]*?<\/hns_sign>)/)
  if (hnsMatch) return hnsMatch[1].trim()

  const signMatch = sigmlText.match(/(<sign[\s\S]*?<\/sign>)/)
  if (signMatch) return signMatch[1].trim()

  return sigmlText.trim()
}

/**
 * Combine multiple gloss SiGML documents into a single <sigml> document.
 * Ported from SignBridge combiner.py.
 * @param {string[]} glosses - Array of uppercase glosses
 * @returns {string|null} Combined SiGML XML or null if none found
 */
export function buildSigmlSequence(glosses) {
  if (!glosses || glosses.length === 0) return null

  const parts = []
  for (const gloss of glosses) {
    const sigml = getSigmlForGloss(gloss)
    if (sigml) {
      const content = extractSignContent(sigml)
      if (content) parts.push(content)
    }
  }

  if (parts.length === 0) return null

  if (parts.length === 1) {
    const sigml = getSigmlForGloss(glosses.find(g => getSigmlForGloss(g)))
    return sigml
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<sigml>
  ${parts.join('\n  ')}
</sigml>`
}

/**
 * Produce combined SiGML for letter-by-letter fingerspelling of a word.
 * @param {string} word - The word to fingerspell
 * @returns {string|null} Combined SiGML XML or null
 */
export function fingerspellToSigml(word) {
  if (!word) return null

  const letters = word.toUpperCase().split('').filter(ch => /[A-Z]/.test(ch))
  if (letters.length === 0) return null

  return buildSigmlSequence(letters)
}

/**
 * Given an array of glosses (from mockTranslate), build the full SiGML
 * by looking up each gloss. Unknown glosses get fingerspelled.
 * @param {string[]} glosses - Array of uppercase glosses
 * @returns {string|null} Combined SiGML XML
 */
export function glossesToSigml(glosses) {
  if (!glosses || glosses.length === 0) return null

  const allParts = []

  for (const gloss of glosses) {
    const sigml = getSigmlForGloss(gloss)
    if (sigml) {
      const content = extractSignContent(sigml)
      if (content) allParts.push(content)
    } else {
      // Unknown gloss — fingerspell it
      const letters = gloss.split('').filter(ch => /[A-Z]/.test(ch))
      for (const letter of letters) {
        const letterSigml = getSigmlForGloss(letter)
        if (letterSigml) {
          const content = extractSignContent(letterSigml)
          if (content) allParts.push(content)
        }
      }
    }
  }

  if (allParts.length === 0) return null
  if (allParts.length === 1) {
    return `<?xml version="1.0" encoding="UTF-8"?>
<sigml>
  ${allParts[0]}
</sigml>`
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<sigml>
  ${allParts.join('\n  ')}
</sigml>`
}
