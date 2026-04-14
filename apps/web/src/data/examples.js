// Preloaded example phrases that ship with a pre-rendered video clip.
// Matching input skips CWASA at runtime and plays the bundled file directly.
//
// All glosses used here were audited by scripts/auditAllGlosses.mjs and render
// cleanly through the CWASA HamNoSys → animation pipeline. The full 131-entry
// SiGML dictionary is authored at varying quality; we only reference the
// validated subset so preloaded playback is deterministic.
//
// The recorder script at scripts/recordPhrases.mjs reads this list, records
// one clip per unique gloss, and ffmpeg-concats to apps/web/public/videos/examples/.

export const EXAMPLES = [
  {
    phrase: 'Hello',
    slug: 'hello',
    glosses: ['HELLO']
  },
  {
    phrase: 'Thank you',
    slug: 'thank-you',
    glosses: ['THANK_YOU']
  },
  {
    phrase: 'Please',
    slug: 'please',
    glosses: ['PLEASE']
  },
  {
    phrase: 'Good',
    slug: 'good',
    glosses: ['GOOD']
  },
  {
    phrase: 'Love',
    slug: 'love',
    glosses: ['LOVE']
  },
  {
    phrase: 'Where from?',
    slug: 'where-from',
    glosses: ['WHERE', 'FROM']
  }
]

/** Normalize user input for comparison against example phrases. */
export function normalize(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

const BY_NORMALIZED = new Map(
  EXAMPLES.map((e) => [normalize(e.phrase), e])
)

export function findExample(text) {
  return BY_NORMALIZED.get(normalize(text)) || null
}
