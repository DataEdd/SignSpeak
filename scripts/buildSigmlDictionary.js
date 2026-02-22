#!/usr/bin/env node
/**
 * Build script: reads all .sigml files and dictionary.json from SignBridge
 * and generates a single JS module for the SignSpeak web frontend.
 */
const fs = require('fs')
const path = require('path')

const SIGML_BASE = path.resolve(__dirname, '../../SignSpeak/13hacks/SignBridge/backend/data/sigml')
const DICT_PATH = path.resolve(__dirname, '../../SignSpeak/13hacks/SignBridge/backend/data/dictionary.json')
const OUTPUT = path.resolve(__dirname, '../apps/web/src/data/sigmlDictionary.js')

function readSigmlFiles(dir, prefix) {
  const entries = {}
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.sigml'))
  for (const file of files) {
    const content = fs.readFileSync(path.join(dir, file), 'utf8').trim()
    const gloss = file.replace('.sigml', '').toUpperCase()
    const key = prefix ? `${prefix}:${gloss}` : gloss
    entries[key] = content
  }
  return entries
}

// Read all sigml files
const wordSigns = readSigmlFiles(path.join(SIGML_BASE, 'words'), null)
const alphabetSigns = readSigmlFiles(path.join(SIGML_BASE, 'alphabet'), null)
const aslSigns = readSigmlFiles(path.join(SIGML_BASE, 'asl'), 'ASL')
const islSigns = readSigmlFiles(path.join(SIGML_BASE, 'isl'), 'ISL')

// Merge all signs (words take precedence for bare keys)
const SIGML_SIGNS = { ...wordSigns, ...alphabetSigns, ...aslSigns, ...islSigns }

// Read dictionary
const dictionary = JSON.parse(fs.readFileSync(DICT_PATH, 'utf8'))
const WORD_TO_GLOSS = {}
for (const [word, entry] of Object.entries(dictionary)) {
  WORD_TO_GLOSS[word] = entry.gloss
}

// Generate JS module
const lines = [
  '/**',
  ' * Auto-generated SiGML dictionary.',
  ` * Generated: ${new Date().toISOString()}`,
  ` * ${Object.keys(SIGML_SIGNS).length} signs, ${Object.keys(WORD_TO_GLOSS).length} dictionary entries`,
  ' */',
  '',
  '// All SiGML sign definitions keyed by gloss (uppercase)',
  `export const SIGML_SIGNS = ${JSON.stringify(SIGML_SIGNS, null, 2)}`,
  '',
  '// Word-to-gloss mapping (lowercase word -> uppercase gloss)',
  `export const WORD_TO_GLOSS = ${JSON.stringify(WORD_TO_GLOSS, null, 2)}`,
  '',
]

fs.mkdirSync(path.dirname(OUTPUT), { recursive: true })
fs.writeFileSync(OUTPUT, lines.join('\n'), 'utf8')
console.log(`Generated ${OUTPUT}`)
console.log(`  Signs: ${Object.keys(SIGML_SIGNS).length}`)
console.log(`  Dictionary entries: ${Object.keys(WORD_TO_GLOSS).length}`)
