// E2E smoke test of the built demo.
//   1. NLP grammar: typed input → glosses on right, no video
//   2. Showcase:    Play button → full-width video below, GLOSSES UNCHANGED
//
// Assumes `npx vite preview --port 4173` is running.

import { chromium } from 'playwright'

const BASE = process.env.PREVIEW_URL || 'http://localhost:4173/SignSpeak/'

const browser = await chromium.launch({ headless: true })
const ctx = await browser.newContext()
const page = await ctx.newPage()

const consoleErrors = []
page.on('console', (m) => {
  if (m.type() === 'error' && !/React DevTools|500 \(Internal/.test(m.text())) {
    consoleErrors.push(m.text())
  }
})
page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`))

await page.goto(BASE, { waitUntil: 'domcontentloaded' })

console.log('--- Step 1: NLP translation ---')
await page.fill('textarea', 'I am going to the store tomorrow')
await page.click('.translate-btn')
await page.waitForFunction(
  () => {
    const tags = Array.from(document.querySelectorAll('.gloss-panel .gloss-text'))
      .map((n) => n.textContent)
    return tags[0] === 'TOMORROW'
  },
  null,
  { timeout: 8000 }
)
const step1 = await page.evaluate(() => ({
  glosses: Array.from(document.querySelectorAll('.gloss-panel .gloss-text')).map((n) => n.textContent),
  inputShown: document.querySelector('.gloss-panel-input p')?.textContent || null,
  showcaseVideoPresent: !!document.querySelector('.showcase-row video')
}))
console.log('step1:', step1)

console.log('\n--- Step 2: showcase (glosses must not change) ---')
const glossesBeforeShowcase = step1.glosses.join(',')
await page.click('.showcase-btn')
await page.waitForFunction(
  () => {
    const v = document.querySelector('.showcase-row video.sign-video')
    return v && v.readyState >= 2 && v.duration > 0
  },
  null,
  { timeout: 15000 }
)
const step2 = await page.evaluate((prev) => {
  const v = document.querySelector('.showcase-row video.sign-video')
  const showcaseRow = document.querySelector('.showcase-row')
  const topRow = document.querySelector('.top-row')
  const gloss = document.querySelector('.gloss-panel')
  const textInput = document.querySelector('.text-input')
  const topRect = topRow?.getBoundingClientRect()
  const showcaseRect = showcaseRow?.getBoundingClientRect()
  const glossRect = gloss?.getBoundingClientRect()
  const textRect = textInput?.getBoundingClientRect()
  const currentGlosses = Array.from(document.querySelectorAll('.gloss-panel .gloss-text')).map((n) => n.textContent).join(',')
  return {
    videoSrc: v?.currentSrc,
    videoW: v?.videoWidth,
    videoH: v?.videoHeight,
    duration: v?.duration,
    hasAudio: !!v && v.mozHasAudio !== false,
    glossesUnchanged: currentGlosses === prev,
    currentGlosses,
    layout: {
      glossRightOfText: glossRect && textRect ? glossRect.left >= textRect.right - 1 : null,
      showcaseBelowTop: showcaseRect && topRect ? showcaseRect.top >= topRect.bottom - 1 : null
    }
  }
}, glossesBeforeShowcase)
console.log('step2:', step2)

await page.screenshot({ path: '/tmp/verify-demo.png', fullPage: true })

if (consoleErrors.length) {
  console.log('\n--- Console errors (unexpected) ---')
  for (const e of consoleErrors) console.log(' ', e)
}

await browser.close()

const step1Ok =
  step1.glosses.length > 0 && step1.glosses[0] === 'TOMORROW' && !step1.showcaseVideoPresent
const step2Ok =
  step2.videoSrc && /showcase\.mp4/.test(step2.videoSrc) &&
  step2.duration > 0.1 && step2.videoW > 0 &&
  step2.glossesUnchanged &&
  step2.layout.glossRightOfText && step2.layout.showcaseBelowTop

console.log('\n--- Verdict ---')
console.log('nlp step   :', step1Ok ? 'OK' : 'FAIL')
console.log('showcase   :', step2Ok ? 'OK' : 'FAIL')
process.exit(step1Ok && step2Ok ? 0 : 1)
