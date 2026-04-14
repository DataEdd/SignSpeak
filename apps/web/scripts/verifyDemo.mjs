// End-to-end smoke test of the built demo.
//   1. The showcase button plays the bundled 3D avatar MP4
//   2. Arbitrary input produces glosses via the client-side NLP engine
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

const network = []
page.on('response', (r) => {
  if (/videos\/showcase|\.mp4/i.test(r.url())) network.push(`${r.status()} ${r.url()}`)
})

await page.goto(BASE, { waitUntil: 'domcontentloaded' })

console.log('--- Step 1: showcase button plays bundled video ---')
await page.waitForSelector('.showcase-btn', { timeout: 10000 })
await page.click('.showcase-btn')

await page.waitForFunction(
  () => {
    const v = document.querySelector('video.sign-video')
    return v && v.readyState >= 2 && v.duration > 0
  },
  null,
  { timeout: 15000 }
).catch((e) => console.log('video load failed:', e.message))

const stats = await page.evaluate(() => {
  const v = document.querySelector('video.sign-video')
  const heading = document.querySelector('.video-header h2')?.textContent || null
  const info = document.querySelector('.video-info')?.textContent || null
  return v ? {
    src: v.currentSrc,
    duration: v.duration,
    videoW: v.videoWidth,
    videoH: v.videoHeight,
    error: v.error?.message || null,
    heading,
    info
  } : null
})
console.log('showcase stats:', stats)

console.log('\n--- Step 2: arbitrary input → grammar-only output ---')
await page.fill('textarea', 'I am going to the store tomorrow')
await page.click('.translate-btn')

// Wait until the video element unmounts (videoUrl becomes null) and glosses
// reflect the new input (expected lead token is TOMORROW from time-fronting rule)
await page.waitForFunction(
  () => {
    const tags = Array.from(document.querySelectorAll('.gloss-tag')).map((n) => n.textContent)
    const noVideo = !document.querySelector('video.sign-video')
    return noVideo && tags[0] === 'TOMORROW'
  },
  null,
  { timeout: 8000 }
).catch((e) => console.log('grammar state did not update:', e.message))

const grammarStats = await page.evaluate(() => ({
  glosses: Array.from(document.querySelectorAll('.gloss-tag')).map((n) => n.textContent),
  note: document.querySelector('.gloss-only-note')?.textContent || null,
  videoPresent: !!document.querySelector('video.sign-video'),
  avatarAnchorPresent: !!document.querySelector('.CWASAAvatar')
}))
console.log('grammar stats:', grammarStats)

await page.screenshot({ path: '/tmp/verify-demo.png', fullPage: true })

console.log('\n--- Network ---')
for (const n of network) console.log(' ', n)

if (consoleErrors.length) {
  console.log('\n--- Console errors (unexpected) ---')
  for (const e of consoleErrors) console.log(' ', e)
}

await browser.close()

const showcaseOk =
  stats && stats.duration > 0.1 && !stats.error && stats.videoW > 0 &&
  /showcase\.mp4/i.test(stats.src)
const grammarOk =
  grammarStats.glosses.length > 0 &&
  !grammarStats.videoPresent &&
  !grammarStats.avatarAnchorPresent

console.log('\n--- Verdict ---')
console.log('showcase:', showcaseOk ? 'OK' : 'FAIL')
console.log('grammar :', grammarOk ? 'OK' : 'FAIL')
process.exit(showcaseOk && grammarOk ? 0 : 1)
