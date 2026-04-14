// End-to-end verification of the built demo:
//   1. Preloaded phrase → bundled MP4 plays
//   2. Non-preloaded phrase → CWASA avatar path engages
//
// Assumes `npx vite preview` is running (default localhost:4173 with /SignSpeak/ base).

import { chromium } from 'playwright'

const BASE = process.env.PREVIEW_URL || 'http://localhost:4173/SignSpeak/'

const browser = await chromium.launch({
  headless: true,
  args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist']
})
const ctx = await browser.newContext()
const page = await ctx.newPage()

const consoleLogs = []
const network = []
page.on('console', (m) => consoleLogs.push(`[${m.type()}] ${m.text()}`))
page.on('pageerror', (e) => consoleLogs.push(`[err] ${e.message}`))
page.on('response', (r) => {
  if (/videos\/examples|allcsa|cwasa\.css|avatars\//i.test(r.url())) {
    network.push(`${r.status()} ${r.url()}`)
  }
})

console.log('--- Step 1: preloaded phrase ---')
await page.goto(BASE, { waitUntil: 'domcontentloaded' })
await page.waitForSelector('.phrase-btn')
await page.click('.phrase-btn') // first preloaded phrase

// Preloaded path should render a <video> with the bundled mp4 src
const preloaded = await page.waitForFunction(
  () => {
    const v = document.querySelector('video.sign-video')
    return v ? { src: v.currentSrc || v.src, error: !!v.error } : null
  },
  null,
  { timeout: 15000 }
).then((h) => h.jsonValue())

console.log('preloaded video:', preloaded)

// Wait for it to load + play a little
await page.waitForFunction(
  () => {
    const v = document.querySelector('video.sign-video')
    return v && v.readyState >= 2 && v.duration > 0
  },
  null,
  { timeout: 15000 }
).catch((e) => console.log('preloaded load failed:', e.message))

const preloadedStats = await page.evaluate(() => {
  const v = document.querySelector('video.sign-video')
  if (!v) return null
  return {
    duration: v.duration,
    readyState: v.readyState,
    error: v.error?.message || null,
    videoWidth: v.videoWidth,
    videoHeight: v.videoHeight
  }
})
console.log('preloaded stats:', preloadedStats)

// Look for the preloaded caption
const caption = await page.textContent('.video-info').catch(() => null)
console.log('caption:', caption)

console.log('\n--- Step 2: non-preloaded phrase (CWASA path) ---')
await page.fill('textarea', 'this is a custom sentence not in examples')
await page.click('.translate-btn')

// CWASA should init and render a canvas within 30s
await page.waitForFunction(
  () => {
    const c = document.querySelector('.CWASAAvatar canvas')
    return !!(c && c.width > 0)
  },
  null,
  { timeout: 30000 }
).catch((e) => console.log('CWASA canvas wait failed:', e.message))

const cwasaStats = await page.evaluate(() => {
  const c = document.querySelector('.CWASAAvatar canvas')
  const banner = document.querySelector('.approximation-banner')
  return {
    canvasW: c?.width,
    canvasH: c?.height,
    bannerVisible: !!banner && banner.offsetHeight > 0,
    bannerText: banner?.textContent || null,
    cwasaReady: !!window.CWASA?.ready
  }
})
console.log('cwasa stats:', cwasaStats)

await page.screenshot({ path: '/tmp/verify-demo.png', fullPage: true })

console.log('\n--- Network (relevant) ---')
for (const n of network) console.log('  ', n)

const errorsSeen = consoleLogs.filter((l) => l.startsWith('[err]') || l.startsWith('[error]') && !l.includes('React DevTools'))
console.log('\n--- Errors seen ---')
for (const e of errorsSeen) console.log('  ', e)

await browser.close()

// Exit-code contract: fail loudly if either path is broken
const preloadedOk = preloadedStats && preloadedStats.duration > 0.1 && !preloadedStats.error && preloadedStats.videoWidth > 0
const cwasaOk = cwasaStats.canvasW > 0 && cwasaStats.canvasH > 0 && cwasaStats.bannerVisible
console.log('\n--- Verdict ---')
console.log('preloaded:', preloadedOk ? 'OK' : 'FAIL')
console.log('cwasa:', cwasaOk ? 'OK' : 'FAIL')
process.exit(preloadedOk && cwasaOk ? 0 : 1)
