// Record one MP4 per preloaded example phrase.
//
// Strategy: record each unique gloss (single-sign SiGML) into a per-gloss cache,
// then concatenate the relevant clips with ffmpeg for each phrase.
// Per-gloss recording sidesteps CWASA's flaky multi-sign SiGML parser,
// and ffmpeg re-muxes webm→mp4 with proper duration metadata.
//
// Requires: ffmpeg on PATH, `npm run dev` running at localhost:5173.

import { chromium } from 'playwright'
import { execFileSync } from 'node:child_process'
import { writeFileSync, mkdirSync, existsSync, unlinkSync, statSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const OUT_DIR = resolve(__dirname, '../public/videos/examples')
const CACHE_DIR = resolve(__dirname, '../.cache/gloss-clips')
mkdirSync(OUT_DIR, { recursive: true })
mkdirSync(CACHE_DIR, { recursive: true })

const URL_BASE = process.env.CWASA_URL || 'http://localhost:5173/'
const PER_GLOSS_TIMEOUT_MS = 20000
const MIN_ACTIVE_WAIT_MS = 3000
const TAIL_BUFFER_MS = 600

async function recordGlossClip(page, gloss) {
  const webmPath = join(CACHE_DIR, `${gloss}.webm`)
  if (existsSync(webmPath) && statSync(webmPath).size > 1000) {
    return webmPath
  }

  const sigml = await page.evaluate(async (g) => {
    const mod = await import('/src/api/sigmlLookup.js')
    return mod.getSigmlForGloss(g)
  }, gloss)

  if (!sigml) {
    console.log(`  skipping ${gloss} (not in dictionary)`)
    return null
  }

  const result = await page.evaluate(
    async ({ gloss, sigml, timeoutMs, minActiveMs, tailMs }) => {
      const tag = `[rec ${gloss}]`
      return await new Promise((resolve) => {
        const canvas = document.querySelector('.CWASAAvatar canvas')
        if (!canvas?.captureStream) { resolve({ ok: false, reason: 'no-canvas' }); return }

        const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
          .find((t) => { try { return MediaRecorder.isTypeSupported(t) } catch { return false } })
        if (!mimeType) { resolve({ ok: false, reason: 'no-mimetype' }); return }

        const stream = canvas.captureStream(30)
        const rec = new MediaRecorder(stream, { mimeType })
        const chunks = []
        rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data) }

        let sawActive = false
        let activeAt = 0
        let stopping = false
        let resolved = false

        const finish = (obj) => {
          if (resolved) return
          resolved = true
          try { window.CWASA.removeHook?.('animactive', onActive) } catch {}
          try { window.CWASA.removeHook?.('animidle', onIdle) } catch {}
          resolve(obj)
        }

        const maybeStop = (reason) => {
          if (stopping) return
          stopping = true
          console.log(`${tag} maybeStop(${reason})`)
          setTimeout(() => {
            try {
              if (rec.state !== 'inactive') rec.stop()
            } catch (e) {
              console.log(`${tag} rec.stop threw: ${e.message}`)
              finish({ ok: false, reason: 'stop-threw' })
            }
          }, tailMs)
        }

        const onActive = (ev) => {
          if (ev?._gloss !== gloss && sawActive) return
          sawActive = true
          activeAt = Date.now()
          console.log(`${tag} animactive`)
        }
        const onIdle = () => {
          if (!sawActive) return
          console.log(`${tag} animidle (sawActive=${sawActive})`)
          const elapsed = Date.now() - activeAt
          const wait = Math.max(0, minActiveMs - elapsed)
          setTimeout(() => maybeStop('idle'), wait)
        }
        window.CWASA.addHook('animactive', onActive)
        window.CWASA.addHook('animidle', onIdle)

        rec.onstop = async () => {
          try {
            console.log(`${tag} rec.onstop fired, chunks=${chunks.length}`)
            const blob = new Blob(chunks, { type: mimeType })
            const buf = await blob.arrayBuffer()
            // Chunked base64 to avoid argument-count limits on String.fromCharCode
            const bytes = new Uint8Array(buf)
            let binary = ''
            const SZ = 0x8000
            for (let i = 0; i < bytes.length; i += SZ) {
              binary += String.fromCharCode.apply(null, bytes.subarray(i, i + SZ))
            }
            const b64 = btoa(binary)
            console.log(`${tag} encoded ${buf.byteLength}B`)
            finish({ ok: true, b64, bytes: buf.byteLength })
          } catch (e) {
            console.log(`${tag} onstop encode failed: ${e.message}`)
            finish({ ok: false, reason: `encode-failed: ${e.message}` })
          }
        }
        rec.onerror = (e) => {
          console.log(`${tag} rec.onerror: ${e?.error?.message || 'unknown'}`)
          finish({ ok: false, reason: 'recorder-error' })
        }

        rec.start(100)
        const start = Date.now()
        try {
          window.CWASA.playSiGMLText(sigml, 0)
          console.log(`${tag} playSiGMLText called`)
        } catch (e) {
          console.log(`${tag} play threw: ${e.message}`)
          finish({ ok: false, reason: 'play-threw' })
          return
        }

        const watchdog = setInterval(() => {
          if (resolved || rec.state === 'inactive') { clearInterval(watchdog); return }
          const elapsed = Date.now() - start
          if (!sawActive && elapsed > 4000) { clearInterval(watchdog); maybeStop('no-active') }
          else if (elapsed > timeoutMs) { clearInterval(watchdog); maybeStop('timeout') }
        }, 250)

        // Hard backstop — force resolve if nothing has completed after 2× timeout
        setTimeout(() => {
          if (!resolved) {
            console.log(`${tag} hard backstop hit`)
            finish({ ok: false, reason: 'hard-backstop' })
          }
        }, timeoutMs * 2)
      })
    },
    { gloss, sigml, timeoutMs: PER_GLOSS_TIMEOUT_MS, minActiveMs: MIN_ACTIVE_WAIT_MS, tailMs: TAIL_BUFFER_MS }
  )

  if (!result.ok || result.bytes < 1000) {
    console.log(`  ${gloss}: FAIL (${result.reason || 'tiny-file'})`)
    return null
  }
  writeFileSync(webmPath, Buffer.from(result.b64, 'base64'))
  console.log(`  ${gloss}: ${(result.bytes / 1024).toFixed(0)}KB`)
  return webmPath
}

const MIN_CLIP_SECONDS = 1.8

function normaliseClipToMp4(webmPath, mp4Path) {
  // MediaRecorder webms have no duration metadata and variable FPS. Force 30fps,
  // reset timestamps, and hold the final frame so short signs aren't blink-and-miss.
  execFileSync('ffmpeg', [
    '-y', '-hide_banner', '-loglevel', 'error',
    '-i', webmPath,
    '-vf', `fps=30,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=${MIN_CLIP_SECONDS}`,
    '-t', String(MIN_CLIP_SECONDS + 0.5),
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23',
    '-movflags', '+faststart',
    '-an',
    mp4Path
  ])
}

function concatAndMux(clipPaths, outMp4) {
  // Normalise each gloss clip to a fixed-FPS MP4, then concat them.
  const normPaths = clipPaths.map((p, i) => {
    const np = p.replace(/\.webm$/, `.norm.mp4`)
    normaliseClipToMp4(p, np)
    return np
  })

  if (normPaths.length === 1) {
    execFileSync('cp', [normPaths[0], outMp4])
    return
  }

  const listPath = join(CACHE_DIR, '_concat.txt')
  writeFileSync(listPath, normPaths.map((p) => `file '${p}'`).join('\n'))
  execFileSync('ffmpeg', [
    '-y', '-hide_banner', '-loglevel', 'error',
    '-f', 'concat', '-safe', '0', '-i', listPath,
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23',
    '-movflags', '+faststart',
    '-an',
    outMp4
  ])
  unlinkSync(listPath)
}

const browser = await chromium.launch({
  headless: true,
  args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist']
})
const ctx = await browser.newContext()
const page = await ctx.newPage()

page.on('console', (m) => {
  const t = m.text()
  if (/\[rec |playSiGMLText|avatarready|animactive|animidle|rec\./i.test(t)) {
    console.log('  ', t)
  } else if (m.type() === 'error' && !/vite|React DevTools|500 \(Internal/.test(t)) {
    console.log('  [browser err]', t)
  }
})

await page.goto(URL_BASE, { waitUntil: 'domcontentloaded' })

// Warm up CWASA via the app's normal flow with a non-preloaded phrase
await page.waitForSelector('textarea', { timeout: 10000 })
await page.fill('textarea', 'warmup phrase not in examples')
await page.click('.translate-btn')
await page.waitForFunction(
  () => !!document.querySelector('.CWASAAvatar canvas') && !!window.CWASA?.ready,
  null,
  { timeout: 25000 }
)
// Wait for first avatarready
await page.waitForFunction(
  () => {
    if (window.__avReady) return true
    if (window.CWASA?.addHook && !window.__avArmed) {
      window.__avArmed = true
      window.CWASA.addHook('avatarready', () => { window.__avReady = true })
    }
    return !!window.__avReady
  },
  null,
  { timeout: 25000 }
)
console.log('CWASA ready.\n')

const { examples } = await page.evaluate(async () => {
  const ex = await import('/src/data/examples.js')
  return { examples: ex.EXAMPLES }
})

const allGlosses = [...new Set(examples.flatMap((e) => e.glosses))]
console.log(`Recording ${allGlosses.length} unique glosses...`)
const glossClip = {}
for (const g of allGlosses) {
  const p = await recordGlossClip(page, g)
  glossClip[g] = p
}

console.log(`\nConcatenating ${examples.length} phrases...`)
const results = []
for (const ex of examples) {
  const clips = ex.glosses.map((g) => glossClip[g]).filter(Boolean)
  if (clips.length === 0) {
    console.log(`  ${ex.slug}: no clips available, skipping`)
    results.push({ slug: ex.slug, ok: false })
    continue
  }
  const outPath = join(OUT_DIR, `${ex.slug}.mp4`)
  try {
    concatAndMux(clips, outPath)
    const sz = statSync(outPath).size
    console.log(`  ${ex.slug}.mp4: ${(sz / 1024).toFixed(0)}KB (${clips.length}/${ex.glosses.length} signs)`)
    results.push({ slug: ex.slug, ok: true, bytes: sz, present: clips.length, total: ex.glosses.length })
  } catch (err) {
    console.log(`  ${ex.slug}: ffmpeg failed — ${err.message.slice(0, 200)}`)
    results.push({ slug: ex.slug, ok: false })
  }
}

// Clean up the old webm output from a previous run, if any
for (const ex of examples) {
  const old = join(OUT_DIR, `${ex.slug}.webm`)
  if (existsSync(old)) unlinkSync(old)
}

await browser.close()

console.log('\n--- Summary ---')
console.table(results)
const failed = results.filter((r) => !r.ok).length
process.exit(failed > 0 ? 1 : 0)
