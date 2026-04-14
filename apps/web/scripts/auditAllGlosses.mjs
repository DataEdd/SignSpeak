// Audit all 131 SiGML entries and print which animate in CWASA.
import { chromium } from 'playwright'

const browser = await chromium.launch({
  headless: true,
  args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist']
})
const page = await (await browser.newContext()).newPage()
page.on('console', () => {})

await page.goto('http://localhost:5173/', { waitUntil: 'domcontentloaded' })
await page.waitForSelector('textarea')
await page.fill('textarea', 'warmup')
await page.click('.translate-btn')
await page.waitForFunction(() => document.querySelector('.CWASAAvatar canvas')?.width > 0, null, { timeout: 25000 })
await page.waitForTimeout(3000)

// Prime CWASA with known-good SiGML
const scotlandSigml = await fetch('https://vhg.cmp.uea.ac.uk/tech/jas/vhg2026/sigml/scotland-H.sigml').then((r) => r.text())
await page.evaluate((s) => window.CWASA.playSiGMLText(s, 0), scotlandSigml)
await page.waitForTimeout(3000)

const allGlosses = await page.evaluate(async () => {
  const mod = await import('/src/data/sigmlDictionary.js')
  return Object.keys(mod.SIGML_SIGNS).sort()
})
console.log(`Auditing ${allGlosses.length} glosses (this takes ~10 min)...\n`)

const working = []
const broken = []
for (let i = 0; i < allGlosses.length; i++) {
  const gloss = allGlosses[i]
  const verdict = await page.evaluate(
    async (g) => {
      const mod = await import('/src/api/sigmlLookup.js')
      const sigml = mod.getSigmlForGloss(g)
      if (!sigml) return { sawActive: false, sawInvalid: true }
      return await new Promise((resolve) => {
        let sawActive = false, sawInvalid = false, activeAt = 0, idleAt = 0
        const onActive = () => { sawActive = true; activeAt = Date.now() }
        const onIdle = () => { idleAt = Date.now() }
        const onStatus = (ev) => { if (ev?.msg && /invalid|Errors/i.test(ev.msg)) sawInvalid = true }
        window.CWASA.addHook('animactive', onActive)
        window.CWASA.addHook('animidle', onIdle)
        window.CWASA.addHook('status', onStatus)
        const start = Date.now()
        window.CWASA.playSiGMLText(sigml, 0)
        setTimeout(() => resolve({
          sawActive, sawInvalid,
          durationMs: sawActive && idleAt > activeAt ? idleAt - activeAt : null
        }), 3500)
      })
    },
    gloss
  )
  const ok = verdict.sawActive && !verdict.sawInvalid
  if (ok) working.push({ gloss, durationMs: verdict.durationMs })
  else broken.push(gloss)
  const marker = ok ? '✓' : '✗'
  process.stdout.write(`${marker} ${gloss}${i % 4 === 3 ? '\n' : '  '}`)
}

console.log(`\n\n--- Summary ---`)
console.log(`Working: ${working.length}/${allGlosses.length}`)
console.log(`Broken: ${broken.length}`)
console.log(`\nWorking glosses:`)
console.log(working.map((w) => `  ${w.gloss} (${w.durationMs}ms)`).join('\n'))
console.log(`\nBroken glosses:`)
console.log(broken.join(', '))

await browser.close()
