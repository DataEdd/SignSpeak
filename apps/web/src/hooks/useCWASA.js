import { useState, useEffect, useCallback, useRef } from 'react'

const SCRIPT_WAIT_MS = 15000
const POLL_INTERVAL_MS = 250

function waitForCWASA(timeoutMs) {
  return new Promise((resolve, reject) => {
    const start = Date.now()
    const tick = () => {
      if (typeof window.CWASA !== 'undefined' && typeof window.CWASA.init === 'function') {
        resolve(window.CWASA)
        return
      }
      if (Date.now() - start > timeoutMs) {
        reject(new Error('CWASA script did not load'))
        return
      }
      setTimeout(tick, POLL_INTERVAL_MS)
    }
    tick()
  })
}

// CWASA is a singleton — init exactly once per page. React 18 StrictMode
// double-mounts effects in dev, so a component-level ref would be flipped by
// the first mount's cleanup and the real mount would skip init.
let cwasaInitStarted = false

export function useCWASA() {
  const [isReady, setIsReady] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [error, setError] = useState(null)
  const avatarIndexRef = useRef(0)

  useEffect(() => {
    let cancelled = false

    waitForCWASA(SCRIPT_WAIT_MS)
      .then((CWASA) => {
        if (cancelled) return

        if (cwasaInitStarted && CWASA.ready) {
          CWASA.ready.then(() => { if (!cancelled) setIsReady(true) })
          CWASA.addHook('animactive', () => { if (!cancelled) setIsPlaying(true) })
          CWASA.addHook('animidle', () => { if (!cancelled) setIsPlaying(false) })
          return
        }

        CWASA.addHook('avatarready', () => { if (!cancelled) setIsReady(true) })
        CWASA.addHook('animactive', () => { if (!cancelled) setIsPlaying(true) })
        CWASA.addHook('animidle', () => { if (!cancelled) setIsPlaying(false) })

        try {
          cwasaInitStarted = true
          CWASA.init({
            avSettings: [{
              width: 380,
              height: 380,
              initAv: 'anna',
              allowSiGMLText: true
            }]
          })
        } catch (err) {
          console.error('CWASA.init threw:', err)
          if (!cancelled) setError('Avatar init failed')
        }
      })
      .catch((err) => {
        console.error(err)
        if (!cancelled) setError('CWASA failed to load')
      })

    return () => { cancelled = true }
  }, [])

  const playSigml = useCallback((sigmlText) => {
    if (!sigmlText) return
    try {
      if (typeof window.CWASA?.playSiGMLText === 'function') {
        window.CWASA.playSiGMLText(sigmlText, avatarIndexRef.current)
        setIsPlaying(true)
      } else {
        setError('CWASA.playSiGMLText unavailable')
      }
    } catch (err) {
      console.error('playSiGMLText threw:', err)
      setError('Failed to play sign')
      setIsPlaying(false)
    }
  }, [])

  const stop = useCallback(() => {
    try {
      if (typeof window.CWASA?.stopSiGML === 'function') {
        window.CWASA.stopSiGML(avatarIndexRef.current)
      }
      setIsPlaying(false)
    } catch (err) {
      console.error('stopSiGML threw:', err)
    }
  }, [])

  return { isReady, isPlaying, error, playSigml, stop }
}
