import { useState, useEffect, useCallback, useRef } from 'react'

const MAX_INIT_RETRIES = 20 // 10 seconds at 500ms intervals

export function useCWASA(speed = 1.0) {
  const [isReady, setIsReady] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [error, setError] = useState(null)
  const avatarIndexRef = useRef(0)
  const initAttemptedRef = useRef(false)
  const retryCountRef = useRef(0)

  useEffect(() => {
    if (initAttemptedRef.current) return
    initAttemptedRef.current = true

    const initCWASA = () => {
      try {
        if (typeof window.CWASA !== 'undefined') {
          if (window.CWASA.isInit && window.CWASA.isInit()) {
            setIsReady(true)
            return
          }

          window.CWASA.init({
            avSettings: [{
              width: 380,
              height: 380,
              initAv: 'anna',
              lod: 1,
              speed: speed,
              allowSiGMLText: true,
              useWebGL: true
            }],
            onAvatarReady: () => {
              setIsReady(true)
              setError(null)
            },
            onPlayStart: () => {
              setIsPlaying(true)
            },
            onPlayEnd: () => {
              setIsPlaying(false)
            },
            onError: (err) => {
              console.error('CWASA error:', err)
              setError('Avatar playback error')
            }
          })
        } else {
          retryCountRef.current += 1
          if (retryCountRef.current < MAX_INIT_RETRIES) {
            setTimeout(initCWASA, 500)
          } else {
            setError('CWASA failed to load')
          }
        }
      } catch (err) {
        console.error('CWASA initialization error:', err)
        setError('Failed to initialize avatar')
      }
    }

    if (document.readyState === 'complete') {
      setTimeout(initCWASA, 100)
    } else {
      window.addEventListener('load', () => setTimeout(initCWASA, 100))
    }
  }, [speed])

  const playSigml = useCallback((sigmlText) => {
    if (!sigmlText) return

    try {
      if (typeof window.CWASA !== 'undefined' && window.CWASA.playSiGMLText) {
        setIsPlaying(true)
        window.CWASA.playSiGMLText(sigmlText, avatarIndexRef.current)
      } else if (typeof window.playSiGMLText === 'function') {
        setIsPlaying(true)
        window.playSiGMLText(sigmlText, avatarIndexRef.current)
      } else {
        setIsPlaying(true)
        setTimeout(() => setIsPlaying(false), 2000)
      }
    } catch (err) {
      console.error('Error playing SiGML:', err)
      setError('Failed to play sign')
      setIsPlaying(false)
    }
  }, [])

  const stop = useCallback(() => {
    try {
      if (typeof window.CWASA !== 'undefined' && window.CWASA.stop) {
        window.CWASA.stop(avatarIndexRef.current)
      }
      setIsPlaying(false)
    } catch (err) {
      console.error('Error stopping avatar:', err)
    }
  }, [])

  const setAvatarSpeed = useCallback((newSpeed) => {
    try {
      if (typeof window.CWASA !== 'undefined' && window.CWASA.setSpeed) {
        window.CWASA.setSpeed(newSpeed, avatarIndexRef.current)
      }
    } catch (err) {
      console.error('Error setting speed:', err)
    }
  }, [])

  useEffect(() => {
    if (isReady) {
      setAvatarSpeed(speed)
    }
  }, [speed, isReady, setAvatarSpeed])

  return { isReady, isPlaying, error, playSigml, stop, setAvatarSpeed }
}
