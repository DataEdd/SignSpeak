import React, { useState, useEffect, useRef } from 'react'
import './GlossAnimation.css'

const GLOSS_INTERVAL_MS = 700

function GlossAnimation({ glosses, confidence }) {
  const [activeIndex, setActiveIndex] = useState(-1)
  const [isAnimating, setIsAnimating] = useState(false)
  const timerRef = useRef(null)

  useEffect(() => {
    setActiveIndex(-1)
    setIsAnimating(false)
    clearInterval(timerRef.current)

    if (!glosses || glosses.length === 0) return

    const startTimer = setTimeout(() => {
      setIsAnimating(true)
      let idx = 0
      setActiveIndex(0)

      timerRef.current = setInterval(() => {
        idx += 1
        if (idx >= glosses.length) {
          clearInterval(timerRef.current)
          setIsAnimating(false)
          return
        }
        setActiveIndex(idx)
      }, GLOSS_INTERVAL_MS)
    }, 300)

    return () => {
      clearTimeout(startTimer)
      clearInterval(timerRef.current)
    }
  }, [glosses])

  const replay = () => {
    setActiveIndex(-1)
    setIsAnimating(false)
    clearInterval(timerRef.current)

    setTimeout(() => {
      setIsAnimating(true)
      let idx = 0
      setActiveIndex(0)

      timerRef.current = setInterval(() => {
        idx += 1
        if (idx >= glosses.length) {
          clearInterval(timerRef.current)
          setIsAnimating(false)
          return
        }
        setActiveIndex(idx)
      }, GLOSS_INTERVAL_MS)
    }, 200)
  }

  if (!glosses || glosses.length === 0) {
    return (
      <div className="gloss-animation">
        <div className="gloss-animation-empty">
          <div className="empty-icon">&#x1F91F;</div>
          <p>Enter text and translate to see ASL gloss sequence</p>
        </div>
      </div>
    )
  }

  return (
    <div className="gloss-animation">
      <div className="gloss-animation-header">
        <h3>ASL Gloss Sequence</h3>
        {confidence != null && (
          <span className="confidence-badge">
            {Math.round(confidence * 100)}% confidence
          </span>
        )}
      </div>

      <div className="gloss-cards">
        {glosses.map((gloss, index) => (
          <div
            key={`${gloss}-${index}`}
            className={`gloss-card ${
              index === activeIndex ? 'active' : ''
            } ${index < activeIndex ? 'completed' : ''} ${
              index > activeIndex ? 'pending' : ''
            }`}
          >
            <span className="gloss-index">{index + 1}</span>
            <span className="gloss-text">{gloss}</span>
          </div>
        ))}
      </div>

      <div className="gloss-animation-footer">
        <button
          className="replay-btn"
          onClick={replay}
          disabled={isAnimating}
        >
          Replay Glosses
        </button>
      </div>
    </div>
  )
}

export default GlossAnimation
