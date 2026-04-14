import React, { useRef, useEffect, useState } from 'react'
import GlossAnimation from './GlossAnimation'
import './VideoPanel.css'

function VideoPanel({ videoUrl, isTranslating, glosses, confidence, isShowcase }) {
  const videoRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (videoUrl && videoRef.current) {
      setError(null)
      videoRef.current.load()
      videoRef.current.play().catch(() => { /* autoplay may be blocked */ })
    }
  }, [videoUrl])

  const handleReplay = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0
      videoRef.current.play()
    }
  }

  // Non-showcase translation: show gloss sequence only (no video).
  // The live demo surface is the grammar engine; the 3D avatar is the showcase.
  if (!videoUrl) {
    return (
      <div className="gloss-only-display">
        <div className="gloss-only-note">
          NLP grammar engine output — see the 3D avatar showcase for a rendered example.
        </div>
        <GlossAnimation glosses={glosses} confidence={confidence} />
      </div>
    )
  }

  return (
    <div className="video-panel">
      <div className="video-header">
        <h2>{isShowcase ? '3D Avatar Showcase' : 'ASL Translation'}</h2>
        <div className="video-status">
          {isTranslating && <span className="status-badge translating">Translating...</span>}
          {isPlaying && <span className="status-badge playing">Playing</span>}
          {!isTranslating && !isPlaying && <span className="status-badge ready">Ready</span>}
        </div>
      </div>

      <div className="video-container">
        <video
          ref={videoRef}
          className="sign-video"
          controls
          playsInline
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onEnded={() => setIsPlaying(false)}
          onError={() => { setError('Failed to load video'); setIsPlaying(false) }}
          src={videoUrl}
        />
      </div>

      {error && <div className="video-error"><p>{error}</p></div>}

      <div className="video-actions">
        <button
          className="action-btn replay-btn"
          onClick={handleReplay}
          disabled={!videoUrl || isTranslating}
        >
          Replay
        </button>
      </div>

      <div className="video-info">
        {isShowcase
          ? <p>Pre-rendered SMPL-X 3D avatar from the SignAvatars motion-capture dataset</p>
          : <p>Video generated server-side</p>}
      </div>
    </div>
  )
}

export default VideoPanel
