import React, { useRef, useEffect, useState, useCallback } from 'react'
import GlossAnimation from './GlossAnimation'
import AvatarPanel from './AvatarPanel'
import './VideoPanel.css'

function VideoPanel({ videoUrl, isTranslating, isDemo, glosses, confidence, sigml }) {
  const videoRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [error, setError] = useState(null)
  const [avatarFailed, setAvatarFailed] = useState(false)

  const handleAvatarLoadFailed = useCallback(() => {
    setAvatarFailed(true)
  }, [])

  useEffect(() => {
    if (videoUrl && videoRef.current) {
      setError(null)
      videoRef.current.load()
      videoRef.current.play().catch(() => {
        // autoplay may be blocked
      })
    }
  }, [videoUrl])

  const handlePlay = () => setIsPlaying(true)
  const handlePause = () => setIsPlaying(false)
  const handleEnded = () => setIsPlaying(false)

  const handleError = () => {
    setError('Failed to load video')
    setIsPlaying(false)
  }

  const handleReplay = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0
      videoRef.current.play()
    }
  }

  // Demo mode: show Avatar + GlossAnimation
  if (isDemo && !videoUrl) {
    return (
      <div className="demo-display">
        {!avatarFailed && (
          <AvatarPanel
            sigml={sigml}
            onLoadFailed={handleAvatarLoadFailed}
          />
        )}
        <GlossAnimation glosses={glosses} confidence={confidence} />
      </div>
    )
  }

  return (
    <div className="video-panel">
      <div className="video-header">
        <h2>ASL Translation</h2>
        <div className="video-status">
          {isTranslating && <span className="status-badge translating">Translating...</span>}
          {isPlaying && <span className="status-badge playing">Playing</span>}
          {!isTranslating && !isPlaying && videoUrl && <span className="status-badge ready">Ready</span>}
        </div>
      </div>

      <div className="video-container">
        {videoUrl ? (
          <video
            ref={videoRef}
            className="sign-video"
            controls
            playsInline
            onPlay={handlePlay}
            onPause={handlePause}
            onEnded={handleEnded}
            onError={handleError}
          >
            <source src={videoUrl} type="video/mp4" />
            Your browser does not support video playback.
          </video>
        ) : (
          <div className="video-placeholder">
            <div className="placeholder-icon">&#x1F91F;</div>
            <p>{isTranslating ? 'Generating sign language video...' : 'Enter text to see ASL translation'}</p>
            {isTranslating && <div className="video-spinner"></div>}
          </div>
        )}
      </div>

      {error && (
        <div className="video-error">
          <p>{error}</p>
        </div>
      )}

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
        <p>Powered by SignSpeak NLP + motion capture pipeline</p>
      </div>
    </div>
  )
}

export default VideoPanel
