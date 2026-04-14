import React, { useRef, useEffect, useState, useCallback } from 'react'
import GlossAnimation from './GlossAnimation'
import AvatarPanel from './AvatarPanel'
import './VideoPanel.css'

function VideoPanel({ videoUrl, isTranslating, isDemo, glosses, confidence, sigml, isPreloaded }) {
  const videoRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [error, setError] = useState(null)
  const [avatarFailed, setAvatarFailed] = useState(false)
  const [recordedVideoUrl, setRecordedVideoUrl] = useState(null)

  const handleAvatarLoadFailed = useCallback(() => {
    setAvatarFailed(true)
  }, [])

  const handleVideoRecorded = useCallback((url) => {
    setRecordedVideoUrl(url)
  }, [])

  // Clear recorded video when new translation starts
  useEffect(() => {
    if (isTranslating) {
      setRecordedVideoUrl(null)
    }
  }, [isTranslating])

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

  // Demo mode: show Avatar (which records video) + GlossAnimation.
  // This path renders non-preloaded input through a best-effort CWASA avatar —
  // NLP ASL grammar works, but individual sign rendering is approximate.
  if (isDemo && !videoUrl) {
    return (
      <div className="demo-display">
        <div className="approximation-banner">
          Avatar is a best-effort approximation. For sign-accurate ASL, try a preloaded example phrase.
        </div>
        {!avatarFailed && (
          <AvatarPanel
            sigml={sigml}
            onLoadFailed={handleAvatarLoadFailed}
            onVideoRecorded={handleVideoRecorded}
          />
        )}
        <GlossAnimation glosses={glosses} confidence={confidence} />
      </div>
    )
  }

  // Real backend video or recorded avatar video
  const activeVideoUrl = videoUrl || recordedVideoUrl

  return (
    <div className="video-panel">
      <div className="video-header">
        <h2>ASL Translation</h2>
        <div className="video-status">
          {isTranslating && <span className="status-badge translating">Translating...</span>}
          {isPlaying && <span className="status-badge playing">Playing</span>}
          {!isTranslating && !isPlaying && activeVideoUrl && <span className="status-badge ready">Ready</span>}
        </div>
      </div>

      <div className="video-container">
        {activeVideoUrl ? (
          <video
            ref={videoRef}
            className="sign-video"
            controls
            playsInline
            onPlay={handlePlay}
            onPause={handlePause}
            onEnded={handleEnded}
            onError={handleError}
            src={activeVideoUrl}
          />
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
          disabled={!activeVideoUrl || isTranslating}
        >
          Replay
        </button>
      </div>

      <div className="video-info">
        {isPreloaded
          ? <p>Preloaded example — sign-accurate ASL clip</p>
          : <p>Powered by SignSpeak NLP + CWASA avatar pipeline</p>}
      </div>
    </div>
  )
}

export default VideoPanel
