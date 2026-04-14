import React, { useRef, useEffect, useState } from 'react'
import './VideoPanel.css'

function VideoPanel({ videoUrl, isShowcase, onClose }) {
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

  if (!videoUrl) return null

  const handleReplay = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0
      videoRef.current.play()
    }
  }

  return (
    <div className="video-panel">
      <div className="video-header">
        <h2>{isShowcase ? '3D Avatar Showcase' : 'ASL Translation'}</h2>
        <div className="video-header-right">
          {isPlaying && <span className="status-badge playing">Playing</span>}
          {!isPlaying && <span className="status-badge ready">Ready</span>}
          {onClose && (
            <button className="video-close-btn" onClick={onClose} aria-label="Close">×</button>
          )}
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
          disabled={!videoUrl}
        >
          Replay
        </button>
      </div>

      <div className="video-info">
        {isShowcase
          ? <p>Pre-rendered SMPL-X 3D avatar signing the NWS winter-storm news broadcast. Captions synced below the avatar.</p>
          : <p>Video generated server-side</p>}
      </div>
    </div>
  )
}

export default VideoPanel
