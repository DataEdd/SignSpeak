import React, { useEffect, useState, useRef, useCallback } from 'react'
import { useCWASA } from '../hooks/useCWASA'
import { useCanvasRecorder } from '../hooks/useCanvasRecorder'
import './AvatarPanel.css'

function findCWASACanvas() {
  return document.querySelector('#CWASAAvatar canvas')
    || document.querySelector('.CWASAvatarPanel canvas')
    || document.querySelector('.cwasa-avatar-wrapper canvas')
}

function AvatarPanel({ sigml, speed = 1.0, onLoadFailed, onVideoRecorded }) {
  const { isReady, isPlaying, error, playSigml, stop } = useCWASA(speed)
  const { isRecording, videoUrl, startRecording, stopRecording, clearRecording } = useCanvasRecorder()
  const [showFallback, setShowFallback] = useState(false)
  const prevPlayingRef = useRef(false)
  const prevSigmlRef = useRef(null)

  // When new sigml arrives, clear old recording and play
  useEffect(() => {
    if (sigml && isReady && sigml !== prevSigmlRef.current) {
      prevSigmlRef.current = sigml
      clearRecording()
      playSigml(sigml)
    }
  }, [sigml, isReady, playSigml, clearRecording])

  // Track play-state transitions to start/stop recording
  useEffect(() => {
    if (isPlaying && !prevPlayingRef.current) {
      // Playback just started — record the canvas
      const canvas = findCWASACanvas()
      if (canvas) startRecording(canvas)
    }
    if (!isPlaying && prevPlayingRef.current) {
      // Playback ended — finalize recording
      stopRecording()
    }
    prevPlayingRef.current = isPlaying
  }, [isPlaying, startRecording, stopRecording])

  // Notify parent when recorded video is available
  useEffect(() => {
    if (videoUrl && onVideoRecorded) {
      onVideoRecorded(videoUrl)
    }
  }, [videoUrl, onVideoRecorded])

  // Fallback timeout if CWASA never loads
  useEffect(() => {
    const timer = setTimeout(() => {
      if (!isReady) {
        setShowFallback(true)
        if (onLoadFailed) onLoadFailed()
      }
    }, 8000)

    if (isReady) clearTimeout(timer)
    return () => clearTimeout(timer)
  }, [isReady, onLoadFailed])

  if (showFallback && !isReady) {
    return (
      <div className="avatar-panel avatar-unavailable">
        <div className="avatar-fallback-msg">
          <span className="fallback-icon">&#x1F916;</span>
          <p>3D Avatar unavailable</p>
          <p className="fallback-hint">CWASA service could not be reached</p>
        </div>
      </div>
    )
  }

  return (
    <div className="avatar-panel">
      <div className="avatar-header">
        <h2>ASL Translation</h2>
        <div className="avatar-status">
          {isPlaying && <span className="status-badge signing">Signing...</span>}
          {isRecording && <span className="status-badge recording">Recording</span>}
          {!isReady && !showFallback && <span className="status-badge loading">Loading Avatar...</span>}
          {isReady && !isPlaying && !videoUrl && <span className="status-badge ready">Ready</span>}
          {videoUrl && !isPlaying && <span className="status-badge ready">Video Ready</span>}
        </div>
      </div>

      {/* Always keep CWASA element in DOM; hide when showing recorded video */}
      <div className="avatar-container" style={{ display: videoUrl && !isPlaying ? 'none' : undefined }}>
        <div id="CWASAAvatar" className="cwasa-avatar-wrapper" />
      </div>

      {/* Show recorded video when available and avatar is not actively playing */}
      {videoUrl && !isPlaying && (
        <div className="avatar-video-container">
          <video
            className="recorded-sign-video"
            controls
            autoPlay
            playsInline
            src={videoUrl}
          />
        </div>
      )}

      {error && (
        <div className="avatar-error">
          <p>{error}</p>
        </div>
      )}

      <div className="avatar-actions">
        <button
          className="action-btn stop-btn"
          onClick={stop}
          disabled={!isPlaying}
        >
          Stop
        </button>
      </div>

      <div className="avatar-info">
        <p>Powered by CWASA (University of East Anglia)</p>
      </div>
    </div>
  )
}

export default AvatarPanel
