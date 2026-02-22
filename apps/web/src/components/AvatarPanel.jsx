import React, { useEffect, useState } from 'react'
import { useCWASA } from '../hooks/useCWASA'
import './AvatarPanel.css'

function AvatarPanel({ sigml, speed = 1.0, onLoadFailed }) {
  const { isReady, isPlaying, error, playSigml, stop } = useCWASA(speed)
  const [showFallback, setShowFallback] = useState(false)

  useEffect(() => {
    if (sigml && isReady) {
      playSigml(sigml)
    }
  }, [sigml, isReady, playSigml])

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
        <h2>3D Signing Avatar</h2>
        <div className="avatar-status">
          {isPlaying && <span className="status-badge signing">Signing</span>}
          {!isReady && !showFallback && <span className="status-badge loading">Loading...</span>}
          {isReady && !isPlaying && <span className="status-badge ready">Ready</span>}
        </div>
      </div>

      <div className="avatar-container">
        <div id="CWASAAvatar" className="cwasa-avatar-wrapper">
          {/* CWASA renders the WebGL avatar here */}
        </div>
      </div>

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
          Stop Signing
        </button>
      </div>

      <div className="avatar-info">
        <p>Powered by CWASA (University of East Anglia)</p>
      </div>
    </div>
  )
}

export default AvatarPanel
